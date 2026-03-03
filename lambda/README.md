# Lambda Deployment — Predictive Alerting Pipeline

This directory contains the AWS Lambda setup for taking our local `src/` pipeline and turning it into a production-ready, event-driven alerting system in the cloud.

```
EventBridge (daily)          EventBridge (every minute)
       │                              │
       ▼                              ▼
 RetrainFunction               InferenceFunction
 ─────────────                 ─────────────────
 1. Fetch CloudWatch           1. Load model from S3
    metric history                (ETag-cached in /tmp)
 2. Preprocess features        2. Fetch last 90 min of
 3. Train LightGBM +              CloudWatch data
    Neural Network             3. Extract window features
 4. Tune thresholds            4. Ensemble predict_proba
 5. Upload artifacts  ──S3──▶  5. Emit custom CW metric
    to S3                      6. If risk ≥ threshold
                                   → publish SNS alert
                                        │
                                        ▼
                                  Email / PagerDuty /
                                  Slack (via SNS sub)
```

---

## Directory Structure

```
lambda/
├── Makefile                  Build, deploy, invoke, tail logs
├── template.yaml             AWS SAM infrastructure definition
├── shared/
│   ├── __init__.py
│   └── features.py           Feature extraction (shared by both Lambdas)
├── retrain/
│   ├── handler.py            Daily retraining Lambda entry point
│   └── requirements.txt
└── inference/
    ├── handler.py            Per-minute inference Lambda entry point
    └── requirements.txt
```

---

## Prerequisites

```bash
pip install aws-sam-cli
aws configure   # or export AWS_PROFILE=your-profile
```

---

## Configuration

Both Lambdas read their configuration from **environment variables**, which are
set via the SAM template parameters or the Lambda console.

### `METRICS_CONFIG`

A JSON array describing which CloudWatch metrics to monitor.

```json
[
  {
    "namespace": "AWS/EC2",
    "metric":    "CPUUtilization",
    "dim_name":  "InstanceId",
    "dim_value": "i-0abc123def456"
  },
  {
    "namespace": "AWS/RDS",
    "metric":    "DatabaseConnections",
    "dim_name":  "DBInstanceIdentifier",
    "dim_value": "prod-postgres"
  },
  {
    "namespace": "AWS/ApplicationELB",
    "metric":    "TargetResponseTime",
    "dim_name":  "LoadBalancer",
    "dim_value": "app/prod-alb/abc123"
  }
]
```

### Other variables

| Variable         | Default                    | Description                              |
|------------------|----------------------------|------------------------------------------|
| `ARTIFACT_BUCKET`| *(required)*               | S3 bucket for model artifacts            |
| `ARTIFACT_PREFIX`| `alerting/models/prod/`    | S3 key prefix                            |
| `LOOKBACK_DAYS`  | `90`                       | Days of history to use for retraining    |
| `SNS_TOPIC_ARN`  | *(required, inference)*    | SNS topic to publish alerts to           |
| `ALERT_MODEL`    | `ensemble`                 | `lgbm`, `nn`, or `ensemble`            |

---

## Deploy

```bash
cd lambda/

# 1. Build the shared layer (Linux arm64 wheels + shared/ code)
make layer

# 2. SAM build + deploy (guided first time)
make deploy ALERT_EMAIL=you@example.com ENV=prod REGION=us-east-1

# 3. Confirm the SNS email subscription in your inbox
```

The first `sam deploy` will prompt for any missing parameters and create
`samconfig.toml` for future runs.

---

## Manual Testing

```bash
# Trigger a retraining run immediately
make invoke-retrain

# Trigger one inference cycle
make invoke-inference

# Tail live logs
make logs-retrain
make logs-inference
```

---

## How it works

### RetrainFunction (Daily, 02:00 UTC)

Every night, this function wakes up to make sure our models stay fresh:
1. It pulls the last `LOOKBACK_DAYS` of 1-minute metrics from CloudWatch for the resources tracked in `METRICS_CONFIG`.
2. It runs the exact same **sliding-window feature extraction** we use locally (mean, std, min, max, last, slope, roc over a 30-step window).
3. It splits the data temporally (70/15/15) and trains our models, applying SMOTE/class-weighting to handle the massive lack of incident data.
4. It sweeps through potential thresholds on the validation set to find the sweet spot for alerting.
5. Finally, it stashes the updated models and thresholds in S3 as both **timestamped** and **`_latest`** files, so we always have an audit trail.
   - `alerting/models/prod/model_lgbm_20260227T020000.pkl`
   - `alerting/models/prod/model_lgbm_latest.pkl`  ← inference always reads this
   - `alerting/models/prod/thresholds_latest.pkl`

### InferenceFunction (Every Minute)

This is the workhorse that runs every minute to keep an eye on things:
1. It quickly checks the `_latest` artifacts in S3. It only downloads them if the **ETag has changed**, which saves a ton of time on warm starts.
2. It grabs the last 90 minutes (our window size plus a 60-point buffer) of live CloudWatch data.
3. It uses a **RunningStats** technique (online EMA normalization, α = 0.01) to gently adapt to slow metric drift, rather than waiting for tomorrow's retraining cycle.
4. It merges the predictions from our models into a single ensemble score.
5. It spits out a custom `AlertingPipeline/IncidentRisk/RiskScore` metric back to CloudWatch, so you can draw fancy dashboards or set standard CW Alarms.
6. If the risk crosses our trained threshold, it fires off an SNS message tagged `HIGH` or `MEDIUM` to wake someone up.

---

## Observability

| Signal | Where |
|--------|-------|
| Lambda invocation metrics | CloudWatch → Lambda namespace |
| Custom risk scores | CloudWatch → `AlertingPipeline/IncidentRisk` |
| Retrain / inference errors | `InferenceErrorAlarm`, `RetrainErrorAlarm` CW Alarms |
| Structured logs | CloudWatch Logs (`/aws/lambda/cloud-metrics-*`) |
| Alert messages | SNS → email / PagerDuty / Slack |

---

## Teardown

```bash
make destroy   # Deletes the CloudFormation stack.
               # S3 bucket is RETAINED (DeletionPolicy: Retain) to preserve model history.
```
