# Lambda Deployment — Predictive Alerting Pipeline

Two AWS Lambda functions that mirror the local `src/` pipeline in a
production-grade, event-driven architecture.

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
    XGBoost                    3. Extract window features
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
| `ALERT_MODEL`    | `ensemble`                 | `lgbm`, `xgb`, or `ensemble`            |

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

### RetrainFunction (daily, 02:00 UTC)

1. Calls `cloudwatch.get_metric_statistics` for each entry in `METRICS_CONFIG`,
   fetching up to `LOOKBACK_DAYS` days of 1-minute averages.
2. Applies the same **sliding-window feature extraction** as `src/preprocess.py`
   (7 features: mean, std, min, max, last, slope, roc over a W=30 look-back).
3. Splits data temporally (70/15/15) and trains both LightGBM and XGBoost with
   `scale_pos_weight` to handle the ~10:1 class imbalance.
4. Sweeps thresholds on the validation set and picks the one maximising F1.
5. Uploads all artifacts to S3 as both **timestamped** and **`_latest`** keys:
   - `alerting/models/prod/model_lgbm_20260227T020000.pkl`
   - `alerting/models/prod/model_lgbm_latest.pkl`  ← inference always reads this
   - `alerting/models/prod/thresholds_latest.pkl`

### InferenceFunction (every minute)

1. Calls `s3.head_object` on the `_latest` artifacts. Only re-downloads if the
   **ETag has changed** — models survive across warm container invocations.
2. Fetches the last `W + 60` minutes of data per metric (60-point buffer guards
   against CloudWatch data latency and missing points).
3. Uses **online EMA normalisation** (`RunningStats`, α = 0.01) so the feature
   scale adapts to slow distribution drift without requiring a full retrain.
4. Computes ensemble probability: `(p_lgbm + p_xgb) / 2`.
5. Emits a `AlertingPipeline/IncidentRisk/RiskScore` CloudWatch metric per
   resource — you can graph this and set independent CloudWatch Alarms on it.
6. Publishes a structured JSON SNS message if `risk ≥ threshold`, with severity
   tagged `HIGH` (risk > threshold + 0.15) or `MEDIUM`.

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
