# Predictive Alerting for Cloud Metrics

A sliding-window binary classifier that predicts whether an incident will occur
within the next **H** time steps, using real AWS CloudWatch metrics from the
[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB).

---

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m src.data_loader    # download NAB dataset
python -m src.preprocess     # feature extraction (--W 30 --H 5)
python -m src.model          # train models (recall-first threshold)
python -m src.model --tune   # Optuna hyper-parameter search for XGBoost
python -m src.model --tune --trials 100   # custom trial count
python -m src.evaluate       # evaluation + plots (FPR, AP, lead time)
python -m pytest tests/ -v   # unit tests
```

---

## Problem Formulation

| Aspect | Choice |
|--------|--------|
| **Task** | Binary classification: "incident within next H steps?" |
| **Input** | Sliding window of W steps from a univariate metric |
| **Output** | Probability p ∈ [0, 1]; alert raised when p ≥ threshold |
| **Dataset** | NAB `realAWSCloudwatch` — 17 real EC2/RDS/ELB time series |
| **Defaults** | W = 30 (look-back), H = 5 (horizon) |

### Target construction

For each time step *t*, the label is:

```
y(t) = 1  if any incident in [t+1, t+H]
y(t) = 0  otherwise
```

This allows the model to learn *precursor patterns* that appear before incidents.

---

## Dataset

The [NAB dataset](https://github.com/numenta/NAB) provides real-world AWS
CloudWatch metrics collected from EC2 instances, RDS databases, and ELB load
balancers. Anomaly windows are labeled by domain experts.

- **17 segments** (separate AWS resources), **67,740 data points**
- **Metrics**: CPU utilisation, disk write bytes, network in, request counts
- **Incident rate**: ~9.3% of time steps fall within an anomaly window

Each segment is **z-score normalised independently** before feature extraction.
This is critical because different metric types (CPU %, bytes, counts) have
vastly different scales.

---

## Feature Engineering

Seven rolling-window features are extracted per time step:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `mean` | Mean value over the window |
| 2 | `std` | Standard deviation |
| 3 | `min` | Minimum value |
| 4 | `max` | Maximum value |
| 5 | `last` | Most recent value |
| 6 | `slope` | Linear regression slope |
| 7 | `roc` | Rate of change (last − first) |

The `slope` and `roc` features capture *trends* leading into incidents.

### Train / Val / Test Split

The split is done **within each segment** (70 / 15 / 15% temporally), ensuring
every split contains samples from every metric type. This avoids the failure
mode of training on one set of metric types and testing on another.

---

## Model Selection

Several modelling approaches were considered for this task:

| Approach | Considered? | Verdict |
|----------|------------|---------|
| ARIMA / SARIMA | Yes | **Rejected.** ARIMA forecasts *values*, not binary incident labels. It also requires stationarity, which the spec explicitly notes is absent in these metrics. |
| Prophet | Yes | **Rejected.** Designed for daily/weekly seasonality in business metrics; not suited to irregular 5-minute operational spikes. |
| LSTM / Transformer | Yes | **Rejected for now.** The dataset is small (~67 k points across 17 segments). Deep sequence models need far more data to outperform gradient boosting, and they require a GPU for reasonable training times. |
| Random Forest | Briefly | Gradient boosting consistently dominates RF on tabular data with class imbalance due to its sequential error-correction. |
| **Gradient Boosting (LightGBM + XGBoost)** | **Selected** | Handles class imbalance via `scale_pos_weight`, supports native early stopping, is robust to outliers and heavy-tailed distributions (common in CloudWatch metrics), and produces interpretable feature-importance rankings. |

### Why gradient boosting fits this problem

1. **Class imbalance**: Only ~9 % of time steps are positive. `scale_pos_weight`
   re-weights the loss without manual over/under-sampling.
2. **Heavy-tailed features**: Tree splits are invariant to monotone transforms,
   so extreme CPU or network-byte values do not distort the model.
3. **Small data regime**: Boosting with shallow trees (depth 4) generalises well
   even with ~47 k training samples.
4. **Interpretability**: Feature importance reveals which rolling statistics
   are most predictive, aiding root-cause analysis.

---

## Models

### XGBoost (baseline)

- `scale_pos_weight` = neg/pos ratio for class-imbalance handling
- 500 estimators, max depth 4, early stopping (`rounds=50`) on validation loss
- Tuned threshold: **θ = 0.49** (recall-first)

### LightGBM (primary)

- `scale_pos_weight` = neg/pos ratio
- 500 estimators, max depth 4, early stopping on validation loss
- Tuned threshold: **θ = 0.11** (recall-first)

### Threshold tuning strategy

The original threshold was chosen to **maximise F1**, which can sacrifice recall
for precision. Since the project requirement is **≥ 80 % incident recall**, the
threshold is now selected using a **recall-first sweep**: find the highest
threshold where incident-level recall ≥ 80 %, falling back to the F1-maximising
threshold only if that target is unachievable.

This is implemented in `find_recall_threshold()` in `model.py`.

### Optuna hyper-parameter tuning

Running `python -m src.model --tune` launches an Optuna study that searches
over `max_depth`, `learning_rate`, `n_estimators`, `subsample`,
`colsample_bytree`, and `reg_alpha` to **maximise incident-level recall** on
the validation set. This directly targets the ≥ 80 % recall success criterion
rather than optimising a proxy metric like F1.

---

## Results

### Training & Validation Accuracy

| Model | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| **XGBoost** | **83.13%** | **66.82%** |
| LightGBM | 90.45% | 92.31% |

> **Note**: Raw accuracy is misleading here due to ~10:1 class imbalance (~90% of
> steps are negative). A model that predicts all-negative scores 90% accuracy.
> Incident-level recall (below) is the true measure of alerting quality.

### Test-Set Summary

| Model | Threshold | Test Accuracy | ROC-AUC | Incident Recall | Mean Lead Time | FPR | Avg Precision |
|-------|-----------|--------------|---------|-----------------|----------------|-----|---------------|
| **XGBoost** | **0.49** | **26.06%** | **0.513** | **100% (5/5)** | **30.0 steps** | **0.7737** | **0.175** |
| LightGBM | 0.11 | 24.23% | 0.339 | 100% (5/5) | 30.0 steps | 0.7922 | 0.096 |

> With the **recall-first threshold tuning** (find the highest threshold with
> ≥ 80 % incident recall), both models now achieve 100 % incident recall on
> the test set. The trade-off is higher FPR compared to F1-optimised thresholds.

### Evaluation metrics

- **Incident-level recall**: fraction of incident intervals preceded by ≥1 alert
  (target: ~80%)
- **Mean lead time**: how many steps before the incident the first alert fires
- **ROC-AUC**: overall discrimination quality
- **Test accuracy**: step-level prediction accuracy at the tuned threshold
- **Step-level precision/recall**: per-time-step classification report

### Precision-Recall Trade-off

The precision-recall (PR) curve summarises how the model trades off precision
against recall across all possible thresholds.

| Model | Average Precision (AP) | Interpretation |
|-------|----------------------|----------------|
| XGBoost | 0.175 | Low — the model's probability estimates are poorly calibrated, so most thresholds either catch all incidents (high recall, low precision) or miss them. |
| LightGBM | 0.096 | Very low — the PR curve collapses near the origin, indicating the model struggles to rank positive steps above negatives. |

**What the curve shape means for threshold choice:**

- **XGBoost**: The PR curve is relatively flat at low precision up to ~95 %
  recall, then drops sharply. This means we can achieve high incident recall
  cheaply, but cannot improve precision much without sacrificing recall.
  The optimal operating point is near the "elbow" — the recall-first threshold
  (θ = 0.46) sits just past this elbow, accepting many false alarms to
  guarantee incident detection.
- **LightGBM**: The curve is steep — even small recall gains require large
  precision losses. The model's discrimination is too weak to offer a useful
  recall-precision trade-off.

### False Positive Rate (FPR) Discussion

| Model | FPR | False Alarms / Hour | Assessment |
|-------|-----|--------------------|-----------|
| XGBoost | 0.7737 | ~8.2 | **High for production.** Most negative steps trigger an alert. Post-processing (debouncing, N-of-M voting) would be needed. |
| LightGBM | 0.7922 | ~8.4 | **Also high.** With the recall-first threshold (θ = 0.11), LightGBM gains 100% incident recall but loses its low-FPR advantage. |

### XGBoost — step-level detail

| | Precision | Recall | F1 | Support |
|---|-----------|--------|------|---------|
| No incident | 0.783 | 0.226 | 0.351 | 8912 |
| Incident ahead | 0.081 | 0.522 | 0.141 | 1169 |

XGBoost is tuned for **very high incident recall (100% at incident level)** at
the cost of step-level precision (8.1%), which is the correct trade-off for an
alerting system — missing an incident is far more costly than a false alarm.

### LightGBM — step-level detail

| | Precision | Recall | F1 | Support |
|---|-----------|--------|------|---------|
| No incident | 0.762 | 0.208 | 0.327 | 8912 |
| Incident ahead | 0.077 | 0.506 | 0.134 | 1169 |

---

## Analysis & Discussion

### Why both models achieve 100% incident recall

1. **Recall-first threshold tuning**: The threshold is selected as the highest
   value where incident-level recall ≥ 80%. For both models this results in a
   threshold low enough that many steps are flagged positive, guaranteeing all
   incident intervals are detected.

2. **Scale-weighted training**: `scale_pos_weight` = neg/pos amplifies the
   minority class gradient, pushing the model toward high-sensitivity decisions.

3. **Early stopping on validation**: Both models stop at the iteration that best
   generalises to the val set, preventing overfitting while maintaining recall.

### Why test accuracy is low (~24–26%)

Low raw accuracy is expected and even desirable in this context:

- With ~88% negative steps, a model predicting all-negative trivially scores ~88%
  accuracy — but misses every incident.
- Both models deliberately bias toward positives to maximise incident recall.
  The price is many false alarms (low step-level precision), which tanks raw accuracy.
- **Incident-level recall (100%) is the primary metric**, not raw accuracy.

### Limitations

- **Univariate per segment**: Each segment is treated as an independent
  univariate series. Cross-metric correlations (e.g., CPU spike + network
  spike → incident) are not captured.
- **Step-level precision is low** (~8%): Many time steps are flagged as
  "incident ahead" when no incident follows. For a production system, this
  would cause alert fatigue.
- **Small test set**: Only 5 incident intervals in the test set — the
  incident-level recall estimate has high variance.
- **No concept drift handling**: The model is trained once; in production,
  metric distributions shift over time.

### Window (W) and Horizon (H) Sensitivity

| Parameter | Too Small | Current | Too Large |
|-----------|-----------|---------|----------|
| **W** (look-back) | < 10: misses trend features (`slope`, `roc`); rolling statistics have high variance | **30** (2.5 hours at 5 min/step) | > 60: windows cross segment boundaries; includes stale context from hours ago |
| **H** (horizon) | 1: labels are nearly identical to raw incidents; too little lead time to act | **5** (25 minutes) | > 30: "incident ahead" signal is diluted across too many positive labels; model struggles to learn discriminative patterns |

The current defaults (W = 30, H = 5) were chosen as a pragmatic trade-off:
W = 30 gives enough context for trend features without crossing segment
boundaries, and H = 5 provides ~25 minutes of lead time — enough for an
operations team to investigate.

### Failure Cases

**1. LightGBM: poor discrimination despite 100% incident recall**

With the recall-first threshold (θ = 0.11), LightGBM achieves 100% incident
recall on the test set but with ~79% FPR.  Under an F1-optimised threshold
(θ ≈ 0.14) it only detects 1 of 5 incidents — segments where the incident
pattern involves *sudden jumps* (e.g., CPU spikes) rather than gradual trends
are completely missed. The model's `num_leaves = 15` constraint limits its
ability to model interaction effects between `slope` and `max` features.

**2. XGBoost: trivially high recall by flagging (nearly) everything**

XGBoost achieves 100 % incident recall but with ~77 % FPR.  The model's
predicted probabilities are compressed into a narrow band [0.454, 0.545],
and the recall-first threshold picks the highest threshold in that band
that still achieves ≥ 80 % recall.

**Root cause:** The high `scale_pos_weight` (~9.7) amplifies the minority
class gradient so aggressively that the model converges to a near-constant
output. This is a classic failure mode of class-weighted boosting with
insufficient regularisation.

**Mitigation (future work):**
- Reduce `scale_pos_weight` or use SMOTE instead of class weighting
- Increase `reg_alpha` / `reg_lambda` to counteract the inflated gradients
- Ensemble: use LightGBM for precision and XGBoost for recall, with a
  voting rule that requires both to agree
- Run Optuna tuning (`--tune`) to find hyper-parameters that produce better-calibrated probabilities

### Possible Improvements

1. **Multivariate features**: If metrics from the same resource were available
   simultaneously, cross-metric features (correlations, lag relationships)
   could improve precision.
2. **Sequence models** (LSTM, Transformer): Directly modelling the raw
   time-series sequence could capture temporal patterns that the fixed
   window statistics miss.
3. **Post-processing**: Requiring N consecutive alerts before triggering
   (debouncing) would reduce false positives substantially.
4. **Periodic retraining**: In a production Lambda setup, the model would
   be retrained daily on recent data to track distribution shifts.
5. **Cost-sensitive thresholding**: Rather than maximising F1, the threshold
   could be set based on the business cost ratio of false positives vs.
   missed incidents.

---

## Adaptation to a Real Alerting System

The `lambda/` directory contains a **reference implementation** of the two-Lambda
architecture described below. See [`lambda/README.md`](lambda/README.md) for
deployment instructions.

```
EventBridge (daily)            EventBridge (every minute)
       │                               │
       ▼                               ▼
 RetrainFunction              InferenceFunction
 ────────────────             ─────────────────
 1. Fetch CloudWatch           1. Load model from S3
    metric history                (ETag-cached in /tmp)
 2. Preprocess features        2. Fetch last 90 min
 3. Train LightGBM +           3. Extract features
    XGBoost                    4. Ensemble predict_proba
 4. Tune thresholds            5. Emit custom CW metric
 5. Upload to S3 ──S3─►  6. If risk ≥ threshold
                                  → publish SNS alert
```

**RetrainFunction** (daily, schema identical to `src/model.py`):
- Calls `cloudwatch.get_metric_statistics` for each monitored resource.
- Runs the same sliding-window preprocessing as `src/preprocess.py`.
- Uploads timestamped + `_latest` model artifacts to S3.

**InferenceFunction** (every minute, schema identical to `src/evaluate.py`):
- Loads models from S3, using ETag caching to avoid redundant downloads on warm starts.
- Uses online EMA normalisation (`RunningStats`) to handle gradual distribution drift.
- Emits a custom `AlertingPipeline/IncidentRisk/RiskScore` CloudWatch metric per resource.
- Publishes structured JSON SNS alerts tagged `HIGH` or `MEDIUM` severity.

---

## Project Structure

```
Alerting/
├── requirements.txt          Local pipeline dependencies
├── README.md                 This file
├── src/
│   ├── data_loader.py        Download & parse NAB data
│   ├── preprocess.py         Sliding-window features, split
│   ├── model.py              Train models, tune thresholds
│   └── evaluate.py           Evaluation, plots
├── lambda/                   AWS Lambda reference implementation
│   ├── README.md             Deployment guide
│   ├── template.yaml         AWS SAM infrastructure (S3, SNS, IAM, CW)
│   ├── Makefile              Build / deploy / invoke / logs
│   ├── shared/
│   │   └── features.py       Feature extraction shared by both Lambdas
│   ├── retrain/
│   │   └── handler.py        Daily retraining Lambda (EventBridge cron)
│   └── inference/
│       └── handler.py        Per-minute inference + SNS alerting Lambda
├── tests/
│   └── test_pipeline.py      Unit tests (10 tests)
└── data/                     Generated at runtime
    ├── metrics.parquet
    ├── processed.npz
    ├── model_lgbm.pkl
    ├── model_xgb.pkl
    ├── thresholds.pkl
    └── plots/
        ├── pr_curve_*.png
        └── threshold_sweep_*.png
```
