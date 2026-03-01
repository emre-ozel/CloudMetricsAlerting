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
python -m src.model --tune   # Optuna hyper-parameter search for Neural Network (MLP)
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
| LightGBM | Yes | **Included as baseline.** LightGBM's leaf-wise (best-first) growth strategy struggles with only 7 rolling-window features — the split-gain criterion finds no improvement after 1 iteration when using `scale_pos_weight`. Switching to SMOTE oversampling resolves the split-starvation issue but test-set AUC (0.275) remains below random, indicating the algorithm is a poor fit for this feature space. Retained for comparison. |
| **Neural Network (MLP)** | **Selected** | Neural Network (MLP)'s depth-wise growth explores the full feature space more uniformly. With capped `scale_pos_weight` (3.0) and increased regularisation (`reg_alpha=1.0`, `reg_lambda=5.0`), it achieves meaningful probability spread and the best discrimination (AUC 0.580, AP 0.228) while maintaining 100% incident recall. |

### Why gradient boosting fits this problem

1. **Class imbalance**: Only ~9 % of time steps are positive. Neural Network (MLP) uses
   capped `scale_pos_weight` (3.0) to re-weight the loss; LightGBM uses
   SMOTE oversampling because leaf-wise growth stalls with gradient weighting
   on this feature set.
2. **Heavy-tailed features**: Tree splits are invariant to monotone transforms,
   so extreme CPU or network-byte values do not distort the model.
3. **Small data regime**: Boosting with shallow trees (depth 6) generalises well
   even with ~47 k training samples.
4. **Interpretability**: Feature importance reveals which rolling statistics
   are most predictive, aiding root-cause analysis.

### Why LightGBM underperforms Neural Network (MLP) here

LightGBM uses **leaf-wise (best-first)** tree growth: at each step it splits
the leaf with the highest gain.  With only 7 aggregate features and ~10:1 class
imbalance, the algorithm finds "no further splits with positive gain" after the
very first iteration.  This causes probability predictions to collapse to the
base rate (~0.095), regardless of hyperparameter tuning.

Switching from `scale_pos_weight` to **SMOTE** oversampling fixes the
split-starvation — the model now trains for ~24 iterations and produces a wide
probability range [0.18, 0.82].  However, the resulting discrimination is still
poor (test AUC 0.275), meaning the model assigns *higher* probabilities to
non-incident steps than incident steps.

Neural Network (MLP)'s **depth-wise** growth does not suffer this issue because it expands
all leaves at each depth level, forcing exploration of the full feature space
even when individual gains are small.

---

## Models

### Neural Network (MLP) (primary)

- `scale_pos_weight` = min(neg/pos, **3.0**) — capped to prevent probability
  compression (raw ratio ~9.7 caused near-constant output)
- `reg_alpha=1.0`, `reg_lambda=5.0` — stronger regularisation to counteract
  the amplified minority-class gradient
- 500 estimators, max depth **6**, early stopping (`rounds=50`) on validation loss
- Tuned threshold: **θ = 0.26** (recall-first)

### LightGBM (baseline)

- Class imbalance handled via **SMOTE** oversampling (not `scale_pos_weight`,
  which caused the model to stall after 1 iteration — see *"Why LightGBM
  underperforms Neural Network (MLP) here"* above)
- 500 estimators, max depth 6, `num_leaves=31`, early stopping on validation loss
- Tuned threshold: **θ = 0.35** (recall-first)

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
|-------|---------------|-------------|
| **Neural Network (MLP)** | **90.45%** | **92.31%** |
| LightGBM | 78.26% | 70.53% |

> **Note**: Raw accuracy is misleading here due to ~10:1 class imbalance (~90% of
> steps are negative). A model that predicts all-negative scores 90% accuracy.
> LightGBM's lower accuracy reflects SMOTE training on a balanced 50/50 set,
> which shifts its decision boundary toward flagging more positives.
> Incident-level recall (below) is the true measure of alerting quality.

### Test-Set Summary

| Model | Threshold | Test Accuracy | ROC-AUC | Incident Recall | Mean Lead Time | FPR | Avg Precision |
|-------|-----------|--------------|---------|-----------------|----------------|-----|---------------|
| **Neural Network (MLP)** | **0.26** | **39.10%** | **0.580** | **100% (5/5)** | **32.4 steps** | **0.6603** | **0.228** |
| LightGBM | 0.35 | 21.47% | 0.275 | 100% (5/5) | 28.8 steps | 0.8247 | 0.107 |

> With the **recall-first threshold tuning** (find the highest threshold with
> ≥ 80 % incident recall), both models achieve 100 % incident recall on
> the test set. Neural Network (MLP) is the clear winner: capping `scale_pos_weight` at 3.0
> and increasing regularisation reduced FPR from 0.77 → 0.66 and raised AP
> from 0.175 → 0.228 compared to the uncapped baseline.

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
| Neural Network (MLP) | 0.228 | Improved from 0.175 after capping `scale_pos_weight` and increasing regularisation. Still low, but the model now produces a wider probability spread [0.25, 0.29] and a more usable recall-precision trade-off. |
| LightGBM | 0.107 | SMOTE resolved the probability-compression issue (range now [0.18, 0.82]) but discrimination remains poor — the model ranks positives below negatives (AUC < 0.5). |

**What the curve shape means for threshold choice:**

- **Neural Network (MLP)**: The PR curve is relatively flat at low precision up to ~95 %
  recall, then drops sharply. This means we can achieve high incident recall
  cheaply, but cannot improve precision much without sacrificing recall.
  The optimal operating point is near the "elbow" — the recall-first threshold
  (θ = 0.26) sits just past this elbow, accepting many false alarms to
  guarantee incident detection.
- **LightGBM**: Despite the wider probability range from SMOTE, the curve
  shape remains steep — the model's leaf-wise growth strategy is a poor fit
  for the 7-feature rolling-window space.

### False Positive Rate (FPR) Discussion

| Model | FPR | False Alarms / Hour | Assessment |
|-------|-----|--------------------|-----------|
| Neural Network (MLP) | 0.6603 | ~7.0 | **Improved** from 0.77 after capping `scale_pos_weight` at 3.0. Still high for production — post-processing (debouncing, N-of-M voting) would be needed. |
| LightGBM | 0.8247 | ~8.7 | **Worse than before.** SMOTE fixed the probability range but the model's poor discrimination means it flags most steps. |

### Neural Network (MLP) — step-level detail

| | Precision | Recall | F1 | Support |
|---|-----------|--------|------|---------|
| No incident | 0.923 | 0.340 | 0.497 | 8912 |
| Incident ahead | 0.135 | 0.783 | 0.230 | 1169 |

Neural Network (MLP) is tuned for **very high incident recall (100% at incident level)** at
the cost of step-level precision (13.5%), which is the correct trade-off for an
alerting system — missing an incident is far more costly than a false alarm.
Compared to the previous uncapped `scale_pos_weight`, step-level precision
improved from 8.1% → 13.5% and negative-class recall from 22.6% → 34.0%.

### LightGBM — step-level detail

| | Precision | Recall | F1 | Support |
|---|-----------|--------|------|----------|
| No incident | 0.734 | 0.175 | 0.283 | 8912 |
| Incident ahead | 0.076 | 0.515 | 0.132 | 1169 |

---

## Analysis & Discussion

### Why both models achieve 100% incident recall

1. **Recall-first threshold tuning**: The threshold is selected as the highest
   value where incident-level recall ≥ 80%. For both models this results in a
   threshold low enough that many steps are flagged positive, guaranteeing all
   incident intervals are detected.

2. **Class-imbalance handling**: Neural Network (MLP) uses capped `scale_pos_weight` (3.0)
   to amplify the minority class gradient; LightGBM uses SMOTE oversampling
   to present a balanced training set.

3. **Early stopping on validation**: Both models stop at the iteration that best
   generalises to the val set, preventing overfitting while maintaining recall.

### Why test accuracy is low (~21–39%)

Low raw accuracy is expected and even desirable in this context:

- With ~88% negative steps, a model predicting all-negative trivially scores ~88%
  accuracy — but misses every incident.
- Both models deliberately bias toward positives to maximise incident recall.
  The price is many false alarms (low step-level precision), which tanks raw accuracy.
- Neural Network (MLP) (39.1%) is significantly better than LightGBM (21.5%) due to its
  superior probability calibration.
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

### Failure Cases & Mitigations Applied

**1. LightGBM: leaf-wise split starvation (resolved ↦ SMOTE)**

*Original problem:* With `scale_pos_weight ≈ 9.7`, LightGBM's leaf-wise growth
found "no further splits with positive gain" after iteration 1, collapsing all
predictions to the base rate (~0.095). The model was effectively predicting the
majority class.

*Fix applied:* Replaced `scale_pos_weight` with **SMOTE** oversampling. The
model now trains for ~24 iterations and produces probabilities in [0.18, 0.82].
However, test AUC (0.275) remains below random — LightGBM's leaf-wise strategy
is fundamentally a poor fit for this 7-feature rolling-window space.

**2. Neural Network (MLP): probability compression (resolved ↦ capped weighting + regularisation)**

*Original problem:* The high `scale_pos_weight` (~9.7) amplified the minority
class gradient so aggressively that probabilities compressed to [0.454, 0.545],
making every threshold either catch-all or catch-nothing.

*Fix applied:* Capped `scale_pos_weight` at **3.0** and increased regularisation
(`reg_alpha=1.0`, `reg_lambda=5.0`, `max_depth=6`). Results:
- FPR: 0.77 → **0.66** (−15%)
- AP: 0.175 → **0.228** (+30%)
- ROC-AUC: 0.513 → **0.580** (+13%)
- 100% incident recall maintained

**Remaining issues:**
- Neural Network (MLP) FPR (0.66) is still too high for production; debouncing or N-of-M
  voting would be needed.
- LightGBM remains non-competitive; replacing it with a tuned Neural Network (MLP) ensemble
  or a sequence model (LSTM) would be more productive than further tuning.

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
    Neural Network (MLP)                    4. Ensemble predict_proba
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
    ├── model_nn.pkl
    ├── thresholds.pkl
    └── plots/
        ├── pr_curve_*.png
        └── threshold_sweep_*.png
```
