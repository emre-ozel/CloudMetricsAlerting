# Predictive Alerting for Cloud Metrics

Cloud service metrics can be tricky to model perfectly, but being able to predict an incident before it happens is invaluable. This project implements a sliding-window binary classifier to anticipate incidents within the next **H** time steps. It's trained and evaluated using real-world AWS CloudWatch metrics from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB).

---

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m src.data_loader    # download NAB dataset
python -m src.preprocess     # feature extraction (--W 30 --H 5)
python -m src.model          # train models (baseline, neural network)
python -m src.model --tune   # Optuna hyper-parameter search for Neural Network (MLP)
python -m src.model --tune --trials 100   # custom trial count
python compare.py            # evaluate gap between train/test accuracy
python -m src.evaluate       # evaluation + plots (FPR, AP, lead time)
python -m pytest tests/ -v   # unit tests
```

---

## Framing the Problem

We approached this as a binary classification task to answer a simple question: *"Is an incident going to happen in the next H steps?"*

| Aspect | Choice |
|--------|--------|
| **Task** | Binary classification: "incident within next H steps?" |
| **Input** | A sliding window of the last **W** steps from a single metric |
| **Output** | A probability score (0 to 1); alerts trigger when this score crosses a threshold |
| **Dataset** | NAB `realAWSCloudwatch` (17 real EC2, RDS, and ELB time series) |
| **Defaults** | W = 30 (our look-back window), H = 5 (prediction horizon) |

### Defining the Target

For each time step *t*, the label is:

```text
y(t) = 1  if any incident occurs in [t+1, t+H]
y(t) = 0  otherwise
```

This setup gives the model a chance to pick up on the subtle *precursor patterns* that tend to show up right before things go wrong.

---

## The Data

We use the [NAB dataset](https://github.com/numenta/NAB), which contains real AWS CloudWatch metrics from EC2 instances, RDS databases, and ELB load balancers. The anomalies in this dataset were labeled by domain experts, making it a great fit for real-world scenarios.

- **17 segments** (distinct AWS resources) with **67,740 data points** in total.
- **Metrics tracked**: CPU utilization, disk write bytes, network in, and request counts.
- **Incident rate**: About ~9.3% of the time steps fall within an anomaly window.

Because metrics like CPU percentage and network bytes operate on completely different scales, each segment is **z-score normalized independently** before we extract any features.

---

## Extracting Features

For each time step, we extract seven rolling-window features to give the model context:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `mean` | Average value over the window |
| 2 | `std` | Standard deviation (captures volatility) |
| 3 | `min` | Lowest value |
| 4 | `max` | Highest value |
| 5 | `last` | The most recent data point |
| 6 | `slope` | Slope of the linear regression |
| 7 | `roc` | Rate of change (difference between last and first) |

The `slope` and `roc` features are especially useful for spotting *trends* that build up just before an incident.

### Training, Validation, and Test Splits

To make sure the model doesn't just memorize one type of metric, we split the data **temporally within each segment** (70% train, 15% validation, 15% test). This guarantees that every split gets a taste of every metric type, preventing the classic trap of training on CPU metrics and testing on network metrics.

---

## Choosing the Right Model

We explored several approaches before settling on our final architecture. Here's a quick look at our thought process:

| Approach | Verdict |
|----------|---------|
| ARIMA / SARIMA | **Rejected.** They forecast exact values rather than binary labels, and they assume the data is stationary (which ours definitely isn't). |
| Prophet | **Rejected.** It's great for daily or weekly business seasonality, but not for sporadic 5-minute operational spikes. |
| LSTM / Transformer | **Rejected.** We only have about 67k points. Deep sequence models need way more data than this to beat simpler models, and they'd require a GPU to train quickly. |
| Random Forest | **Briefly considered.** Gradient boosting usually beats it on tabular data with class imbalance, so we moved on. |
| LightGBM (Baseline) | **Included for comparison.** It struggled here. Its leaf-wise growth strategy couldn't do much with just 7 rolling-window features. Even with SMOTE oversampling, it barely performed better than random guessing (test AUC 0.275). |
| **Logistic Regression** | **Included as a baseline.** A trusty `sklearn` classic to help us set a performance floor. |
| **Neural Network (MLP)** | **Our top pick.** The layer-wise parameter updates explored our small feature space much better. We kept it small (one hidden layer) and added heavy regularization to keep overfitting in check. |

### The Struggle with LightGBM

LightGBM loves to grow trees **leaf-wise (best-first)**, meaning it aggressively splits the leaf with the highest gain. But with only 7 aggregate features and a ~10:1 class imbalance, it quickly ran out of useful splits. It basically gave up after one iteration, defaulting everything to the base rate (~0.095).

Switching to **SMOTE** oversampling helped keep it training for a bit longer (~24 iterations) and gave us a decent spread of probabilities. Still, the actual prediction quality was poor (test AUC 0.275), meaning it actually assigned *higher* risk scores to normal steps than to actual incidents.

The **Neural Network (MLP)** sidesteps this entirely by expanding all nodes at each layer. It’s forced to look at the whole feature space, even when the immediate gain is tiny, resulting in much better generalization for this specific dataset.

---

## Our Models

### Neural Network (MLP) (Primary)

- Built with standard `sklearn.neural_network.MLPClassifier`.
- Intentionally kept small to prevent overfitting: just **1 hidden layer with 16 neurons**.
- **Aggressive Regularization**: Uses the `Adam` optimizer with an `alpha=0.01` L2 penalty.
- **Early Stopping**: Stops training automatically when the validation loss bottoms out.
- Handled class imbalance using **SMOTE**.
- Threshold tuned to **θ = ~0.45** (prioritizing recall).

### Logistic Regression (Baseline)
- Built with `sklearn.linear_model.LogisticRegression`.
- Wrapped in a `StandardScaler` to ensure everything is treated fairly.
- Handled class imbalance using **SMOTE**.

### LightGBM (Baseline)

- Handled class imbalance using **SMOTE** oversampling (because `scale_pos_weight` caused it to stall after 1 iteration as discussed above).
- 500 estimators, max depth 6, `num_leaves=31`, early stopping on validation loss.
- Tuned threshold: **θ = 0.35** (prioritizing recall).

### Tuning the Alert Threshold

If you optimize strictly for the F1 score, you often end up sacrificing recall for precision. But in a real alerting system, missing an incident is way worse than getting a false alarm. So, we use a **recall-first** strategy. We find the highest threshold that still catches ≥ 80% of incident windows, and only fall back to F1 if that goal is completely out of reach.

You can check out how we do this in `find_recall_threshold()` over in `model.py`.

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
| **Neural Network (MLP)** | **74.76%** | **67.93%** |
| Logistic Regression | 68.09% | 72.63% |
| LightGBM | 78.26% | 70.53% |

> **Note**: Raw accuracy is misleading here due to ~10:1 class imbalance (~90% of
> steps are negative). A model that predicts all-negative scores 90% accuracy.
> LightGBM's lower accuracy reflects SMOTE training on a balanced 50/50 set,
> which shifts its decision boundary toward flagging more positives.
> Incident-level recall (below) is the true measure of alerting quality.

### Test-Set Summary

| Model | Threshold | Test Accuracy |
|-------|-----------|--------------|
| **Neural Network (MLP)** | **0.45** | **68.59%** |
| Logistic Regression | 0.41 | 70.88% |

> By dramatically shrinking the network capacity (16 hidden nodes) mixed with L2 Weight Decay (alpha=0.01) and early stopping, we've wrestled the notoriously severe overfitting down.

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
    ├── model_lor.pkl
    ├── model_nn.pkl
    ├── thresholds.pkl
    └── plots/
        ├── pr_curve_*.png
        └── threshold_sweep_*.png
```
