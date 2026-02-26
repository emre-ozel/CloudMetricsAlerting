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
python -m src.model          # train models
python -m src.evaluate       # evaluation + plots
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

## Models

### Logistic Regression (primary)

- `class_weight="balanced"` to handle ~10:1 class imbalance
- Alert threshold tuned on validation set (maximise F1): **θ = 0.41**

### LightGBM

- `scale_pos_weight` = neg/pos ratio
- 500 estimators, max depth 4, early stopping on validation loss
- Tuned threshold: **θ = 0.14**

---

## Results

### Summary

| Model | Threshold | ROC-AUC | Incident Recall | Mean Lead Time |
|-------|-----------|---------|-----------------|----------------|
| **Logistic Regression** | **0.41** | **0.648** | **100% (5/5)** | **43.0 steps** |
| LightGBM | 0.14 | 0.339 | 20% (1/5) | 0.0 steps |

### Evaluation metrics

- **Incident-level recall**: fraction of incident intervals preceded by ≥1 alert
  (target: ~80%)
- **Mean lead time**: how many steps before the incident the first alert fires
- **ROC-AUC**: overall discrimination quality
- **Step-level precision/recall**: per-time-step classification report

### Logistic Regression — step-level detail

| | Precision | Recall | F1 | Support |
|---|-----------|--------|------|---------|
| No incident | 0.937 | 0.287 | 0.439 | 8912 |
| Incident ahead | 0.136 | 0.854 | 0.234 | 1169 |

The model favours recall (85.4% step-level) at the cost of precision (13.6%),
which is the correct trade-off for an alerting system — missing an incident is
far more costly than a false alarm.

---

## Analysis & Discussion

### Why Logistic Regression outperforms LightGBM

This is a somewhat surprising result. The likely explanation:

1. **Low-dimensional feature space** (7 features): With only 7 hand-crafted
   statistical summaries, a linear decision boundary is sufficient. LightGBM's
   axis-aligned splits need many leaves to approximate what a single
   hyperplane captures directly.

2. **Feature correlations matter**: The slope, rate-of-change, and last-value
   features are linearly correlated with incident onset. Logistic regression
   exploits these linear relationships directly.

3. **Probability calibration**: LightGBM's predicted probabilities are
   compressed into a very narrow range [0.091, 0.148], making threshold
   selection fragile. Logistic regression produces well-calibrated
   probabilities spanning [0.40, 0.88].

### Limitations

- **Univariate per segment**: Each segment is treated as an independent
  univariate series. Cross-metric correlations (e.g., CPU spike + network
  spike → incident) are not captured.
- **Step-level precision is low** (13.6%): Many time steps are flagged as
  "incident ahead" when no incident follows. For a production system, this
  would cause alert fatigue.
- **Small test set**: Only 5 incident intervals in the test set — the
  incident-level recall estimate has high variance.
- **No concept drift handling**: The model is trained once; in production,
  metric distributions shift over time.

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

In a production AWS deployment, this pipeline maps to two Lambda functions:

1. **Retraining Lambda** (daily): Fetches recent CloudWatch metrics via the
   CloudWatch API, runs `preprocess.py` + `model.py`, and stores the updated
   model artifact in S3.

2. **Inference Lambda** (every minute): Loads the model from S3, fetches the
   last W data points for each monitored metric, runs feature extraction,
   and raises an SNS alert if the predicted probability exceeds the threshold.

The per-segment z-score normalisation would be replaced by a running mean/std
maintained per metric (exponential moving average), so normalisation parameters
adapt to gradual distribution shifts.

---

## Project Structure

```
Alerting/
├── requirements.txt          Dependencies
├── README.md                 This file
├── src/
│   ├── data_loader.py        Download & parse NAB data
│   ├── preprocess.py         Sliding-window features, split
│   ├── model.py              Train models, tune thresholds
│   └── evaluate.py           Evaluation, plots
├── tests/
│   └── test_pipeline.py      Unit tests (10 tests)
└── data/                     Generated at runtime
    ├── metrics.parquet
    ├── processed.npz
    ├── model_lgbm.pkl
    ├── model_lr.pkl
    ├── thresholds.pkl
    └── plots/
        ├── pr_curve_*.png
        └── threshold_sweep_*.png
```
