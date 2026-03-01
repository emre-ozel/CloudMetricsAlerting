"""
retrain/handler.py — Daily retraining Lambda.

Trigger: EventBridge scheduled rule (cron: daily at 02:00 UTC).

What it does
────────────
1. Fetches the last `LOOKBACK_DAYS` days of CloudWatch metric data for every
   monitored namespace/metric, one API call per metric.
2. Runs the same preprocessing pipeline as src/preprocess.py (sliding-window
   feature extraction, temporal 70/15/15 split within each segment).
3. Trains both a LightGBM and a Neural Network classifier with class-imbalance
   handling and early stopping.
4. Tunes alert thresholds on the validation set (maximise F1).
5. Uploads model artifacts + thresholds to S3 so the inference Lambda can
   load the freshest version.
6. Returns a structured JSON result for downstream monitoring / alerting on
   the retraining job itself.

Environment variables (set via SAM template / Lambda console)
─────────────────────────────────────────────────────────────
  ARTIFACT_BUCKET   – S3 bucket to store model artifacts
  ARTIFACT_PREFIX   – S3 key prefix (default: "alerting/models/")
  LOOKBACK_DAYS     – How many days of history to fetch (default: 90)
  METRICS_CONFIG    – JSON string listing metrics to monitor (see below)
  AWS_REGION        – Injected automatically by the Lambda runtime

METRICS_CONFIG example
──────────────────────
[
  {"namespace": "AWS/EC2",  "metric": "CPUUtilization",  "dim_name": "InstanceId",  "dim_value": "i-0abc123"},
  {"namespace": "AWS/RDS",  "metric": "DatabaseConnections", "dim_name": "DBInstanceIdentifier", "dim_value": "prod-db"},
  {"namespace": "AWS/ELB",  "metric": "RequestCount",    "dim_name": "LoadBalancerName", "dim_value": "prod-elb"}
]
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
import numpy as np

# Lambda packages its own deps; shared/ is bundled alongside handler.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.features import WINDOW_SIZE, extract_features_from_series  # noqa: E402

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Environment ───────────────────────────────────────────────────────────────
ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
ARTIFACT_PREFIX = os.environ.get("ARTIFACT_PREFIX", "alerting/models/")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "90"))
METRICS_CONFIG = json.loads(os.environ.get("METRICS_CONFIG", "[]"))
REGION = os.environ.get("AWS_REGION", "us-east-1")

# ── AWS clients ───────────────────────────────────────────────────────────────
cloudwatch = boto3.client("cloudwatch", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


# ── CloudWatch data fetching ──────────────────────────────────────────────────


def fetch_metric_history(
    namespace: str,
    metric_name: str,
    dim_name: str,
    dim_value: str,
    days: int = LOOKBACK_DAYS,
    period: int = 60,  # 1-minute granularity
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull up to `days` days of 1-minute metric data from CloudWatch.

    Returns
    -------
    timestamps : np.ndarray of datetime (sorted ascending)
    values     : np.ndarray of float
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    paginator = cloudwatch.get_paginator("get_metric_statistics")
    pages = paginator.paginate(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{"Name": dim_name, "Value": dim_value}],
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=["Average"],
    )

    records = []
    for page in pages:
        for dp in page["Datapoints"]:
            records.append((dp["Timestamp"], dp["Average"]))

    if not records:
        logger.warning(
            "No data for %s/%s %s=%s", namespace, metric_name, dim_name, dim_value
        )
        return np.array([]), np.array([])

    records.sort(key=lambda r: r[0])
    timestamps = np.array([r[0] for r in records])
    values = np.array([r[1] for r in records], dtype=np.float64)
    return timestamps, values


# ── Preprocessing ─────────────────────────────────────────────────────────────


def build_target(values: np.ndarray, anomaly_indices: set[int], H: int) -> np.ndarray:
    """
    y(t) = 1 if any anomaly in [t+1, t+H], else 0.
    Without ground-truth labels at inference time we use a heuristic:
    steps where the metric exceeds mean + 3σ are treated as anomalies.
    """
    n = len(values)
    mu, sigma = values.mean(), values.std()
    sigma = max(sigma, 1e-12)
    is_anomaly = np.abs((values - mu) / sigma) > 3.0  # 3-sigma rule

    y = np.zeros(n - H, dtype=np.int32)
    cumsum = np.cumsum(is_anomaly.astype(int))
    for t in range(n - H):
        total = cumsum[t + H] - cumsum[t]
        y[t] = int(total > 0)
    return y


def segment_to_Xy(
    values: np.ndarray, H: int = 5
) -> tuple[np.ndarray, np.ndarray] | None:
    """Preprocess one metric segment into (X, y)."""
    W = WINDOW_SIZE
    if len(values) < W + H:
        return None

    X = extract_features_from_series(values)  # (n-W+1, 7) batch z-score
    y_full = build_target(values, set(), H)  # (n-H,)

    usable = min(len(X), len(y_full) - (W - 1))
    if usable <= 0:
        return None

    X = X[:usable]
    y = y_full[W - 1 : W - 1 + usable]
    return X, y


def build_datasets(metric_configs: list[dict]) -> tuple:
    """Fetch all metrics and build train/val/test splits."""
    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []

    for cfg in metric_configs:
        _, values = fetch_metric_history(
            cfg["namespace"], cfg["metric"], cfg["dim_name"], cfg["dim_value"]
        )
        if len(values) == 0:
            continue

        result = segment_to_Xy(values)
        if result is None:
            logger.warning(
                "Segment %s/%s too short, skipping", cfg["namespace"], cfg["metric"]
            )
            continue

        X_seg, y_seg = result
        n = len(X_seg)
        i1 = int(n * 0.70)
        i2 = int(n * 0.85)

        train_X.append(X_seg[:i1])
        train_y.append(y_seg[:i1])
        val_X.append(X_seg[i1:i2])
        val_y.append(y_seg[i1:i2])
        test_X.append(X_seg[i2:])
        test_y.append(y_seg[i2:])

        logger.info(
            "Segment %s/%s: %d samples, %.1f%% pos",
            cfg["namespace"],
            cfg["metric"],
            n,
            100 * y_seg.mean(),
        )

    if not train_X:
        raise RuntimeError("No usable metric segments found — aborting retrain.")

    return (
        np.vstack(train_X),
        np.concatenate(train_y),
        np.vstack(val_X),
        np.concatenate(val_y),
        np.vstack(test_X),
        np.concatenate(test_y),
    )


# ── Threshold tuning ──────────────────────────────────────────────────────────


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return the one with the highest F1."""
    from sklearn.metrics import f1_score as sk_f1

    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= th).astype(int)
        f = sk_f1(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = float(th)
    return best_th, best_f1


# ── Model training ────────────────────────────────────────────────────────────


def train_lgbm(X_train, y_train, X_val, y_val):
    import lightgbm as lgb
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_res,
        y_res,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def train_nn(X_train, y_train, X_val, y_val):
    from sklearn.neural_network import MLPClassifier
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_res, y_res)
    return model


# ── S3 artifact upload ────────────────────────────────────────────────────────


def upload_artifact(obj: Any, key_suffix: str) -> str:
    """Pickle `obj` and upload to S3. Returns the full S3 key."""
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    key = f"{ARTIFACT_PREFIX.rstrip('/')}/{key_suffix}"
    s3.upload_fileobj(buf, ARTIFACT_BUCKET, key)
    logger.info("Uploaded s3://%s/%s", ARTIFACT_BUCKET, key)
    return key


# ── Lambda entry point ────────────────────────────────────────────────────────


def handler(event: dict, context: Any) -> dict:
    """
    Lambda entry point — triggered daily by EventBridge.

    Returns a JSON-serialisable dict so Step Functions / CloudWatch Logs
    can inspect the outcome.
    """
    logger.info("Retrain Lambda started. event=%s", json.dumps(event))

    # 1. Build datasets from live CloudWatch data
    metrics = METRICS_CONFIG or []
    if not metrics:
        logger.warning("METRICS_CONFIG is empty — using synthetic demo data.")
        # Demo mode: generate synthetic data so the Lambda can be tested
        # without real CloudWatch metrics.
        rng = np.random.default_rng(42)
        n = 5000
        values = rng.normal(0, 1, n)
        # Inject a few anomaly spikes
        for i in rng.integers(500, n - 100, size=10):
            values[i : i + 20] += 4.0
        metrics = [
            {
                "namespace": "DEMO",
                "metric": "synthetic",
                "dim_name": "env",
                "dim_value": "demo",
            }
        ]
        result = segment_to_Xy(values)
        if result is None:
            return {"status": "ERROR", "reason": "synthetic data too short"}
        X_seg, y_seg = result
        n_s = len(X_seg)
        i1, i2 = int(n_s * 0.70), int(n_s * 0.85)
        X_train, y_train = X_seg[:i1], y_seg[:i1]
        X_val, y_val = X_seg[i1:i2], y_seg[i1:i2]
        X_test, y_test = X_seg[i2:], y_seg[i2:]
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = build_datasets(metrics)

    logger.info(
        "Dataset: train=%d (%.1f%% pos), val=%d, test=%d",
        len(y_train),
        100 * y_train.mean(),
        len(y_val),
        len(y_test),
    )

    # 2. Train models
    thresholds: dict[str, float] = {}
    model_keys: dict[str, str] = {}

    for model_name, train_fn in [("lgbm", train_lgbm), ("nn", train_nn)]:
        logger.info("Training %s …", model_name)
        model = train_fn(X_train, y_train, X_val, y_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        th, val_f1 = find_best_threshold(y_val, val_prob)
        thresholds[model_name] = th

        from sklearn.metrics import accuracy_score

        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        logger.info(
            "%s — train_acc=%.4f  val_acc=%.4f  test_acc=%.4f  threshold=%.2f  val_F1=%.3f",
            model_name,
            train_acc,
            val_acc,
            test_acc,
            th,
            val_f1,
        )

        # 3. Upload artifact to S3
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = upload_artifact(model, f"model_{model_name}_{timestamp}.pkl")
        # Also write a "latest" pointer so inference Lambda always loads current
        upload_artifact(model, f"model_{model_name}_latest.pkl")
        model_keys[model_name] = key

    upload_artifact(thresholds, "thresholds_latest.pkl")
    logger.info("All artifacts uploaded. thresholds=%s", thresholds)

    return {
        "status": "OK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "model_keys": model_keys,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "pos_rate": float(y_train.mean()),
    }
