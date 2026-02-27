"""
inference/handler.py — Per-minute inference & alerting Lambda.

Trigger: EventBridge scheduled rule (rate: 1 minute).

What it does
────────────
1. Loads the latest model artifacts from S3 into /tmp (cached across warm
   invocations so the S3 download only happens when the container is cold or
   the artifacts have been updated).
2. For every monitored metric, fetches the last W+buffer minutes from
   CloudWatch (one GetMetricStatistics call per metric).
3. Applies online EMA normalisation (RunningStats) to produce z-score
   normalised values consistent with training.
4. Extracts the 7-feature vector for the most-recent window.
5. Runs predict_proba with both LightGBM and XGBoost.
6. If the *ensemble* probability (average of the two models) exceeds the
   tuned threshold, publishes a structured JSON alert to SNS.
7. Emits a custom CloudWatch metric ("IncidentRisk") for each monitored
   resource so the risk score can be graphed and alarmed on.

Environment variables
─────────────────────
  ARTIFACT_BUCKET   – S3 bucket containing models
  ARTIFACT_PREFIX   – S3 key prefix (default: "alerting/models/")
  SNS_TOPIC_ARN     – SNS topic to publish alerts to
  METRICS_CONFIG    – JSON string (same format as retrain handler)
  ALERT_MODEL       – Which model to use: "lgbm", "xgb", or "ensemble" (default)
  AWS_REGION        – Injected by Lambda runtime
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.features import (  # noqa: E402
    WINDOW_SIZE,
    RunningStats,
    features_for_latest_window,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Environment ───────────────────────────────────────────────────────────────
ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
ARTIFACT_PREFIX = os.environ.get("ARTIFACT_PREFIX", "alerting/models/")
SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
METRICS_CONFIG = json.loads(os.environ.get("METRICS_CONFIG", "[]"))
ALERT_MODEL = os.environ.get("ALERT_MODEL", "ensemble")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# How many extra points to fetch beyond the window (guard for gaps/missing pts)
FETCH_BUFFER = 60  # fetch last W + 60 minutes

# ── AWS clients ───────────────────────────────────────────────────────────────
cloudwatch = boto3.client("cloudwatch", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)

# ── In-memory artifact cache (survives across warm Lambda invocations) ────────
_cache: dict[str, Any] = {}  # {"lgbm": model, "xgb": model, "thresholds": dict}
_cache_etag: dict[str, str] = {}  # ETag → know when S3 artifact changed
_running_stats: dict[str, RunningStats] = {}  # metric_key → RunningStats


# ── S3 artifact loading ────────────────────────────────────────────────────────


def _s3_key(suffix: str) -> str:
    return f"{ARTIFACT_PREFIX.rstrip('/')}/{suffix}"


def _load_artifact_if_changed(key_suffix: str, cache_name: str) -> Any:
    """
    Download and unpickle an S3 object, but only if its ETag has changed
    since the last invocation (avoids unnecessary downloads on warm start).
    """
    key = _s3_key(key_suffix)
    resp = s3.head_object(Bucket=ARTIFACT_BUCKET, Key=key)
    etag = resp["ETag"]

    if _cache_etag.get(cache_name) == etag and cache_name in _cache:
        logger.info("Cache hit for %s (ETag %s)", cache_name, etag)
        return _cache[cache_name]

    logger.info("Downloading %s from s3://%s/%s …", cache_name, ARTIFACT_BUCKET, key)
    buf = io.BytesIO()
    s3.download_fileobj(ARTIFACT_BUCKET, key, buf)
    buf.seek(0)
    obj = pickle.load(buf)

    _cache[cache_name] = obj
    _cache_etag[cache_name] = etag
    return obj


def load_models_and_thresholds() -> tuple[Any, Any, dict]:
    """Return (lgbm_model, xgb_model, thresholds)."""
    lgbm = _load_artifact_if_changed("model_lgbm_latest.pkl", "lgbm")
    xgb = _load_artifact_if_changed("model_xgb_latest.pkl", "xgb")
    thresholds = _load_artifact_if_changed("thresholds_latest.pkl", "thresholds")
    return lgbm, xgb, thresholds


# ── CloudWatch data fetching ──────────────────────────────────────────────────


def fetch_recent_values(
    namespace: str,
    metric_name: str,
    dim_name: str,
    dim_value: str,
    n_minutes: int = WINDOW_SIZE + FETCH_BUFFER,
    period: int = 60,
) -> np.ndarray:
    """
    Fetch the last `n_minutes` of 1-minute averages from CloudWatch.
    Returns a numpy array sorted by timestamp ascending (most recent last).
    Missing data-points are forward-filled from the last known value.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=n_minutes)

    resp = cloudwatch.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{"Name": dim_name, "Value": dim_value}],
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=["Average"],
    )

    datapoints = sorted(resp["Datapoints"], key=lambda dp: dp["Timestamp"])
    if not datapoints:
        return np.array([])

    # Fill gaps: build a minute-by-minute array
    values = np.array([dp["Average"] for dp in datapoints], dtype=np.float64)

    # Forward-fill if fewer than WINDOW_SIZE points were returned
    if len(values) < WINDOW_SIZE:
        logger.warning(
            "Only %d points for %s/%s, need %d",
            len(values),
            namespace,
            metric_name,
            WINDOW_SIZE,
        )
        if len(values) == 0:
            return np.array([])
        values = np.concatenate(
            [
                np.full(WINDOW_SIZE - len(values), values[0]),
                values,
            ]
        )

    return values


# ── Prediction ────────────────────────────────────────────────────────────────


def predict_risk(
    values: np.ndarray,
    lgbm_model: Any,
    xgb_model: Any,
    metric_key: str,
) -> float:
    """
    Given recent metric values, return an ensemble incident-risk probability.
    Uses per-metric RunningStats for online normalisation (consistent with training).
    """
    if metric_key not in _running_stats:
        _running_stats[metric_key] = RunningStats(alpha=0.01)

    feat = features_for_latest_window(values, _running_stats[metric_key])
    if feat is None:
        return 0.0

    p_lgbm = float(lgbm_model.predict_proba(feat)[0, 1])
    p_xgb = float(xgb_model.predict_proba(feat)[0, 1])

    if ALERT_MODEL == "lgbm":
        return p_lgbm
    elif ALERT_MODEL == "xgb":
        return p_xgb
    else:  # "ensemble"
        return (p_lgbm + p_xgb) / 2.0


# ── SNS alerting ──────────────────────────────────────────────────────────────


def publish_alert(cfg: dict, risk: float, threshold: float) -> None:
    """Publish a structured alert message to SNS."""
    message = {
        "alert": "IncidentRisk",
        "severity": "HIGH" if risk > threshold + 0.15 else "MEDIUM",
        "namespace": cfg["namespace"],
        "metric": cfg["metric"],
        "dimension": f"{cfg['dim_name']}={cfg['dim_value']}",
        "risk_score": round(risk, 4),
        "threshold": round(threshold, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": (
            f"Incident predicted for {cfg['metric']} on "
            f"{cfg['dim_name']}={cfg['dim_value']}. "
            f"Risk score {risk:.3f} exceeds threshold {threshold:.3f}."
        ),
    }
    subject = f"[ALERT] Incident risk on {cfg['metric']} ({cfg['dim_value']})"
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject[:100],  # SNS subject limit
        Message=json.dumps(message, indent=2),
    )
    logger.warning("ALERT published: %s", json.dumps(message))


# ── Custom CloudWatch metric emission ─────────────────────────────────────────


def emit_risk_metric(cfg: dict, risk: float) -> None:
    """
    Emit a custom CloudWatch metric so risk scores can be graphed and
    CloudWatch Alarms can be set on them independently of this Lambda.
    """
    cloudwatch.put_metric_data(
        Namespace="AlertingPipeline/IncidentRisk",
        MetricData=[
            {
                "MetricName": "RiskScore",
                "Dimensions": [
                    {"Name": "AwsNamespace", "Value": cfg["namespace"]},
                    {"Name": "MetricName", "Value": cfg["metric"]},
                    {"Name": cfg["dim_name"], "Value": cfg["dim_value"]},
                ],
                "Timestamp": datetime.now(timezone.utc),
                "Value": risk,
                "Unit": "None",
            }
        ],
    )


# ── Lambda entry point ────────────────────────────────────────────────────────


def handler(event: dict, context: Any) -> dict:
    """
    Lambda entry point — triggered every minute by EventBridge.
    """
    logger.info("Inference Lambda invoked. event=%s", json.dumps(event))
    invocation_ts = datetime.now(timezone.utc).isoformat()

    # 1. Load models (cached; re-downloads only if ETag changed in S3)
    lgbm_model, xgb_model, thresholds = load_models_and_thresholds()

    # Ensemble threshold: average of the two individual thresholds
    th_lgbm = thresholds.get("lgbm", 0.14)
    th_xgb = thresholds.get("xgb", 0.46)
    ensemble_threshold = (th_lgbm + th_xgb) / 2.0

    threshold = {
        "lgbm": th_lgbm,
        "xgb": th_xgb,
        "ensemble": ensemble_threshold,
    }.get(ALERT_MODEL, ensemble_threshold)

    metrics = METRICS_CONFIG or []
    if not metrics:
        logger.warning("METRICS_CONFIG empty — nothing to score.")
        return {"status": "OK", "alerts": [], "timestamp": invocation_ts}

    # 2. Score every monitored metric
    results = []
    alerts = []
    errors = []

    for cfg in metrics:
        metric_key = f"{cfg['namespace']}/{cfg['metric']}/{cfg['dim_value']}"
        try:
            # Fetch recent data
            values = fetch_recent_values(
                cfg["namespace"],
                cfg["metric"],
                cfg["dim_name"],
                cfg["dim_value"],
            )
            if len(values) < WINDOW_SIZE:
                logger.warning("Insufficient data for %s, skipping.", metric_key)
                continue

            # Compute incident risk
            risk = predict_risk(values, lgbm_model, xgb_model, metric_key)

            # Emit custom CloudWatch metric
            emit_risk_metric(cfg, risk)

            result = {
                "metric": metric_key,
                "risk": round(risk, 4),
                "threshold": round(threshold, 4),
                "alert": risk >= threshold,
            }
            results.append(result)
            logger.info(
                "Scored %s: risk=%.4f threshold=%.4f alert=%s",
                metric_key,
                risk,
                threshold,
                risk >= threshold,
            )

            # 3. Trigger alert if threshold exceeded
            if risk >= threshold:
                publish_alert(cfg, risk, threshold)
                alerts.append(metric_key)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing %s: %s", metric_key, exc)
            errors.append({"metric": metric_key, "error": str(exc)})

    return {
        "status": "OK" if not errors else "PARTIAL_ERROR",
        "timestamp": invocation_ts,
        "model": ALERT_MODEL,
        "threshold": round(threshold, 4),
        "scored": len(results),
        "alerts": alerts,
        "errors": errors,
        "results": results,
    }
