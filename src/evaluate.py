"""
evaluate.py — Evaluate trained models on the test set.

Reports: step-level classification, incident-level recall/precision,
detection lead time, PR curves, and threshold sweeps.
Uses thresholds tuned on the val set (from model.py).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    f1_score,
)
import joblib

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLOTS_DIR = DATA_DIR / "plots"


def load_data():
    d = np.load(DATA_DIR / "processed.npz")
    return d["X_test"], d["y_test"]


def incident_intervals(y: np.ndarray) -> list[tuple[int, int]]:
    """Identify contiguous incident intervals. Returns (start, end) inclusive."""
    intervals = []
    in_inc = False
    start = 0
    for i, v in enumerate(y):
        if v == 1 and not in_inc:
            start = i
            in_inc = True
        elif v == 0 and in_inc:
            intervals.append((start, i - 1))
            in_inc = False
    if in_inc:
        intervals.append((start, len(y) - 1))
    return intervals


def incident_level_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Incident-level recall & mean detection lead time.
    An incident is detected if ≥ 1 alert fires within or up to 50 steps
    before the incident interval.
    """
    intervals = incident_intervals(y_true)
    if not intervals:
        return {
            "incident_recall": float("nan"),
            "mean_lead_time": float("nan"),
            "detected": 0,
            "total_incidents": 0,
        }

    alert_set = set(np.where(y_pred == 1)[0])
    detected = 0
    lead_times = []

    for start, end in intervals:
        earliest = None
        for idx in range(max(0, start - 50), end + 1):
            if idx in alert_set:
                if earliest is None:
                    earliest = idx
        if earliest is not None:
            detected += 1
            lead_times.append(max(0, start - earliest))

    recall = detected / len(intervals)
    mean_lt = float(np.mean(lead_times)) if lead_times else 0.0
    return {
        "incident_recall": recall,
        "detected": detected,
        "total_incidents": len(intervals),
        "mean_lead_time": mean_lt,
    }


def plot_pr_curve(y_true, y_prob, label, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, lw=2, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve ({label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_threshold_sweep(y_true, y_prob, label, save_path):
    thresholds = np.linspace(0.05, 0.95, 30)
    recalls, precisions, f1s = [], [], []
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        m = incident_level_metrics(y_true, pred)
        recalls.append(m["incident_recall"])
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        precisions.append(tp / max(tp + fp, 1))
        f1s.append(f1_score(y_true, pred, zero_division=0))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, recalls, "o-", label="Incident Recall", lw=2)
    plt.plot(thresholds, precisions, "s-", label="Precision", lw=2)
    plt.plot(thresholds, f1s, "^-", label="F1", lw=2)
    plt.xlabel("Alert Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Sweep ({label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Positive Rate = FP / (FP + TN)."""
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return fp / max(fp + tn, 1)


def compute_false_alarm_rate_per_hour(
    y_true: np.ndarray, y_pred: np.ndarray, step_minutes: float = 5.0
) -> float:
    """False alarms per hour = FP / total_hours."""
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    total_hours = len(y_true) * step_minutes / 60.0
    return fp / max(total_hours, 1e-9)


def evaluate_model(model_path: Path, X_test, y_test, label: str, threshold: float):
    """Full evaluation of a single model at the given threshold."""
    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'═' * 60}")
    print(f"  Model: {label}  (threshold={threshold:.2f})")
    print(f"{'═' * 60}")

    print(f"\n── Step-level Classification Report ──")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy:   {test_acc:.4f}")

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")
    print(f"  ROC-AUC: {auc:.3f}")

    # ── New metrics: FPR, false alarm rate, average precision ──
    fpr = compute_fpr(y_test, y_pred)
    fa_per_hour = compute_false_alarm_rate_per_hour(y_test, y_pred)
    try:
        ap = average_precision_score(y_test, y_prob)
    except ValueError:
        ap = float("nan")
    print(f"  False Positive Rate (FPR): {fpr:.4f}")
    print(f"  False Alarm Rate:  {fa_per_hour:.2f} / hour")
    print(f"  Avg Precision (AP): {ap:.3f}")

    print(f"\n── Incident-level Metrics ──")
    inc = incident_level_metrics(y_test, y_pred)
    print(
        f"  Incident recall: {inc['incident_recall']:.3f}"
        f"  ({inc['detected']}/{inc['total_incidents']} detected)"
    )
    print(f"  Mean lead time:  {inc['mean_lead_time']:.1f} steps")

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_pr_curve(y_test, y_prob, label, PLOTS_DIR / f"pr_curve_{label}.png")
    plot_threshold_sweep(
        y_test, y_prob, label, PLOTS_DIR / f"threshold_sweep_{label}.png"
    )
    print(f"  Saved plots to {PLOTS_DIR}/")

    return {
        "label": label, "auc": auc, "threshold": threshold,
        "fpr": fpr, "fa_per_hour": fa_per_hour, "ap": ap,
        **inc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "xgb", "both"], default="both")
    args = parser.parse_args()

    X_test, y_test = load_data()
    print(f"Test set: {X_test.shape[0]} samples, {y_test.mean():.3f} positive rate")
    print(f"Incident intervals in test set: {len(incident_intervals(y_test))}")

    # Load tuned thresholds
    th_path = DATA_DIR / "thresholds.pkl"
    thresholds = joblib.load(th_path) if th_path.exists() else {}

    results = []
    if args.model in ("lgbm", "both"):
        path = DATA_DIR / "model_lgbm.pkl"
        if path.exists():
            th = thresholds.get("lgbm", 0.5)
            results.append(evaluate_model(path, X_test, y_test, "LightGBM", th))

    if args.model in ("xgb", "both"):
        path = DATA_DIR / "model_xgb.pkl"
        if path.exists():
            th = thresholds.get("xgb", 0.5)
            results.append(evaluate_model(path, X_test, y_test, "XGBoost", th))

    if results:
        print(f"\n{'═' * 60}")
        print("  Summary")
        print(f"{'═' * 60}")
        header = (
            f"  {'Model':<16} {'Thresh':>6} {'AUC':>6} {'Inc.Rec':>8} "
            f"{'LeadT':>6} {'FPR':>7} {'FA/h':>7} {'AP':>6}"
        )
        print(header)
        for r in results:
            print(
                f"  {r['label']:<16} {r['threshold']:>6.2f} {r['auc']:>6.3f} "
                f"{r['incident_recall']:>8.3f} {r['mean_lead_time']:>6.1f} "
                f"{r['fpr']:>7.4f} {r['fa_per_hour']:>7.2f} {r['ap']:>6.3f}"
            )


if __name__ == "__main__":
    main()
