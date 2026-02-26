"""
preprocess.py — Sliding-window feature extraction and train/val/test split.

Reads the segment-based metrics parquet and produces feature matrices
for binary classification: "will an incident happen in the next H steps?"

Key design choices:
  - Windows never cross segment boundaries.
  - Each segment is normalised independently (z-score) before feature
    extraction, so different metric types (CPU%, bytes, counts) are
    on a comparable scale.
  - The train/val/test split is done WITHIN each segment (temporal 70/15/15)
    so that every split contains samples from every metric type.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METRICS_PATH = DATA_DIR / "metrics.parquet"

FEATURE_SUFFIXES = ["mean", "std", "min", "max", "last", "slope", "roc"]


def build_target(incident: np.ndarray, H: int) -> np.ndarray:
    """
    y(t) = 1 if any incident in [t+1, t+H], else 0.
    Returns array of length len(incident) - H.
    """
    n = len(incident)
    y = np.zeros(n - H, dtype=np.int32)
    cumsum = np.cumsum(incident)
    for t in range(n - H):
        total = cumsum[t + H] - cumsum[t]
        y[t] = int(total > 0)
    return y


def extract_window_features(values: np.ndarray, W: int) -> np.ndarray:
    """
    Extract rolling-window features from a 1D array.
    For each time step t (from W-1 onwards), compute features over [t-W+1, t].
    Returns shape (n - W + 1, 7).
    """
    n = len(values)
    n_out = n - W + 1
    X = np.zeros((n_out, 7), dtype=np.float64)
    t_axis = np.arange(W, dtype=np.float64)

    for i in range(n_out):
        w = values[i : i + W]
        X[i, 0] = np.mean(w)
        X[i, 1] = np.std(w)
        X[i, 2] = np.min(w)
        X[i, 3] = np.max(w)
        X[i, 4] = w[-1]  # last value
        if np.std(w) > 1e-12:
            X[i, 5] = np.polyfit(t_axis, w, 1)[0]  # slope
        X[i, 6] = w[-1] - w[0]  # rate of change
    return X


def process_segment(
    seg_df: pd.DataFrame, W: int, H: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Process one segment → (X, y), respecting boundaries.
    Values are z-score normalised per-segment before feature extraction."""
    values = seg_df["metric_value"].values.astype(np.float64)
    incident = seg_df["incident"].values

    if len(values) < W + H:
        return None

    # Per-segment z-score normalisation
    mu, sigma = values.mean(), values.std()
    if sigma < 1e-12:
        sigma = 1.0
    values = (values - mu) / sigma

    X = extract_window_features(values, W)  # (n-W+1, 7)
    y_full = build_target(incident, H)  # (n-H,)

    # Align: X[i] → t = i + W - 1,  y_full[t] → t
    usable = min(len(X), len(y_full) - (W - 1))
    if usable <= 0:
        return None
    X = X[:usable]
    y = y_full[W - 1 : W - 1 + usable]
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--W", type=int, default=30, help="Look-back window size")
    parser.add_argument("--H", type=int, default=5, help="Prediction horizon")
    args = parser.parse_args()
    W, H = args.W, args.H

    print(f"Preprocessing with W={W}, H={H}")
    df = pd.read_parquet(METRICS_PATH)
    print(f"  Total rows: {len(df)}, segments: {df['segment_id'].nunique()}")

    # Process each segment and split within it ────────────────────────────
    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []

    for seg_id, seg_df in df.groupby("segment_id", sort=True):
        seg_df = seg_df.sort_values("timestamp").reset_index(drop=True)
        result = process_segment(seg_df, W, H)
        if result is None:
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

        print(
            f"  Segment {seg_id} ({seg_df['segment_name'].iloc[0]}): "
            f"{n} samples, {y_seg.sum()} pos ({100 * y_seg.mean():.1f}%)"
        )

    X_train = np.vstack(train_X)
    y_train = np.concatenate(train_y)
    X_val = np.vstack(val_X)
    y_val = np.concatenate(val_y)
    X_test = np.vstack(test_X)
    y_test = np.concatenate(test_y)

    print(f"\n  Train: {X_train.shape[0]} ({100 * y_train.mean():.1f}% pos)")
    print(f"  Val:   {X_val.shape[0]} ({100 * y_val.mean():.1f}% pos)")
    print(f"  Test:  {X_test.shape[0]} ({100 * y_test.mean():.1f}% pos)")

    # Save ─────────────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "processed.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    names = [f"metric__{s}" for s in FEATURE_SUFFIXES]
    pd.Series(names).to_csv(DATA_DIR / "feature_names.csv", index=False, header=False)
    print(f"\n  Saved processed.npz, feature_names.csv")


if __name__ == "__main__":
    main()
