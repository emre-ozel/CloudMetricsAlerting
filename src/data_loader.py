"""
data_loader.py — Download and parse the NAB realAWSCloudwatch dataset.

Each AWS CloudWatch CSV is a separate univariate time series with its own
anomaly windows.  We load all of them, normalise to a common schema, and
concatenate them row-wise (each file becomes a separate segment).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
NAB_REPO = "https://github.com/numenta/NAB.git"
NAB_DIR = Path(__file__).resolve().parent.parent / "data" / "NAB"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "metrics.parquet"


def clone_nab():
    """Shallow-clone the NAB repo if not already present."""
    if NAB_DIR.exists():
        print(f"NAB repo already exists at {NAB_DIR}, skipping clone.")
        return
    NAB_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning NAB repository (shallow) …")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", NAB_REPO, str(NAB_DIR)],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print("Done.")


def load_labels() -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Load combined anomaly-window labels from NAB."""
    labels_path = NAB_DIR / "labels" / "combined_windows.json"
    with open(labels_path) as f:
        raw = json.load(f)
    parsed: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for key, windows in raw.items():
        parsed[key] = [(pd.Timestamp(w[0]), pd.Timestamp(w[1])) for w in windows]
    return parsed


def load_all_aws_files() -> pd.DataFrame:
    """
    Load every CSV in realAWSCloudwatch/, attach incident labels,
    and concatenate row-wise with a segment_id column.
    """
    clone_nab()
    labels = load_labels()
    aws_dir = NAB_DIR / "data" / "realAWSCloudwatch"

    segments = []
    for seg_id, csv_path in enumerate(sorted(aws_dir.glob("*.csv"))):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.rename(columns={"value": "metric_value"})
        df["segment_id"] = seg_id
        df["segment_name"] = csv_path.stem

        # Map anomaly windows → binary incident labels
        label_key = f"realAWSCloudwatch/{csv_path.name}"
        windows = labels.get(label_key, [])
        df["incident"] = 0
        for start, end in windows:
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            df.loc[mask, "incident"] = 1

        segments.append(df)
        print(
            f"  Loaded {csv_path.name}: {len(df)} rows, "
            f"{df['incident'].sum()} incident steps, "
            f"{len(windows)} anomaly windows"
        )

    merged = pd.concat(segments, ignore_index=True)
    return merged


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_aws_files()
    df.to_parquet(OUTPUT_PATH, index=False)

    n_incidents = df["incident"].sum()
    n_segments = df["segment_id"].nunique()
    print(f"\nSaved {len(df)} rows ({n_segments} segments) to {OUTPUT_PATH}")
    print(f"  Incident steps: {n_incidents} ({100 * n_incidents / len(df):.1f}%)")


if __name__ == "__main__":
    main()
