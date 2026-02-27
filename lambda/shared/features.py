"""
features.py — Shared feature-extraction utilities used by both Lambdas.

Mirrors the logic in src/preprocess.py so that training and inference
always produce identical feature vectors.
"""

from __future__ import annotations

import numpy as np


# ── Constants (must match src/preprocess.py) ─────────────────────────────────
WINDOW_SIZE = 30  # W: look-back window
FEATURE_NAMES = ["mean", "std", "min", "max", "last", "slope", "roc"]
N_FEATURES = len(FEATURE_NAMES)  # 7


# ── Running normalisation state (per-metric, updated online) ──────────────────


class RunningStats:
    """
    Exponential moving average of mean and variance for online z-score
    normalisation.  Alpha controls how quickly the stats adapt to drift.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mean: float | None = None
        self.var: float = 1.0

    def update(self, value: float) -> None:
        if self.mean is None:
            self.mean = value
        else:
            delta = value - self.mean
            self.mean += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta**2)

    def normalise(self, value: float) -> float:
        if self.mean is None:
            return 0.0
        std = max(self.var**0.5, 1e-12)
        return (value - self.mean) / std

    def normalise_array(self, arr: np.ndarray) -> np.ndarray:
        """Batch normalise; updates running stats with each value."""
        out = np.empty_like(arr, dtype=np.float64)
        for i, v in enumerate(arr):
            self.update(float(v))
            out[i] = self.normalise(float(v))
        return out


# ── Feature extraction ────────────────────────────────────────────────────────


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Compute the 7 features for a single window of length W.

    Parameters
    ----------
    window : np.ndarray of shape (W,)
        Z-score normalised metric values, most-recent value last.

    Returns
    -------
    np.ndarray of shape (7,)
        [mean, std, min, max, last, slope, roc]
    """
    if len(window) < 2:
        return np.zeros(N_FEATURES, dtype=np.float64)

    t_axis = np.arange(len(window), dtype=np.float64)
    slope = float(np.polyfit(t_axis, window, 1)[0]) if window.std() > 1e-12 else 0.0

    return np.array(
        [
            window.mean(),
            window.std(),
            window.min(),
            window.max(),
            window[-1],  # last value
            slope,
            float(window[-1] - window[0]),  # rate of change
        ],
        dtype=np.float64,
    )


def extract_features_from_series(
    values: np.ndarray,
    running_stats: RunningStats | None = None,
) -> np.ndarray:
    """
    Extract features for every valid window position in a time series.

    Parameters
    ----------
    values : np.ndarray of shape (T,)
        Raw metric values in chronological order.
    running_stats : RunningStats, optional
        If provided, uses online EMA normalisation (inference path).
        If None, uses per-batch z-score (training path).

    Returns
    -------
    np.ndarray of shape (T - W + 1, 7)
    """
    W = WINDOW_SIZE

    if running_stats is not None:
        # Online normalisation: update running stats and normalise
        norm = running_stats.normalise_array(values)
    else:
        # Batch z-score (mirrors preprocess.py)
        mu, sigma = values.mean(), values.std()
        sigma = max(sigma, 1e-12)
        norm = (values - mu) / sigma

    n = len(norm)
    n_out = n - W + 1
    if n_out <= 0:
        return np.zeros((0, N_FEATURES), dtype=np.float64)

    X = np.zeros((n_out, N_FEATURES), dtype=np.float64)
    for i in range(n_out):
        X[i] = extract_window_features(norm[i : i + W])
    return X


def features_for_latest_window(
    values: np.ndarray,
    running_stats: RunningStats | None = None,
) -> np.ndarray | None:
    """
    Extract a single feature vector for the most recent window.
    Returns shape (1, 7) or None if not enough data.

    Used by the inference Lambda to score one window per invocation.
    """
    if len(values) < WINDOW_SIZE:
        return None
    window_raw = values[-WINDOW_SIZE:]

    if running_stats is not None:
        window_norm = running_stats.normalise_array(window_raw)
    else:
        mu, sigma = window_raw.mean(), window_raw.std()
        sigma = max(sigma, 1e-12)
        window_norm = (window_raw - mu) / sigma

    feat = extract_window_features(window_norm)
    return feat.reshape(1, -1)
