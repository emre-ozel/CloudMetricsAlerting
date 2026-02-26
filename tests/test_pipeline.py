"""
test_pipeline.py — Sanity tests for the alerting pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_target, extract_window_features
from src.evaluate import incident_intervals, incident_level_metrics


class TestBuildTarget:
    def test_basic(self):
        incident = np.array([0, 0, 1, 0, 0, 1, 0])
        H = 2
        y = build_target(incident, H)
        # y(0): incident[1..2] = [0,1] → 1
        # y(1): incident[2..3] = [1,0] → 1
        # y(2): incident[3..4] = [0,0] → 0
        # y(3): incident[4..5] = [0,1] → 1
        # y(4): incident[5..6] = [1,0] → 1
        assert len(y) == 5
        np.testing.assert_array_equal(y, [1, 1, 0, 1, 1])

    def test_all_zeros(self):
        y = build_target(np.zeros(20, dtype=int), 5)
        assert y.sum() == 0 and len(y) == 15

    def test_all_ones(self):
        y = build_target(np.ones(10, dtype=int), 3)
        assert (y == 1).all()


class TestExtractFeatures:
    def test_shape(self):
        vals = np.random.randn(50)
        X = extract_window_features(vals, W=10)
        assert X.shape == (41, 7)

    def test_deterministic(self):
        np.random.seed(42)
        vals = np.random.randn(30)
        X1 = extract_window_features(vals, 5)
        X2 = extract_window_features(vals, 5)
        np.testing.assert_array_equal(X1, X2)


class TestIncidentIntervals:
    def test_single(self):
        y = np.array([0, 0, 1, 1, 1, 0, 0])
        assert incident_intervals(y) == [(2, 4)]

    def test_multiple(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        assert incident_intervals(y) == [(0, 1), (4, 4)]

    def test_none(self):
        assert incident_intervals(np.zeros(10, dtype=int)) == []


class TestIncidentLevelMetrics:
    def test_perfect_detection(self):
        y_true = np.array([0, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0])  # alert at step 1, before incident
        m = incident_level_metrics(y_true, y_pred)
        assert m["incident_recall"] == 1.0
        assert m["detected"] == 1

    def test_no_detection(self):
        y_true = np.array([0, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        m = incident_level_metrics(y_true, y_pred)
        assert m["incident_recall"] == 0.0
