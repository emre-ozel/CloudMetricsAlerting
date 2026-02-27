"""
model.py — Train incident-prediction models and tune alert thresholds.

Models: LightGBM (primary) + XGBoost (baseline).
The best threshold is selected on the validation set to maximise
incident-level recall while keeping precision reasonable.
"""

import argparse
from pathlib import Path

import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import joblib

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    d = np.load(DATA_DIR / "processed.npz")
    return (
        d["X_train"],
        d["y_train"],
        d["X_val"],
        d["y_val"],
        d["X_test"],
        d["y_test"],
    )


def find_best_threshold(y_true, y_prob):
    """Pick threshold that maximises F1 on the given set."""
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= th).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
    return best_th, best_f1


def train_lgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM classifier with class-imbalance handling."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        scale_pos_weight=scale,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def train_xgb(X_train, y_train, X_val, y_val):
    """XGBoost baseline with class-imbalance handling and early stopping."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "xgb", "both"], default="both")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train pos rate: {y_train.mean():.3f}")

    thresholds = {}

    if args.model in ("lgbm", "both"):
        print("\n── LightGBM ──────────────────────────────────────")
        lgbm_model = train_lgbm(X_train, y_train, X_val, y_val)
        joblib.dump(lgbm_model, DATA_DIR / "model_lgbm.pkl")
        val_prob = lgbm_model.predict_proba(X_val)[:, 1]
        th, f1 = find_best_threshold(y_val, val_prob)
        thresholds["lgbm"] = th
        print(f"  Val prob range: [{val_prob.min():.3f}, {val_prob.max():.3f}]")
        print(f"  Best threshold: {th:.2f}  (val F1={f1:.3f})")
        print(f"  Saved model_lgbm.pkl")

    if args.model in ("xgb", "both"):
        print("\n── XGBoost ────────────────────────────────────────")
        xgb_model = train_xgb(X_train, y_train, X_val, y_val)
        joblib.dump(xgb_model, DATA_DIR / "model_xgb.pkl")
        val_prob = xgb_model.predict_proba(X_val)[:, 1]
        th, f1 = find_best_threshold(y_val, val_prob)
        thresholds["xgb"] = th
        print(f"  Val prob range: [{val_prob.min():.3f}, {val_prob.max():.3f}]")
        print(f"  Best threshold: {th:.2f}  (val F1={f1:.3f})")
        print(f"  Saved model_xgb.pkl")

    # Save thresholds for evaluate.py
    joblib.dump(thresholds, DATA_DIR / "thresholds.pkl")
    print(f"\n  Saved thresholds: {thresholds}")


if __name__ == "__main__":
    main()
