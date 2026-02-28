"""
model.py — Train incident-prediction models and tune alert thresholds.

Models: LightGBM (primary) + XGBoost (baseline).
The best threshold is selected on the validation set using a recall-first
strategy: find the highest threshold where incident-level recall >= 80%.
Optuna hyperparameter tuning is available via the --tune flag.
"""

import argparse
from pathlib import Path

import numpy as np
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except (ImportError, OSError) as _lgbm_err:
    lgb = None  # type: ignore[assignment]
    _LGBM_AVAILABLE = False
    _LGBM_IMPORT_ERROR = _lgbm_err
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
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


# ---------------------------------------------------------------------------
# Incident helpers (duplicated from evaluate.py to keep model.py self-contained)
# ---------------------------------------------------------------------------


def _incident_intervals(y: np.ndarray) -> list[tuple[int, int]]:
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


def _incident_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute incident-level recall (matches evaluate.py logic)."""
    intervals = _incident_intervals(y_true)
    if not intervals:
        return float("nan")
    alert_set = set(np.where(y_pred == 1)[0])
    detected = 0
    for start, end in intervals:
        for idx in range(max(0, start - 50), end + 1):
            if idx in alert_set:
                detected += 1
                break
    return detected / len(intervals)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def find_best_threshold(y_true, y_prob):
    """Pick threshold that maximises F1 on the given set (legacy helper)."""
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= th).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
    return best_th, best_f1


def find_recall_threshold(y_true, y_prob, min_recall: float = 0.80):
    """
    Find the **highest** threshold where incident-level recall >= *min_recall*.

    This aligns threshold tuning directly with the project goal (≥ 80 %
    incident recall) rather than optimising F1, which can sacrifice recall
    for precision.

    Falls back to the F1-maximising threshold if the recall target is
    unachievable at every threshold.
    """
    best_th = None
    for th in np.arange(0.95, 0.04, -0.01):
        pred = (y_prob >= th).astype(int)
        rec = _incident_recall(y_true, pred)
        if rec >= min_recall:
            best_th = th
            break  # highest threshold meeting the recall target

    if best_th is None:
        # Recall target unachievable — fall back to F1-max
        best_th, _ = find_best_threshold(y_true, y_prob)
        print(
            f"  ⚠ Recall target ({min_recall:.0%}) not achievable; "
            f"falling back to F1-max threshold {best_th:.2f}"
        )

    pred = (y_prob >= best_th).astype(int)
    rec = _incident_recall(y_true, pred)
    f1 = f1_score(y_true, pred, zero_division=0)
    return best_th, rec, f1


def train_lgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM classifier with class-imbalance handling."""
    if not _LGBM_AVAILABLE:
        raise RuntimeError(
            f"LightGBM could not be imported: {_LGBM_IMPORT_ERROR}\n"
            "On macOS, install the OpenMP runtime with: brew install libomp"
        ) from _LGBM_IMPORT_ERROR
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


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_xgb_with_optuna(X_train, y_train, X_val, y_val, n_trials: int = 50):
    """
    Search XGBoost hyper-parameters with Optuna.

    Objective: maximise **incident-level recall** on the validation set,
    which directly targets the ≥ 80 % recall success criterion.
    """
    import optuna

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 700),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=50,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        # Use the recall-first threshold, then measure incident recall
        th, rec, _ = find_recall_threshold(y_val, y_prob)
        return rec

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="xgb-incident-recall")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Optuna best incident recall: {study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")

    # Re-train with best params
    best = study.best_params
    model = XGBClassifier(
        **best,
        scale_pos_weight=scale,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "xgb", "both"], default="both")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning for XGBoost before training.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50).",
    )
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train pos rate: {y_train.mean():.3f}")

    thresholds = {}

    if args.model in ("lgbm", "both"):
        print("\n── LightGBM ──────────────────────────────────────")
        if not _LGBM_AVAILABLE:
            print(
                f"  ⚠ LightGBM unavailable ({_LGBM_IMPORT_ERROR}).\n"
                "  On macOS run: brew install libomp\n"
                "  Skipping LightGBM training."
            )
            if args.model == "lgbm":
                return
            # fall through to XGBoost when model == "both"
            args.model = "xgb"
            print()
        lgbm_model = train_lgbm(X_train, y_train, X_val, y_val)
        joblib.dump(lgbm_model, DATA_DIR / "model_lgbm.pkl")
        train_acc = accuracy_score(y_train, lgbm_model.predict(X_train))
        val_prob = lgbm_model.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, lgbm_model.predict(X_val))
        th, rec, f1 = find_recall_threshold(y_val, val_prob)
        thresholds["lgbm"] = th
        print(f"  Train accuracy:  {train_acc:.4f}")
        print(f"  Val accuracy:    {val_acc:.4f}")
        print(f"  Val prob range: [{val_prob.min():.3f}, {val_prob.max():.3f}]")
        print(f"  Best threshold: {th:.2f}  (inc. recall={rec:.3f}, F1={f1:.3f})")
        print(f"  Saved model_lgbm.pkl")

    if args.model in ("xgb", "both"):
        print("\n── XGBoost ────────────────────────────────────────")
        if args.tune:
            print(f"  Running Optuna tuning ({args.trials} trials)…")
            xgb_model = tune_xgb_with_optuna(
                X_train, y_train, X_val, y_val, n_trials=args.trials
            )
        else:
            xgb_model = train_xgb(X_train, y_train, X_val, y_val)
        joblib.dump(xgb_model, DATA_DIR / "model_xgb.pkl")
        train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
        val_prob = xgb_model.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, xgb_model.predict(X_val))
        th, rec, f1 = find_recall_threshold(y_val, val_prob)
        thresholds["xgb"] = th
        print(f"  Train accuracy:  {train_acc:.4f}")
        print(f"  Val accuracy:    {val_acc:.4f}")
        print(f"  Val prob range: [{val_prob.min():.3f}, {val_prob.max():.3f}]")
        print(f"  Best threshold: {th:.2f}  (inc. recall={rec:.3f}, F1={f1:.3f})")
        print(f"  Saved model_xgb.pkl")

    # Save thresholds for evaluate.py
    joblib.dump(thresholds, DATA_DIR / "thresholds.pkl")
    print(f"\n  Saved thresholds: {thresholds}")


if __name__ == "__main__":
    main()
