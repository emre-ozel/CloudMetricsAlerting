"""
model.py — Train incident-prediction models and tune alert thresholds.

Models: Neural Network (MLP) (primary) + LightGBM (baseline).
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

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
    """Train a LightGBM classifier with SMOTE oversampling.

    LightGBM's leaf-wise (best-first) growth strategy struggles with
    the small rolling-window feature set (7 features) when combined with
    ``scale_pos_weight`` — the model converges to a near-constant output
    after just 1 iteration.  Replacing class weighting with SMOTE
    oversampling resolves this by presenting a balanced training set,
    allowing gradient boosting to find meaningful splits.
    """
    if not _LGBM_AVAILABLE:
        raise RuntimeError(
            f"LightGBM could not be imported: {_LGBM_IMPORT_ERROR}\n"
            "On macOS, install the OpenMP runtime with: brew install libomp"
        ) from _LGBM_IMPORT_ERROR
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


def train_lor(X_train, y_train):
    """Logistic Regression baseline with SMOTE handling class imbalance and scaling."""
    from imblearn.over_sampling import SMOTE
    from sklearn.pipeline import make_pipeline

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)
    )
    model.fit(X_res, y_res)
    return model


class SimplifiedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Simplified: Only 1 hidden layer with 4 neurons, aggressive dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PyTorchNNClassifier:
    """Wrapper to make PyTorch model behave like a scikit-learn classifier"""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            probs = self.model(X_tensor).numpy()

        # Scikit-learn expects shape (n_samples, 2) for binary classification predict_proba
        out = np.zeros((len(probs), 2))
        out[:, 1] = probs
        out[:, 0] = 1 - probs
        return out

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_nn(X_train, y_train, X_val, y_val):
    """Simplified PyTorch Neural Network with SMOTE, Dropout, and Early Stopping."""
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Scale the data
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_val_scaled = scaler.transform(X_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_res_scaled)
    y_train_tensor = torch.FloatTensor(y_res)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = X_train.shape[1]
    model = SimplifiedNN(input_dim)

    criterion = nn.BCELoss()
    # Add heavy L2 Regularization through weight_decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

    # Early stopping setup
    patience = 20
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_weights = None

    max_epochs = 300

    for epoch in range(max_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        # Check early stopping monitor
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"  Early stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.4f}"
                )
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return PyTorchNNClassifier(model, scaler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "lor", "nn", "all"], default="all")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train pos rate: {y_train.mean():.3f}")

    thresholds = {}

    if args.model in ("lgbm", "all"):
        print("\n── LightGBM ──────────────────────────────────────")
        if not _LGBM_AVAILABLE:
            print(
                f"  ⚠ LightGBM unavailable ({_LGBM_IMPORT_ERROR}).\n"
                "  On macOS run: brew install libomp\n"
                "  Skipping LightGBM training."
            )
        else:
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

    if args.model in ("lor", "all"):
        print("\n── Logistic Regression (Baseline) ──────────────────────────────────")
        lor_model = train_lor(X_train, y_train)
        joblib.dump(lor_model, DATA_DIR / "model_lor.pkl")
        train_acc = accuracy_score(y_train, lor_model.predict(X_train))
        val_prob = lor_model.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, lor_model.predict(X_val))
        th, rec, f1 = find_recall_threshold(y_val, val_prob)
        thresholds["lor"] = th
        print(f"  Train accuracy:  {train_acc:.4f}")
        print(f"  Val accuracy:    {val_acc:.4f}")
        print(f"  Val prob range: [{val_prob.min():.3f}, {val_prob.max():.3f}]")
        print(f"  Best threshold: {th:.2f}  (inc. recall={rec:.3f}, F1={f1:.3f})")
        print(f"  Saved model_lor.pkl")

    if args.model in ("nn", "all"):
        print("\n── Simplified PyTorch Neural Network ─────────────────────────────")
        nn_model = train_nn(X_train, y_train, X_val, y_val)
        joblib.dump(nn_model, DATA_DIR / "model_nn.pkl")
        train_acc = accuracy_score(y_train, nn_model.predict(X_train))
        val_prob = nn_model.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, nn_model.predict(X_val))
        th, rec, f1 = find_recall_threshold(y_val, val_prob)
        thresholds["nn"] = th
        print(f"  Train accuracy:  {train_acc:.4f}")
        print(f"  Val accuracy:    {val_acc:.4f}")
        print(f"  Val prob range: [{val_prob.min():.3f}, {val_prob.max():.3f}]")
        print(f"  Best threshold: {th:.2f}  (inc. recall={rec:.3f}, F1={f1:.3f})")
        print(f"  Saved model_nn.pkl")

    # Save thresholds for evaluate.py
    joblib.dump(thresholds, DATA_DIR / "thresholds.pkl")
    print(f"\n  Saved thresholds: {thresholds}")


if __name__ == "__main__":
    main()
