import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
import joblib

import sys

sys.path.append(str(Path(__file__).resolve().parent))
from src.model import PyTorchNNClassifier, SimplifiedNN

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data():
    d = np.load(DATA_DIR / "processed.npz")
    return (
        d["X_train"],
        d["y_train"],
        d["X_test"],
        d["y_test"],
    )


def compare_models():
    """Load train/test data and trained models, and print their accuracy gaps."""
    X_train, y_train, X_test, y_test = load_data()

    print("Loading models...")

    models = {}

    try:
        models["Logistic Regression"] = joblib.load(DATA_DIR / "model_lor.pkl")
    except FileNotFoundError:
        print(
            "  Logistic Regression model not found. Run `python src/model.py --model lor` first."
        )

    try:
        models["Simplified Neural Network"] = joblib.load(DATA_DIR / "model_nn.pkl")
    except FileNotFoundError:
        print(
            "  Neural Network model not found. Run `python src/model.py --model nn` first."
        )

    print("\n--- Accuracy Comparison ---\n")

    for name, model in models.items():
        # Scikit-learn and our PyTorch wrapper both support .predict() directly
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        gap = train_acc - test_acc

        print(f"[{name}]")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:     {test_acc:.4f}")
        print(f"  Gap (Train-Test):  {gap:+.4f} ({(gap * 100):.2f}%)")
        print("-" * 30)


if __name__ == "__main__":
    compare_models()
