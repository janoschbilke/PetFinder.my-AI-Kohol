import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.utils import print_cv_summary, quadratic_weighted_kappa

CACHE_DIR = Path("cache")
DATA_ROOT = Path("/Users/I568521/.cache/kagglehub/competitions/petfinder-adoption-prediction")

ADOPTION_SPEED_LABELS = ["Same day", "1-7 days", "8-30 days", "31-90 days", ">100 days"]


def load_model():
    model_path = CACHE_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at {model_path}. Run model.py first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_oof() -> tuple[np.ndarray, np.ndarray] | None:
    oof_path = CACHE_DIR / "oof_predictions.npy"
    labels_path = CACHE_DIR / "oof_labels.npy"
    if not oof_path.exists() or not labels_path.exists():
        return None
    return np.load(oof_path), np.load(labels_path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))
    axes[0].set_xticklabels(ADOPTION_SPEED_LABELS, rotation=30, ha="right", fontsize=8)
    axes[0].set_yticklabels(ADOPTION_SPEED_LABELS, fontsize=8)
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    axes[1].set_title("Confusion Matrix (row %)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(ADOPTION_SPEED_LABELS, rotation=30, ha="right", fontsize=8)
    axes[1].set_yticklabels(ADOPTION_SPEED_LABELS, fontsize=8)
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center", fontsize=7,
                         color="white" if cm_pct[i, j] > 50 else "black")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")


def plot_feature_importance(model, col_names: list[str], top_n: int = 40) -> None:
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(10, 12))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [col_names[i] for i in indices])
    plt.xlabel("Feature importance (split)")
    plt.title(f"Top {top_n} features")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved: feature_importance.png")


def run(force: bool = False) -> None:
    print("Loading saved model...")
    model = load_model()

    col_names_path = CACHE_DIR / "col_names.npy"
    col_names = np.load(col_names_path, allow_pickle=True).tolist() if col_names_path.exists() else []

    oof = load_oof()
    if oof is not None:
        oof_preds, oof_labels = oof
        print("\n--- Out-of-Fold Evaluation ---")
        qwk = quadratic_weighted_kappa(oof_labels, oof_preds)
        acc = accuracy_score(oof_labels, oof_preds)
        f1_macro = f1_score(oof_labels, oof_preds, average="macro")
        f1_weighted = f1_score(oof_labels, oof_preds, average="weighted")
        print(f"  QWK (Quadratic Weighted Kappa): {qwk:.4f}")
        print(f"  Accuracy:                       {acc:.4f}")
        print(f"  F1 macro:                       {f1_macro:.4f}")
        print(f"  F1 weighted:                    {f1_weighted:.4f}")
        print("\n--- Per-Class Report ---")
        print(classification_report(oof_labels, oof_preds, target_names=ADOPTION_SPEED_LABELS))
        plot_confusion_matrix(oof_labels, oof_preds)
    else:
        print("No OOF predictions found. Run model.py first.")

    if col_names:
        plot_feature_importance(model, col_names)
    else:
        print("No column names found, skipping feature importance plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)