import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from src.utils import print_cv_summary, print_qwk, quadratic_weighted_kappa

CACHE_DIR = Path("cache")
N_FOLDS = 5
RANDOM_STATE = 42

ADOPTION_SPEED_LABELS = ["Same day", "1-7 days", "8-30 days", "31-90 days", ">100 days"]

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,          # grow full trees; forest variance is controlled by bagging
    "min_samples_leaf": 4,
    "max_features": "sqrt",     # standard for classification
    "class_weight": "balanced", # compensate for class imbalance
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the preprocessed tabular features (no image embeddings)."""
    print("  Loading tabular features from preprocessing cache...")
    train_feat = pd.read_parquet(CACHE_DIR / "train_features.parquet")
    test_feat = pd.read_parquet(CACHE_DIR / "test_features.parquet")

    y = train_feat["AdoptionSpeed"].values.astype(int)
    test_pet_ids = test_feat["PetID"].tolist()

    feature_cols = [c for c in train_feat.columns if c not in ("AdoptionSpeed", "PetID")]
    X_train = train_feat[feature_cols].values.astype(np.float32)
    X_test = test_feat[feature_cols].values.astype(np.float32)

    print(f"  Feature matrix: train={X_train.shape}  test={X_test.shape}")
    print(f"  Number of features: {len(feature_cols)}")
    return X_train, y, X_test, test_pet_ids, feature_cols


def print_oof_metrics(oof_preds: np.ndarray, oof_labels: np.ndarray) -> None:
    """Print full evaluation metrics on the out-of-fold predictions."""
    qwk = quadratic_weighted_kappa(oof_labels, oof_preds)
    acc = accuracy_score(oof_labels, oof_preds)
    f1_macro = f1_score(oof_labels, oof_preds, average="macro")
    f1_weighted = f1_score(oof_labels, oof_preds, average="weighted")

    print("\n--- Out-of-Fold Evaluation ---")
    print(f"  QWK (Quadratic Weighted Kappa): {qwk:.4f}")
    print(f"  Accuracy:                       {acc:.4f}")
    print(f"  F1 macro:                       {f1_macro:.4f}")
    print(f"  F1 weighted:                    {f1_weighted:.4f}")

    print("\n--- Per-Class Report ---")
    print(classification_report(oof_labels, oof_preds, target_names=ADOPTION_SPEED_LABELS))

    print("--- Confusion Matrix (counts) ---")
    cm = confusion_matrix(oof_labels, oof_preds)
    # Header row
    col_width = 11
    header = " " * col_width + "".join(f"{lbl:>{col_width}}" for lbl in ADOPTION_SPEED_LABELS)
    print(header)
    for i, row_label in enumerate(ADOPTION_SPEED_LABELS):
        row = f"{row_label:>{col_width}}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(5))
        print(row)
    print()


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray
) -> tuple[RandomForestClassifier, np.ndarray]:
    """5-fold stratified CV with a Random Forest; returns best model + OOF predictions."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y), dtype=int)
    qwk_scores: list[float] = []
    models: list[RandomForestClassifier] = []

    fold_iter = tqdm(
        enumerate(skf.split(X, y), start=1),
        total=N_FOLDS,
        desc="CV folds",
    )
    for fold, (train_idx, val_idx) in fold_iter:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # class_weight="balanced" is already set in RF_PARAMS, but we also pass
        # explicit sample weights so the split criterion uses them correctly.
        sample_weights = compute_sample_weight("balanced", y_tr)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr, sample_weight=sample_weights)

        val_preds = clf.predict(X_val)
        oof_preds[val_idx] = val_preds
        score = quadratic_weighted_kappa(y_val, val_preds)
        qwk_scores.append(score)
        fold_iter.set_postfix({"QWK": f"{score:.4f}"})
        print_qwk(fold, score)
        models.append(clf)

    print_cv_summary(qwk_scores)
    best_model = models[int(np.argmax(qwk_scores))]
    return best_model, oof_preds


def run(force: bool = False) -> None:
    rf_submission_path = Path("submission_rf.csv")
    rf_model_path = CACHE_DIR / "rf_model.pkl"

    if not force and rf_submission_path.exists() and rf_model_path.exists():
        print("Random Forest model and submission already exist. Use --force to retrain.")
        return

    print("Loading cached tabular features...")
    X_train, y_train, X_test, test_pet_ids, col_names = load_data()

    print(f"\nTraining Random Forest with {N_FOLDS}-fold CV...")
    best_model, oof_preds = train_and_evaluate(X_train, y_train)

    print_oof_metrics(oof_preds, y_train)

    print("\nSaving model and OOF predictions...")
    CACHE_DIR.mkdir(exist_ok=True)
    with open(rf_model_path, "wb") as f:
        pickle.dump(best_model, f)
    np.save(CACHE_DIR / "rf_oof_predictions.npy", oof_preds)
    np.save(CACHE_DIR / "rf_oof_labels.npy", y_train)
    np.save(CACHE_DIR / "rf_col_names.npy", np.array(col_names, dtype=object))
    print(f"Saved: {rf_model_path}")

    # print("\nGenerating test predictions...")
    # test_preds = best_model.predict(X_test)

    # submission = pd.DataFrame(
    #     {"PetID": test_pet_ids, "AdoptionSpeed": test_preds.astype(int)}
    # )
    # submission.to_csv(rf_submission_path, index=False)
    print(f"Saved: {rf_submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest on the preprocessed tabular features."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache and retrain")
    args = parser.parse_args()
    run(force=args.force)