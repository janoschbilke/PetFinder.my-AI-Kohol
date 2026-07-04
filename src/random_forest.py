"""Random Forest training pipeline with mode/experiment support.

Mirrors the LightGBM interface so it can be driven by the same experimental
orchestrator (`src/experimental_pipeline.py`).
"""

import argparse
import json
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
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from src.utils import quadratic_weighted_kappa

CACHE_DIR = Path("cache")
N_FOLDS = 5
RANDOM_STATE = 42

MODE_CONFIGS = {
    "all_multiclass": {
        "cache_suffix": "",
        "num_classes": 5,
        "labels": ["Same day", "1-7 days", "8-30 days", "31-90 days", ">100 days"],
        "original_classes": [0, 1, 2, 3, 4],
    },
    "all_4class": {
        "cache_suffix": "_4class",
        "num_classes": 4,
        "labels": ["1-7 days", "8-30 days", "31-90 days", ">100 days"],
        "original_classes": [1, 2, 3, 4],
    },
    "dogs_extreme": {
        "cache_suffix": "_dogs_extreme",
        "num_classes": 2,
        "labels": ["Same day", ">100 days"],
        "original_classes": [0, 4],
    },
    "dogs_month_vs_100": {
        "cache_suffix": "_dogs_month_vs_100",
        "num_classes": 2,
        "labels": ["8-30 days", ">100 days"],
        "original_classes": [2, 4],
    },
    "dogs_adjacent": {
        "cache_suffix": "_dogs_adjacent",
        "num_classes": 2,
        "labels": ["8-30 days", "31-90 days"],
        "original_classes": [2, 3],
    },
    "cats_month_vs_100": {
        "cache_suffix": "_cats_month_vs_100",
        "num_classes": 2,
        "labels": ["8-30 days", ">100 days"],
        "original_classes": [2, 4],
    },
}

RF_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 15, 25],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt"],
}


def load_data(mode="all_multiclass", feature_suffix=None):
    suffix = feature_suffix if feature_suffix is not None else MODE_CONFIGS[mode]["cache_suffix"]
    print(f"  Loading features from cache ({mode} / suffix='{suffix}')...")
    train_feat = pd.read_parquet(CACHE_DIR / f"train_features{suffix}.parquet")
    test_feat = pd.read_parquet(CACHE_DIR / f"test_features{suffix}.parquet")

    y = train_feat["AdoptionSpeed"].values.astype(int)
    test_pet_ids = test_feat["PetID"].tolist()

    feature_cols = [c for c in train_feat.columns if c not in ("AdoptionSpeed", "PetID")]
    X_train = train_feat[feature_cols].values.astype(np.float32)
    X_test = test_feat[feature_cols].values.astype(np.float32)

    print(f"  Feature matrix: train={X_train.shape}  test={X_test.shape}")
    print(f"  Classes: {np.unique(y)} (n_classes={len(np.unique(y))})")
    return X_train, y, X_test, test_pet_ids, feature_cols


def print_oof_metrics(oof_preds, oof_labels, mode, oof_probs=None):
    config = MODE_CONFIGS[mode]
    labels = config["labels"]
    n_classes = config["num_classes"]

    acc = accuracy_score(oof_labels, oof_preds)
    f1_macro = f1_score(oof_labels, oof_preds, average="macro")
    f1_weighted = f1_score(oof_labels, oof_preds, average="weighted")

    print("\n--- Out-of-Fold Evaluation ---")
    print(f"  Accuracy:                       {acc:.4f}")
    print(f"  F1 macro:                       {f1_macro:.4f}")
    print(f"  F1 weighted:                    {f1_weighted:.4f}")
    if n_classes == 2 and oof_probs is not None:
        auc = roc_auc_score(oof_labels, oof_probs)
        print(f"  AUC-ROC:                        {auc:.4f}")
    else:
        qwk = quadratic_weighted_kappa(oof_labels, oof_preds)
        print(f"  QWK (Quadratic Weighted Kappa): {qwk:.4f}")

    print("\n--- Per-Class Report ---")
    print(classification_report(oof_labels, oof_preds, target_names=labels))

    print("--- Confusion Matrix (counts) ---")
    cm = confusion_matrix(oof_labels, oof_preds)
    col_width = max(len(lbl) for lbl in labels) + 2
    header = " " * col_width + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_width}}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(n_classes))
        print(row)
    print()


def tune_hyperparameters(X, y, mode="all_multiclass"):
    from sklearn.metrics import make_scorer
    config = MODE_CONFIGS[mode]
    n_classes = config["num_classes"]
    scorer = "roc_auc" if n_classes == 2 else make_scorer(quadratic_weighted_kappa)

    base_params = {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    base_est = RandomForestClassifier(**base_params)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n{'='*60}")
    print(f"  RF Grid Search  (mode={mode})")
    print(f"  Grid: {RF_GRID}")
    print(f"{'='*60}")

    gs = GridSearchCV(base_est, RF_GRID, scoring=scorer, cv=skf, n_jobs=-1, verbose=1)
    gs.fit(X, y)

    print(f"\n  Best score: {gs.best_score_:.4f}")
    print(f"  Best params: {gs.best_params_}")

    best_params = {**base_params, **gs.best_params_}
    return best_params


def train_and_evaluate(X, y, mode="all_multiclass", params=None, use_smote=False):
    config = MODE_CONFIGS[mode]
    n_classes = config["num_classes"]

    if params is None:
        params = RF_DEFAULT_PARAMS.copy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y), dtype=int)
    if n_classes == 2:
        oof_probs = np.zeros(len(y), dtype=np.float64)
    else:
        oof_probs = np.zeros((len(y), n_classes), dtype=np.float64)

    scores = []
    models = []
    metric_name = "AUC" if n_classes == 2 else "QWK"

    fold_iter = tqdm(
        enumerate(skf.split(X, y), start=1),
        total=N_FOLDS,
        desc="RF CV folds",
    )
    for fold, (train_idx, val_idx) in fold_iter:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if use_smote:
            from imblearn.over_sampling import SMOTE
            k = max(1, min(5, int(np.bincount(y_tr).min()) - 1))
            X_tr, y_tr = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(X_tr, y_tr)

        sample_weights = compute_sample_weight("balanced", y_tr)
        clf = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr, sample_weight=sample_weights)

        val_preds = clf.predict(X_val)
        val_proba = clf.predict_proba(X_val)
        oof_preds[val_idx] = val_preds

        if n_classes == 2:
            oof_probs[val_idx] = val_proba[:, 1]
            score = roc_auc_score(y_val, val_proba[:, 1])
        else:
            proba_full = np.zeros((len(val_idx), n_classes), dtype=np.float64)
            for i, cls in enumerate(clf.classes_):
                proba_full[:, int(cls)] = val_proba[:, i]
            oof_probs[val_idx] = proba_full
            score = quadratic_weighted_kappa(y_val, val_preds)

        scores.append(score)
        fold_iter.set_postfix({metric_name: f"{score:.4f}"})
        print(f"  Fold {fold}  {metric_name}: {score:.4f}")
        models.append(clf)

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    print(f"\n  CV {metric_name}: {mean_score:.4f} +/- {std_score:.4f}")

    best_model = models[int(np.argmax(scores))]
    return best_model, oof_preds, oof_probs, scores


def run(
    force=False,
    tune=False,
    mode="all_multiclass",
    feature_suffix=None,
    experiment_id=None,
    use_smote=False,
):
    """Run RF training pipeline. Returns metrics dict."""
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODE_CONFIGS.keys())}")

    config = MODE_CONFIGS[mode]
    out_suffix = experiment_id if experiment_id is not None else config["cache_suffix"]
    rf_model_path = CACHE_DIR / f"rf_model{out_suffix}.pkl"
    metrics_path = CACHE_DIR / f"rf_metrics{out_suffix}.json"

    if not force and not tune and rf_model_path.exists() and metrics_path.exists():
        print(f"RF model already exists ({out_suffix}). Use --force to retrain.")
        with open(metrics_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Random Forest Training - Mode: {mode}  |  ID: {out_suffix}")
    print(f"  {config['labels']}")
    print(f"{'='*60}\n")

    print("Loading cached tabular features...")
    X_train, y_train, X_test, test_pet_ids, col_names = load_data(mode=mode, feature_suffix=feature_suffix)

    # Determine which parameters to use
    params = RF_DEFAULT_PARAMS.copy()
    params_path = CACHE_DIR / f"rf_best_params{out_suffix}.json"

    if tune:
        params = tune_hyperparameters(X_train, y_train, mode=mode)
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2, default=str)
        print(f"  Saved tuned params: {params_path}")
    elif params_path.exists():
        print(f"\n  Loading tuned params from {params_path}")
        with open(params_path) as f:
            params = json.load(f)
        # class_weight and other non-serializable defaults are OK as strings here
        params.setdefault("random_state", RANDOM_STATE)
        params.setdefault("n_jobs", -1)
        print(f"  Using tuned parameters (run with tune=True to re-tune)")
    else:
        print("\n  Using default parameters (pass tune=True to optimize)")

    print(f"\nTraining Random Forest with {N_FOLDS}-fold CV...")
    best_model, oof_preds, oof_probs, fold_scores = train_and_evaluate(
        X_train, y_train, mode=mode, params=params, use_smote=use_smote,
    )

    print_oof_metrics(oof_preds, y_train, mode=mode, oof_probs=oof_probs)

    n_classes = config["num_classes"]
    metrics = {
        "mode": mode,
        "out_suffix": out_suffix,
        "model": "random_forest",
        "use_smote": use_smote,
        "accuracy": float(accuracy_score(y_train, oof_preds)),
        "f1_macro": float(f1_score(y_train, oof_preds, average="macro")),
        "f1_weighted": float(f1_score(y_train, oof_preds, average="weighted")),
        "fold_scores": [float(s) for s in fold_scores],
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "n_features": int(X_train.shape[1]),
        "n_samples": int(X_train.shape[0]),
        "params": {k: (str(v) if v is None or not isinstance(v, (int, float, str, bool, list)) else v)
                   for k, v in params.items()},
    }
    if n_classes > 2:
        metrics["qwk"] = float(quadratic_weighted_kappa(y_train, oof_preds))
    if n_classes == 2 and oof_probs is not None:
        metrics["auc"] = float(roc_auc_score(y_train, oof_probs))

    print("\nSaving model and OOF predictions...")
    CACHE_DIR.mkdir(exist_ok=True)
    with open(rf_model_path, "wb") as f:
        pickle.dump(best_model, f)
    np.save(CACHE_DIR / f"rf_oof_predictions{out_suffix}.npy", oof_preds)
    np.save(CACHE_DIR / f"rf_oof_labels{out_suffix}.npy", y_train)
    np.save(CACHE_DIR / f"rf_col_names{out_suffix}.npy", np.array(col_names, dtype=object))
    if oof_probs is not None:
        np.save(CACHE_DIR / f"rf_oof_probs{out_suffix}.npy", oof_probs)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {rf_model_path}")
    print(f"  Saved: {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest on preprocessed features.")
    parser.add_argument("--force", action="store_true", help="Ignore cache and retrain")
    parser.add_argument("--tune", action="store_true", help="Run grid-search hyperparameter tuning")
    parser.add_argument("--mode", choices=list(MODE_CONFIGS.keys()), default="all_multiclass")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE inside each CV fold")
    parser.add_argument("--feature-suffix", default=None,
                        help="Override the parquet suffix used to load features")
    parser.add_argument("--experiment-id", default=None,
                        help="Suffix for output artefacts (model, metrics, OOF files)")
    args = parser.parse_args()
    run(
        force=args.force, tune=args.tune, mode=args.mode,
        feature_suffix=args.feature_suffix, experiment_id=args.experiment_id,
        use_smote=args.use_smote,
    )
