import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.utils import print_cv_summary, print_qwk, quadratic_weighted_kappa

CACHE_DIR = Path("cache")
N_FOLDS = 5
RANDOM_STATE = 42
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
N_TUNING_TRIALS = 50  # Number of Optuna trials for hyperparameter search

# --- Mode configurations ---
# Must match cache_suffix in preprocessing.py MODES dict.
MODE_CONFIGS = {
    "all_multiclass": {
        "cache_suffix": "",
        "num_classes": 5,
        "labels": ["Same day", "1-7 days", "8-30 days", "31-90 days", ">100 days"],
        "objective": "multiclass",
        "metric": "multi_logloss",
    },
    "dogs_extreme": {
        "cache_suffix": "_dogs_extreme",
        "num_classes": 2,
        "labels": ["Same day", ">100 days"],
        "objective": "binary",
        "metric": "binary_logloss",
    },
    "dogs_month_vs_100": {
        "cache_suffix": "_dogs_month_vs_100",
        "num_classes": 2,
        "labels": ["8-30 days", ">100 days"],
        "objective": "binary",
        "metric": "binary_logloss",
    },
    "dogs_adjacent": {
        "cache_suffix": "_dogs_adjacent",
        "num_classes": 2,
        "labels": ["8-30 days", "31-90 days"],
        "objective": "binary",
        "metric": "binary_logloss",
    },
    "cats_month_vs_100": {
        "cache_suffix": "_cats_month_vs_100",
        "num_classes": 2,
        "labels": ["8-30 days", ">100 days"],
        "objective": "binary",
        "metric": "binary_logloss",
    },
}

# Default parameters for multiclass (used when not tuning)
LGBM_PARAMS_MULTICLASS = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "class_weight": "balanced",
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "n_jobs": -1,
}

# Default parameters for binary classification
LGBM_PARAMS_BINARY = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "is_unbalance": True,  # Handle class imbalance for binary
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "n_jobs": -1,
}


def get_default_params(mode: str) -> dict:
    """Get default LightGBM parameters for the given mode."""
    if MODE_CONFIGS[mode]["num_classes"] == 2:
        return LGBM_PARAMS_BINARY.copy()
    return LGBM_PARAMS_MULTICLASS.copy()


def load_data(mode: str = "full") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the preprocessed tabular features."""
    suffix = MODE_CONFIGS[mode]["cache_suffix"]
    print(f"  Loading tabular features from preprocessing cache ({mode} mode)...")
    train_feat = pd.read_parquet(CACHE_DIR / f"train_features{suffix}.parquet")
    test_feat = pd.read_parquet(CACHE_DIR / f"test_features{suffix}.parquet")

    y = train_feat["AdoptionSpeed"].values.astype(int)
    test_pet_ids = test_feat["PetID"].tolist()

    feature_cols = [c for c in train_feat.columns if c not in ("AdoptionSpeed", "PetID")]
    X_train = train_feat[feature_cols].values.astype(np.float32)
    X_test = test_feat[feature_cols].values.astype(np.float32)

    print(f"  Feature matrix: train={X_train.shape}  test={X_test.shape}")
    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Classes: {np.unique(y)} (n_classes={len(np.unique(y))})")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X_train, y, X_test, test_pet_ids, feature_cols


def find_optimal_threshold(oof_probs: np.ndarray, oof_labels: np.ndarray,
                           labels: list[str]) -> float:
    """Find the threshold that maximizes F1-macro on OOF predictions.

    Searches thresholds from 0.05 to 0.95 and returns the one that gives
    the best balanced performance between both classes.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0

    for thresh in thresholds:
        preds = (oof_probs >= thresh).astype(int)
        f1 = f1_score(oof_labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Print results at optimal threshold
    opt_preds = (oof_probs >= best_threshold).astype(int)
    acc = accuracy_score(oof_labels, opt_preds)
    f1_macro = f1_score(oof_labels, opt_preds, average="macro")
    auc = roc_auc_score(oof_labels, oof_probs)

    print(f"\n{'='*60}")
    print(f"  Optimal Threshold Search (maximizing F1-macro)")
    print(f"{'='*60}")
    print(f"  Optimal threshold:  {best_threshold:.2f}  (default was 0.50)")
    print(f"  AUC-ROC:            {auc:.4f}  (threshold-independent)")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  F1 macro:           {f1_macro:.4f}")

    print(f"\n  --- Per-Class Report (threshold={best_threshold:.2f}) ---")
    print(classification_report(oof_labels, opt_preds, target_names=labels))

    print(f"  --- Confusion Matrix (threshold={best_threshold:.2f}) ---")
    cm = confusion_matrix(oof_labels, opt_preds)
    col_width = max(len(lbl) for lbl in labels) + 2
    header = " " * col_width + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_width}}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(labels)))
        print(row)
    print()

    return best_threshold


def print_oof_metrics(oof_preds: np.ndarray, oof_labels: np.ndarray, mode: str = "full",
                      oof_probs: np.ndarray | None = None) -> None:
    """Print full evaluation metrics on the out-of-fold predictions."""
    config = MODE_CONFIGS[mode]
    labels = config["labels"]
    n_classes = config["num_classes"]

    acc = accuracy_score(oof_labels, oof_preds)
    f1_macro = f1_score(oof_labels, oof_preds, average="macro")
    f1_weighted = f1_score(oof_labels, oof_preds, average="weighted")

    print("\n--- Out-of-Fold Evaluation (threshold=0.50) ---")
    print(f"  Accuracy:                       {acc:.4f}")
    print(f"  F1 macro:                       {f1_macro:.4f}")
    print(f"  F1 weighted:                    {f1_weighted:.4f}")

    if n_classes == 2:
        # Binary-specific metrics
        if oof_probs is not None:
            auc = roc_auc_score(oof_labels, oof_probs)
            print(f"  AUC-ROC:                        {auc:.4f}")
    else:
        # Multiclass-specific metrics
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

    # For binary mode: find and display optimal threshold
    if n_classes == 2 and oof_probs is not None:
        find_optimal_threshold(oof_probs, oof_labels, labels)


def cv_score(
    X: np.ndarray, y: np.ndarray, feature_cols: list[str], params: dict, mode: str = "full"
) -> float:
    """Run K-fold CV and return mean score (QWK for multiclass, AUC for binary)."""
    config = MODE_CONFIGS[mode]
    n_classes = config["num_classes"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for _, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        val_raw = model.predict(X_val, num_iteration=model.best_iteration)

        if n_classes == 2:
            # Binary: val_raw is probabilities for class 1
            val_pred = (val_raw > 0.5).astype(int)
            score = roc_auc_score(y_val, val_raw)
        else:
            # Multiclass: val_raw is (n_samples, n_classes)
            val_pred = np.argmax(val_raw, axis=1)
            score = quadratic_weighted_kappa(y_val, val_pred)

        scores.append(score)

    return float(np.mean(scores))


def tune_hyperparameters(
    X: np.ndarray, y: np.ndarray, feature_cols: list[str],
    mode: str = "full", n_trials: int = N_TUNING_TRIALS
) -> dict:
    """Use Optuna to find the best LightGBM hyperparameters."""
    config = MODE_CONFIGS[mode]
    n_classes = config["num_classes"]
    metric_name = "AUC" if n_classes == 2 else "QWK"

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": config["objective"],
            "metric": config["metric"],
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "n_jobs": -1,
            # --- Tunable parameters ---
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        if n_classes > 2:
            params["num_class"] = n_classes
            params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", None])
        else:
            params["is_unbalance"] = trial.suggest_categorical("is_unbalance", [True, False])

        mean_score = cv_score(X, y, feature_cols, params, mode=mode)
        return mean_score

    # Suppress Optuna's info messages
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"lgbm_{mode}_tuning",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    print(f"\n{'='*60}")
    print(f"  Optuna Hyperparameter Tuning ({n_trials} trials, {N_FOLDS}-fold CV)")
    print(f"  Mode: {mode} | Metric: {metric_name}")
    print(f"{'='*60}\n")

    def print_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(
            f"  Trial {trial.number:3d}/{n_trials}  "
            f"{metric_name}={trial.value:.4f}  "
            f"Best={study.best_value:.4f}"
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[print_trial])

    print(f"\n{'='*60}")
    print(f"  Tuning Complete!")
    print(f"  Best {metric_name}: {study.best_value:.4f}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"{'='*60}")
    print("\n  Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print()

    # Build the full params dict from the best trial
    best_params = {
        "objective": config["objective"],
        "metric": config["metric"],
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "n_jobs": -1,
        **study.best_params,
    }
    if n_classes > 2:
        best_params["num_class"] = n_classes

    # Save best params to disk
    params_path = CACHE_DIR / f"lgbm_best_params_{mode}.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"  Saved best params: {params_path}")

    return best_params


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, feature_cols: list[str],
    params: dict | None = None, mode: str = "full"
) -> tuple[lgb.Booster, np.ndarray, np.ndarray | None]:
    """K-fold stratified CV with LightGBM; returns best model + OOF predictions + OOF probs."""
    config = MODE_CONFIGS[mode]
    n_classes = config["num_classes"]

    if params is None:
        params = get_default_params(mode)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y), dtype=int)

    # For binary: store probabilities for AUC calculation
    if n_classes == 2:
        oof_probs = np.zeros(len(y), dtype=np.float64)
    else:
        oof_probs = np.zeros((len(y), n_classes), dtype=np.float64)

    scores: list[float] = []
    models: list[lgb.Booster] = []
    metric_name = "AUC" if n_classes == 2 else "QWK"

    fold_iter = tqdm(
        enumerate(skf.split(X, y), start=1),
        total=N_FOLDS,
        desc="CV folds",
    )
    for fold, (train_idx, val_idx) in fold_iter:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        val_raw = model.predict(X_val, num_iteration=model.best_iteration)

        if n_classes == 2:
            val_pred = (val_raw > 0.5).astype(int)
            oof_probs[val_idx] = val_raw
            score = roc_auc_score(y_val, val_raw)
        else:
            val_pred = np.argmax(val_raw, axis=1)
            oof_probs[val_idx] = val_raw
            score = quadratic_weighted_kappa(y_val, val_pred)

        oof_preds[val_idx] = val_pred
        scores.append(score)
        fold_iter.set_postfix({metric_name: f"{score:.4f}"})
        print(f"  Fold {fold}  {metric_name}: {score:.4f}")
        models.append(model)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\n  CV {metric_name}: {mean_score:.4f} +/- {std_score:.4f}")

    best_model = models[int(np.argmax(scores))]
    return best_model, oof_preds, oof_probs if n_classes == 2 else None


def run(force: bool = False, tune: bool = False, mode: str = "all_multiclass",
        n_trials: int = N_TUNING_TRIALS) -> None:
    """Run LightGBM training pipeline.

    Args:
        force: Retrain even if cache exists
        tune: Run Optuna hyperparameter tuning
        mode: One of the keys in MODE_CONFIGS (e.g. "all_multiclass", "dogs_extreme", etc.)
        n_trials: Number of Optuna trials
    """
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODE_CONFIGS.keys())}")

    config = MODE_CONFIGS[mode]
    suffix = config["cache_suffix"]
    lgbm_model_path = CACHE_DIR / f"lgbm_model{suffix}.pkl"

    if not force and not tune and lgbm_model_path.exists():
        print(f"LightGBM model already exists ({mode} mode). Use --force to retrain.")
        return

    print(f"\n{'='*60}")
    print(f"  LightGBM Training — Mode: {mode}")
    print(f"  {config['labels']}")
    print(f"{'='*60}\n")

    print("Loading cached tabular features...")
    X_train, y_train, X_test, test_pet_ids, col_names = load_data(mode=mode)

    # Determine which parameters to use
    params = get_default_params(mode)
    params_path = CACHE_DIR / f"lgbm_best_params_{mode}.json"

    if tune:
        params = tune_hyperparameters(X_train, y_train, col_names, mode=mode, n_trials=n_trials)
    elif params_path.exists():
        print(f"\n  Loading tuned params from {params_path}")
        with open(params_path) as f:
            params = json.load(f)
        print(f"  Using tuned parameters (run with --tune to re-tune)")
    else:
        print("\n  Using default parameters (run with --tune to optimize)")

    print(f"\nTraining LightGBM with {N_FOLDS}-fold CV...")
    best_model, oof_preds, oof_probs = train_and_evaluate(
        X_train, y_train, col_names, params, mode=mode
    )

    print_oof_metrics(oof_preds, y_train, mode=mode, oof_probs=oof_probs)

    print("\nSaving model and OOF predictions...")
    CACHE_DIR.mkdir(exist_ok=True)
    with open(lgbm_model_path, "wb") as f:
        pickle.dump(best_model, f)
    np.save(CACHE_DIR / f"lgbm_oof_predictions{suffix}.npy", oof_preds)
    np.save(CACHE_DIR / f"lgbm_oof_labels{suffix}.npy", y_train)
    np.save(CACHE_DIR / f"lgbm_col_names{suffix}.npy", np.array(col_names, dtype=object))
    if oof_probs is not None:
        np.save(CACHE_DIR / f"lgbm_oof_probs{suffix}.npy", oof_probs)
    print(f"  Saved: {lgbm_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model on the preprocessed tabular features."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache and retrain")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--mode", choices=list(MODE_CONFIGS.keys()), default="all_multiclass",
                        help="Data subset/task mode (default: all_multiclass)")
    parser.add_argument(
        "--n-trials", type=int, default=N_TUNING_TRIALS,
        help=f"Number of Optuna trials (default: {N_TUNING_TRIALS})"
    )
    args = parser.parse_args()
    run(force=args.force, tune=args.tune, mode=args.mode, n_trials=args.n_trials)
