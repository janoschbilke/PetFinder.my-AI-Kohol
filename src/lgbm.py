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

ADOPTION_SPEED_LABELS = ["Same day", "1-7 days", "8-30 days", "31-90 days", ">100 days"]

# Default parameters (used when not tuning)
LGBM_PARAMS = {
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
    col_width = 11
    header = " " * col_width + "".join(f"{lbl:>{col_width}}" for lbl in ADOPTION_SPEED_LABELS)
    print(header)
    for i, row_label in enumerate(ADOPTION_SPEED_LABELS):
        row = f"{row_label:>{col_width}}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(5))
        print(row)
    print()


def cv_score(
    X: np.ndarray, y: np.ndarray, feature_cols: list[str], params: dict
) -> float:
    """Run K-fold CV and return mean QWK score (used by both tuning and final training)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    qwk_scores: list[float] = []

    for _, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=EARLY_STOPPING_ROUNDS),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        val_prob = model.predict(X_val, num_iteration=model.best_iteration)
        val_pred = np.argmax(val_prob, axis=1)
        score = quadratic_weighted_kappa(y_val, val_pred)
        qwk_scores.append(score)

    return float(np.mean(qwk_scores))


def tune_hyperparameters(
    X: np.ndarray, y: np.ndarray, feature_cols: list[str], n_trials: int = N_TUNING_TRIALS
) -> dict:
    """Use Optuna to find the best LightGBM hyperparameters maximizing QWK."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
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
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }

        mean_qwk = cv_score(X, y, feature_cols, params)
        return mean_qwk

    # Suppress Optuna's info messages, show only warnings+
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_qwk_tuning",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    print(f"\n{'='*60}")
    print(f"  Optuna Hyperparameter Tuning ({n_trials} trials, {N_FOLDS}-fold CV)")
    print(f"{'='*60}\n")

    # Custom callback to print progress
    def print_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(
            f"  Trial {trial.number:3d}/{n_trials}  "
            f"QWK={trial.value:.4f}  "
            f"Best={study.best_value:.4f}"
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[print_trial])

    print(f"\n{'='*60}")
    print(f"  Tuning Complete!")
    print(f"  Best QWK: {study.best_value:.4f}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"{'='*60}")
    print("\n  Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print()

    # Build the full params dict from the best trial
    best_params = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "n_jobs": -1,
        **study.best_params,
    }

    # Save best params to disk
    params_path = CACHE_DIR / "lgbm_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"  Saved best params: {params_path}")

    return best_params


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, feature_cols: list[str], params: dict | None = None
) -> tuple[lgb.Booster, np.ndarray]:
    """5-fold stratified CV with LightGBM; returns best model + OOF predictions."""
    if params is None:
        params = LGBM_PARAMS

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), 5), dtype=np.float64)
    qwk_scores: list[float] = []
    models: list[lgb.Booster] = []

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
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=EARLY_STOPPING_ROUNDS),  # suppress per-iteration logs
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        val_prob = model.predict(X_val, num_iteration=model.best_iteration)
        val_pred = np.argmax(val_prob, axis=1)

        oof_preds[val_idx] = val_pred
        oof_probs[val_idx] = val_prob

        score = quadratic_weighted_kappa(y_val, val_pred)
        qwk_scores.append(score)
        fold_iter.set_postfix({"QWK": f"{score:.4f}"})
        print_qwk(fold, score)
        models.append(model)

    print_cv_summary(qwk_scores)
    best_model = models[int(np.argmax(qwk_scores))]
    return best_model, oof_preds


def run(force: bool = False, tune: bool = False, n_trials: int = N_TUNING_TRIALS) -> None:
    lgbm_submission_path = Path("submission_lgbm.csv")
    lgbm_model_path = CACHE_DIR / "lgbm_model.pkl"

    if not force and not tune and lgbm_submission_path.exists() and lgbm_model_path.exists():
        print("LightGBM model and submission already exist. Use --force to retrain.")
        return

    print("Loading cached tabular features...")
    X_train, y_train, X_test, test_pet_ids, col_names = load_data()

    # Determine which parameters to use
    params = LGBM_PARAMS
    params_path = CACHE_DIR / "lgbm_best_params.json"

    if tune:
        # Run hyperparameter tuning
        params = tune_hyperparameters(X_train, y_train, col_names, n_trials=n_trials)
    elif params_path.exists():
        # Load previously tuned parameters
        print(f"\n  Loading tuned params from {params_path}")
        with open(params_path) as f:
            params = json.load(f)
        print(f"  Using tuned parameters (run with --tune to re-tune)")
    else:
        print("\n  Using default parameters (run with --tune to optimize)")

    print(f"\nTraining LightGBM with {N_FOLDS}-fold CV...")
    best_model, oof_preds = train_and_evaluate(X_train, y_train, col_names, params)

    print_oof_metrics(oof_preds, y_train)

    print("\nSaving model and OOF predictions...")
    CACHE_DIR.mkdir(exist_ok=True)
    with open(lgbm_model_path, "wb") as f:
        pickle.dump(best_model, f)
    np.save(CACHE_DIR / "lgbm_oof_predictions.npy", oof_preds)
    np.save(CACHE_DIR / "lgbm_oof_labels.npy", y_train)
    np.save(CACHE_DIR / "lgbm_col_names.npy", np.array(col_names, dtype=object))
    print(f"Saved: {lgbm_model_path}")

    # print("\nGenerating test predictions...")
    # test_probs = best_model.predict(X_test, num_iteration=best_model.best_iteration)
    # test_preds = np.argmax(test_probs, axis=1)

    # submission = pd.DataFrame(
    #     {"PetID": test_pet_ids, "AdoptionSpeed": test_preds.astype(int)}
    # )
    # submission.to_csv(lgbm_submission_path, index=False)
    # print(f"Saved: {lgbm_submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model on the preprocessed tabular features."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache and retrain")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument(
        "--n-trials", type=int, default=N_TUNING_TRIALS,
        help=f"Number of Optuna trials (default: {N_TUNING_TRIALS})"
    )
    args = parser.parse_args()
    run(force=args.force, tune=args.tune, n_trials=args.n_trials)
