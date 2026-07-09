"""Hyperparameter tuning for Random Forest and LightGBM.

* **LightGBM** tuning is delegated to
  :func:`src.lgbm.tune_hyperparameters` (Optuna with 5-fold stratified CV
  inside each trial), so the tuning behaviour matches the main pipeline
  exactly.

* **Random Forest** tuning is implemented locally (the main pipeline
  doesn't tune RF).  It uses Optuna with 5-fold stratified CV inside
  each trial and optimises Quadratic Weighted Kappa.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src import lgbm as lgbm_mod

RANDOM_STATE = 42
N_FOLDS = 5


def _qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ----------------------------------------------------------------------
# Random Forest tuning (5-fold CV, Optuna)
# ----------------------------------------------------------------------
def tune_random_forest(
    X_train,
    y_train,
    n_trials: int = 50,
    random_state: int = RANDOM_STATE,
    log_path: Path | None = None,
) -> tuple[dict, float, float]:
    """Return ``(best_params, best_qwk, tuning_time_seconds)``."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.75]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
            "random_state": random_state,
            "n_jobs": -1,
        }
        skf = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=random_state
        )
        scores: list[float] = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            sw = compute_sample_weight("balanced", y_tr)
            clf = RandomForestClassifier(**params)
            clf.fit(X_tr, y_tr, sample_weight=sw)
            scores.append(_qwk(y_val, clf.predict(X_val)))
        return float(np.mean(scores))

    return _run_study(objective, n_trials, log_path, study_name="rf_tuning")


# ----------------------------------------------------------------------
# LightGBM tuning — delegate to src.lgbm.tune_hyperparameters
# ----------------------------------------------------------------------
def tune_lightgbm(
    X_train,
    y_train,
    feature_names: list[str],
    n_trials: int = 50,
    use_smote: bool = False,
    log_path: Path | None = None,
    mode: str | None = None,
) -> tuple[dict, float, float]:
    """Wrapper around :func:`src.lgbm.tune_hyperparameters`.

    Uses Optuna with 5-fold stratified CV inside each trial (identical
    behaviour to running ``main.py --tune``).

    Returns ``(best_params, best_qwk_or_nan, tuning_time_seconds)``.
    ``best_qwk`` is not surfaced by the upstream function; it is
    returned here as ``nan`` for backwards compatibility.
    """
    n_classes = int(len(np.unique(y_train)))
    if mode is None:
        mode = "all_multiclass" if n_classes == 5 else "all_4class"

    t0 = time.perf_counter()
    best_params = lgbm_mod.tune_hyperparameters(
        X_train,
        y_train,
        feature_names,
        mode=mode,
        n_trials=n_trials,
        imbalance_strategy="balanced",
        use_smote=use_smote,
    )
    tuning_time = time.perf_counter() - t0

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(
                {
                    "study_name": f"lgbm_{mode}_tuning",
                    "n_trials": n_trials,
                    "best_params": best_params,
                    "tuning_time_seconds": tuning_time,
                    "note": "Full trial history is not surfaced by "
                            "src.lgbm.tune_hyperparameters; the checkpointed "
                            "best params live at "
                            f"cache/lgbm_best_params_{mode}.json",
                },
                f,
                indent=2,
                default=str,
            )

    return best_params, float("nan"), tuning_time


# ----------------------------------------------------------------------
# Shared Optuna study runner (used by tune_random_forest)
# ----------------------------------------------------------------------
def _run_study(
    objective,
    n_trials: int,
    log_path: Path | None,
    study_name: str,
) -> tuple[dict, float, float]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    def _print_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(
            f"  [{study_name}] trial {trial.number:3d}/{n_trials}  "
            f"QWK={trial.value:.4f}  best={study_.best_value:.4f}"
        )

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[_print_trial])
    tuning_time = time.perf_counter() - t0

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "study_name": study_name,
            "n_trials": n_trials,
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "tuning_time_seconds": tuning_time,
            "trials": [
                {
                    "number": t.number,
                    "value": (float(t.value) if t.value is not None else None),
                    "params": t.params,
                    "state": t.state.name,
                }
                for t in study.trials
            ],
        }
        with open(log_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    return study.best_params, float(study.best_value), tuning_time