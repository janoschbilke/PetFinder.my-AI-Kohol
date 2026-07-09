"""Optuna-based hyperparameter tuning for Random Forest and LightGBM.

Optimises Quadratic Weighted Kappa on a held-out validation split
(the same split the final evaluation is done on). This mimics a plain
single-split tuning setup, which keeps the total runtime for 6 tuned
models manageable while still being reproducible.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

RANDOM_STATE = 42


def _qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------
def tune_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
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
        }
        model = RandomForestClassifier(
            **params, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return _qwk(y_val, preds)

    return _run_study(objective, n_trials, log_path, study_name="rf_tuning")


# ----------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------
def tune_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 50,
    random_state: int = RANDOM_STATE,
    log_path: Path | None = None,
    fixed_class_weight: str | None | dict = "unset",
) -> tuple[dict, float, float]:
    """Tune LightGBM hyperparameters.

    ``fixed_class_weight``:
        * ``"unset"``  → let Optuna choose between ``"balanced"`` and ``None``
        * ``"balanced"`` / ``None`` / dict → fix, don't tune
          (useful for SMOTE-balanced training data)
    """
    n_classes = int(len(np.unique(y_train)))

    def objective(trial: optuna.Trial) -> float:
        params: dict = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes

        if fixed_class_weight == "unset":
            params["class_weight"] = trial.suggest_categorical(
                "class_weight", ["balanced", None]
            )
        else:
            params["class_weight"] = fixed_class_weight

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return _qwk(y_val, preds)

    return _run_study(objective, n_trials, log_path, study_name="lgbm_tuning")


# ----------------------------------------------------------------------
# Shared study runner
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

    def _print_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(
            f"  [{study_name}] trial {trial.number:3d}/{n_trials}  "
            f"QWK={trial.value:.4f}  best={study.best_value:.4f}"
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