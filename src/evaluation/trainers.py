"""Train + benchmark a single model configuration.

This module **delegates** the actual training to the existing pipeline
implementations in :mod:`src.random_forest` and :mod:`src.lgbm` so the
metrics we report here are directly comparable to what the main
pipeline produces.

For every model we:

1. Run the shared 5-fold stratified CV via the existing implementations
   on the training slice of :class:`FeatureBundle`.
2. Measure wall-clock **training time** around the CV call
   (``time.perf_counter``).
3. Compute OOF **accuracy / QWK / F1 / precision / recall** on the CV
   predictions.
4. Use the best CV model to benchmark **inference time** on the held-out
   validation slice (never seen by any fold).
5. Persist the model to disk to record its size.
"""
from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from src import lgbm as lgbm_mod
from src import random_forest as rf_mod
from src.evaluation.features import FeatureBundle
from src.evaluation.results_manager import ModelResult
from src.utils import quadratic_weighted_kappa

RANDOM_STATE = 42


def _benchmark_inference(X_val, predict_fn, warmup: int = 1):
    n = len(X_val)
    slice_ = X_val.iloc[: min(16, n)] if hasattr(X_val, "iloc") else X_val[: min(16, n)]
    for _ in range(warmup):
        _ = predict_fn(slice_)
    t0 = time.perf_counter()
    preds = predict_fn(X_val)
    total = time.perf_counter() - t0
    per_sample_ms = (total / n) * 1000.0
    return preds, total, per_sample_ms


def _model_size_mb(model: Any, tmp_dir: Path) -> float:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "tmp_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size = path.stat().st_size / (1024 * 1024)
    path.unlink(missing_ok=True)
    return size


def _fill_metrics_from_predictions(result: ModelResult, y_true, y_pred) -> None:
    result.accuracy = float(accuracy_score(y_true, y_pred))
    result.qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    result.f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    result.f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    result.precision_macro = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result.recall_macro = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------
def _rf_train_cv(X, y, params, feature_names):
    """5-fold stratified CV mirroring ``src.random_forest.train_and_evaluate``
    but with caller-supplied hyperparameters and DataFrame-based fits so
    feature names propagate.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight

    X_df = pd.DataFrame(X, columns=feature_names)
    skf = StratifiedKFold(
        n_splits=rf_mod.N_FOLDS, shuffle=True, random_state=rf_mod.RANDOM_STATE
    )
    oof_preds = np.zeros(len(y), dtype=int)
    scores: list[float] = []
    models = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_df, y), start=1):
        X_tr = X_df.iloc[tr_idx]
        X_val = X_df.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        sample_weights = compute_sample_weight("balanced", y_tr)
        clf = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr, sample_weight=sample_weights)
        preds = clf.predict(X_val)
        oof_preds[val_idx] = preds
        scores.append(quadratic_weighted_kappa(y_val, preds))
        models.append(clf)
        print(f"    RF fold {fold}/{rf_mod.N_FOLDS}  QWK={scores[-1]:.4f}")
    best = models[int(np.argmax(scores))]
    return best, oof_preds


def train_random_forest(bundle: FeatureBundle, result: ModelResult,
                        params: dict | None = None,
                        tmp_dir: Path | None = None) -> ModelResult:
    """5-fold CV training + inference benchmark for Random Forest.

    Defaults from ``src.random_forest.RF_PARAMS``; ``params`` overrides.
    """
    merged: dict[str, Any] = {**rf_mod.RF_PARAMS, **(params or {})}
    merged.setdefault("random_state", RANDOM_STATE)
    merged.setdefault("n_jobs", -1)

    t0 = time.perf_counter()
    best_model, oof_preds = _rf_train_cv(
        bundle.X_train, bundle.y_train, merged, bundle.feature_names
    )
    training_time = time.perf_counter() - t0

    _fill_metrics_from_predictions(result, bundle.y_train, oof_preds)

    X_val_df = pd.DataFrame(bundle.X_val, columns=bundle.feature_names)
    _preds, inf_total, inf_ms = _benchmark_inference(
        X_val_df, predict_fn=best_model.predict
    )

    result.training_time_seconds = float(training_time)
    result.inference_time_seconds_total = float(inf_total)
    result.inference_time_ms_per_sample = float(inf_ms)
    result.inference_batch_size = int(len(bundle.y_val))

    result.hyperparameters = {k: _json_safe(v) for k, v in merged.items()}
    result.n_train_samples = int(len(bundle.y_train))
    result.n_val_samples = int(len(bundle.y_val))
    result.n_features = int(bundle.n_features)
    result.n_classes = int(len(np.unique(bundle.y_train)))
    if tmp_dir is not None:
        result.model_size_mb = _model_size_mb(best_model, tmp_dir)
    return result


# ----------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------
def _predict_lgbm_booster(model, X, feature_names):
    """Label predictions with an ``lgb.Booster``.

    Uses a DataFrame with the training feature names to silence the
    "X does not have valid feature names" warning.
    """
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(
        X, columns=feature_names
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names"
        )
        raw = model.predict(X_df, num_iteration=model.best_iteration)
    if raw.ndim == 1:
        return (raw > 0.5).astype(int)
    return np.argmax(raw, axis=1)


def train_lightgbm(bundle: FeatureBundle, result: ModelResult,
                   params: dict | None = None,
                   tmp_dir: Path | None = None,
                   use_smote: bool = False) -> ModelResult:
    """5-fold CV training via ``src.lgbm.train_and_evaluate``."""
    n_classes = int(len(np.unique(bundle.y_train)))
    mode = "all_multiclass" if n_classes == 5 else "all_4class"

    if params is None:
        params = lgbm_mod.get_default_params(mode)
    params = dict(params)
    params.setdefault("verbosity", -1)
    params.setdefault("seed", RANDOM_STATE)
    if n_classes > 2:
        params.setdefault("objective", "multiclass")
        params["num_class"] = n_classes

    t0 = time.perf_counter()
    best_model, oof_preds, _oof_probs = lgbm_mod.train_and_evaluate(
        bundle.X_train,
        bundle.y_train,
        bundle.feature_names,
        params=params,
        mode=mode,
        use_smote=use_smote,
    )
    training_time = time.perf_counter() - t0

    _fill_metrics_from_predictions(result, bundle.y_train, oof_preds)

    def _predict(X):
        return _predict_lgbm_booster(best_model, X, bundle.feature_names)

    X_val_df = pd.DataFrame(bundle.X_val, columns=bundle.feature_names)
    _preds, inf_total, inf_ms = _benchmark_inference(X_val_df, predict_fn=_predict)

    result.training_time_seconds = float(training_time)
    result.inference_time_seconds_total = float(inf_total)
    result.inference_time_ms_per_sample = float(inf_ms)
    result.inference_batch_size = int(len(bundle.y_val))

    result.hyperparameters = {k: _json_safe(v) for k, v in params.items()}
    result.smote = bool(use_smote or result.smote)
    result.n_train_samples = int(len(bundle.y_train))
    result.n_val_samples = int(len(bundle.y_val))
    result.n_features = int(bundle.n_features)
    result.n_classes = n_classes
    if tmp_dir is not None:
        result.model_size_mb = _model_size_mb(best_model, tmp_dir)
    return result


# ----------------------------------------------------------------------
# SMOTE helper (kept for backward compatibility, but SMOTE is now applied
# *inside* ``src.lgbm.train_and_evaluate`` per-fold via ``use_smote=True``,
# which is the correct place to do it.)
# ----------------------------------------------------------------------
def apply_smote(X_train, y_train, random_state: int = RANDOM_STATE):
    from imblearn.over_sampling import SMOTE
    k = max(1, min(5, int(np.bincount(y_train).min()) - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    return smote.fit_resample(X_train, y_train)


# ----------------------------------------------------------------------
# Public param defaults kept for imports from workflow.py
# ----------------------------------------------------------------------
DEFAULT_RF_PARAMS: dict[str, Any] = dict(rf_mod.RF_PARAMS)
DEFAULT_LGBM_PARAMS: dict[str, Any] = dict(lgbm_mod.LGBM_PARAMS_MULTICLASS)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value