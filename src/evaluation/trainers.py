"""Train a single model configuration, benchmark it, and return a
:class:`ModelResult`.

Everything the workflow needs to compare models fairly lives here:

    * training-time measurement (``time.perf_counter``)
    * inference-time measurement (total + ms/sample) on the same val set
    * accuracy / QWK / precision / recall / F1
    * on-disk model size

Each ``train_*`` function is fully self-contained so it can be invoked
independently by the orchestrator.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.evaluation.features import FeatureBundle
from src.evaluation.results_manager import ModelResult

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------
def _qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _benchmark_inference(
    model: Any,
    X_val: np.ndarray,
    warmup: int = 1,
) -> tuple[np.ndarray, float, float]:
    """Return (predictions, total_seconds, ms_per_sample).

    A short warm-up pass is run first so we don't measure lazy JIT /
    tree-loading cost.
    """
    for _ in range(warmup):
        _ = model.predict(X_val[: min(16, len(X_val))])

    t0 = time.perf_counter()
    preds = model.predict(X_val)
    total = time.perf_counter() - t0
    per_sample_ms = (total / len(X_val)) * 1000.0
    return preds, total, per_sample_ms


def _model_size_mb(model: Any, tmp_dir: Path) -> float:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "tmp_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size = path.stat().st_size / (1024 * 1024)
    path.unlink(missing_ok=True)
    return size


def _fill_metrics(
    result: ModelResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_time: float,
    inference_total: float,
    inference_ms: float,
    n_val: int,
) -> ModelResult:
    result.accuracy = float(accuracy_score(y_true, y_pred))
    result.qwk = float(_qwk(y_true, y_pred))
    result.f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    result.f1_weighted = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    result.precision_macro = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result.recall_macro = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result.training_time_seconds = float(training_time)
    result.inference_time_seconds_total = float(inference_total)
    result.inference_time_ms_per_sample = float(inference_ms)
    result.inference_batch_size = int(n_val)
    return result


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------
DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def train_random_forest(
    bundle: FeatureBundle,
    result: ModelResult,
    params: dict | None = None,
    tmp_dir: Path | None = None,
) -> ModelResult:
    params = {**DEFAULT_RF_PARAMS, **(params or {})}
    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("n_jobs", -1)

    model = RandomForestClassifier(**params)

    t0 = time.perf_counter()
    model.fit(bundle.X_train, bundle.y_train)
    training_time = time.perf_counter() - t0

    preds, inf_total, inf_ms = _benchmark_inference(model, bundle.X_val)

    _fill_metrics(
        result,
        bundle.y_val,
        preds,
        training_time,
        inf_total,
        inf_ms,
        n_val=len(bundle.y_val),
    )
    result.hyperparameters = {k: _json_safe(v) for k, v in params.items()}
    result.n_train_samples = int(len(bundle.y_train))
    result.n_val_samples = int(len(bundle.y_val))
    result.n_features = int(bundle.n_features)
    result.n_classes = int(len(np.unique(bundle.y_train)))
    if tmp_dir is not None:
        result.model_size_mb = _model_size_mb(model, tmp_dir)
    return result


# ----------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------
DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


def train_lightgbm(
    bundle: FeatureBundle,
    result: ModelResult,
    params: dict | None = None,
    tmp_dir: Path | None = None,
) -> ModelResult:
    n_classes = int(len(np.unique(bundle.y_train)))
    merged = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    if n_classes > 2:
        merged.setdefault("objective", "multiclass")
        merged["num_class"] = n_classes
    merged.setdefault("random_state", RANDOM_STATE)
    merged.setdefault("n_jobs", -1)
    merged.setdefault("verbosity", -1)

    model = LGBMClassifier(**merged)

    t0 = time.perf_counter()
    model.fit(bundle.X_train, bundle.y_train)
    training_time = time.perf_counter() - t0

    preds, inf_total, inf_ms = _benchmark_inference(model, bundle.X_val)

    _fill_metrics(
        result,
        bundle.y_val,
        preds,
        training_time,
        inf_total,
        inf_ms,
        n_val=len(bundle.y_val),
    )
    result.hyperparameters = {k: _json_safe(v) for k, v in merged.items()}
    result.n_train_samples = int(len(bundle.y_train))
    result.n_val_samples = int(len(bundle.y_val))
    result.n_features = int(bundle.n_features)
    result.n_classes = n_classes
    if tmp_dir is not None:
        result.model_size_mb = _model_size_mb(model, tmp_dir)
    return result


# ----------------------------------------------------------------------
# SMOTE helper
# ----------------------------------------------------------------------
def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to the training data.

    Matches the k-neighbour heuristic already used in ``src/lgbm.py``:
    ``k = min(5, min_class_count - 1)`` so it works with rare classes.
    """
    from imblearn.over_sampling import SMOTE

    k = max(1, min(5, int(np.bincount(y_train).min()) - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# ----------------------------------------------------------------------
# JSON-safety
# ----------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Coerce numpy scalars / other non-JSON types into plain Python."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value