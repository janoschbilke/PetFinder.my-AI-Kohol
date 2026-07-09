"""Results manager: save, load, and merge evaluation results.

Each model run produces a JSON file with metrics. Files across different
machines / runs can be merged into a single CSV for comparison.
"""
from __future__ import annotations

import json
import platform
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ModelResult:
    """Container for a single model evaluation result."""

    model_id: str            # e.g. "model_1_rf_ohe_default"
    model_name: str          # e.g. "Random Forest (OHE, default)"
    algorithm: str           # "RandomForest" | "LightGBM"
    feature_set: str         # "OHE_raw" | "BreedPCA" | "ImageEmbeddingsPCA64"
    tuned: bool
    smote: bool = False

    # Metrics
    accuracy: float | None = None
    qwk: float | None = None
    f1_macro: float | None = None
    f1_weighted: float | None = None
    precision_macro: float | None = None
    recall_macro: float | None = None

    # Timing (seconds unless noted)
    training_time_seconds: float | None = None
    tuning_time_seconds: float | None = None
    inference_time_seconds_total: float | None = None
    inference_time_ms_per_sample: float | None = None
    inference_batch_size: int | None = None

    # Additional
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    n_train_samples: int | None = None
    n_val_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    model_size_mb: float | None = None
    n_tuning_trials: int | None = None

    # Metadata
    hostname: str = field(default_factory=socket.gethostname)
    platform_info: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResultsManager:
    """Save and aggregate model results for a single run."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.tuning_dir = self.run_dir / "tuning_logs"
        self.viz_dir = self.run_dir / "visualizations"
        for d in (self.metrics_dir, self.tuning_dir, self.viz_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- Save / Load --------------------------------------------------------
    def save_result(self, result: ModelResult) -> Path:
        out = self.metrics_dir / f"{result.model_id}.json"
        with open(out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return out

    def load_results(self) -> list[dict[str, Any]]:
        return _load_results_from_dir(self.metrics_dir)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.load_results())

    def export_csv(self, path: Path | None = None) -> Path:
        df = self.to_dataframe()
        out = path or (self.run_dir / "merged_results.csv")
        df.to_csv(out, index=False)
        return out


def _load_results_from_dir(directory: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in sorted(directory.glob("*.json")):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def merge_results(results_root: Path, output_csv: Path | None = None) -> pd.DataFrame:
    """Merge model results from all runs under ``results_root``.

    Recursively finds ``metrics/*.json`` so results from different machines
    (each with their own timestamped run directory) are combined.
    """
    results_root = Path(results_root)
    records: list[dict[str, Any]] = []
    for metrics_dir in results_root.rglob("metrics"):
        if not metrics_dir.is_dir():
            continue
        run_id = metrics_dir.parent.name
        for rec in _load_results_from_dir(metrics_dir):
            rec.setdefault("run_id", run_id)
            records.append(rec)

    df = pd.DataFrame(records)
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return df