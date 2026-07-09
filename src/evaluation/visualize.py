"""Generate comparison plots from an evaluation run.

Produces:

    * ``qwk_accuracy_bar.png``  – QWK & Accuracy grouped bars per model
    * ``qwk_vs_accuracy.png``   – scatter plot with model labels
    * ``training_times.png``    – training time (log scale) per model
    * ``inference_times.png``   – ms/sample per model
    * ``metrics_summary.csv``   – tabular summary
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.results_manager import ResultsManager, merge_results


def _sort_by_model_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("model_id").reset_index(drop=True)


_BACKBONE_SHORT = {
    "alexnet": "AN",
    "resnet50": "RN50",
    "efficientnet_b0": "EN-B0",
}


def _short_label(row: pd.Series) -> str:
    algo = "RF" if row["algorithm"] == "RandomForest" else "LGBM"
    fs = {
        "ohe_raw": "OHE",
        "breed_pca": "BrdPCA",
        "embeddings_pca64": "EMB",
    }.get(row["feature_set"], row["feature_set"])
    backbone = row.get("backbone")
    if fs == "EMB" and backbone:
        fs = f"EMB/{_BACKBONE_SHORT.get(backbone, backbone)}"
    parts = row["model_id"].split("_")
    n = parts[1] if len(parts) > 1 else ""
    suffix = ""
    if row.get("smote"):
        suffix += "+SMOTE"
    if not row.get("tuned"):
        suffix += " (def)"
    return f"{n} {algo}/{fs}{suffix}"


def plot_qwk_accuracy_bar(df: pd.DataFrame, out_path: Path) -> Path:
    df = _sort_by_model_id(df)
    labels = df.apply(_short_label, axis=1).tolist()
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, df["qwk"], width, label="QWK", color="#DD8452")

    for i, (a, q) in enumerate(zip(df["accuracy"], df["qwk"])):
        ax.text(i - width / 2, a + 0.005, f"{a:.3f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, q + 0.005, f"{q:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison -- Accuracy & QWK on validation set")
    top = max(df[["accuracy", "qwk"]].max().max() * 1.15, 0.5)
    ax.set_ylim(0, top)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_qwk_vs_accuracy(df: pd.DataFrame, out_path: Path) -> Path:
    df = _sort_by_model_id(df)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#4C72B0" if a == "RandomForest" else "#DD8452"
              for a in df["algorithm"]]
    ax.scatter(df["accuracy"], df["qwk"], c=colors, s=100, edgecolor="black")

    for _, row in df.iterrows():
        ax.annotate(_short_label(row),
                    (row["accuracy"], row["qwk"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("QWK (Quadratic Weighted Kappa)")
    ax.set_title("QWK vs Accuracy per model")
    ax.grid(alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#4C72B0", label="Random Forest"),
        Patch(color="#DD8452", label="LightGBM"),
    ])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_time_bar(df: pd.DataFrame, column: str, ylabel: str,
                   title: str, out_path: Path, log: bool = False) -> Path:
    df = _sort_by_model_id(df)
    labels = df.apply(_short_label, axis=1).tolist()
    values = df[column].astype(float).values

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color="#55A868")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.2f}" if v >= 1 else f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)

    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_training_times(df: pd.DataFrame, out_path: Path) -> Path:
    return _plot_time_bar(
        df, "training_time_seconds",
        ylabel="Training time (s, log scale)",
        title="Training time per model",
        out_path=out_path, log=True,
    )


def plot_inference_times(df: pd.DataFrame, out_path: Path) -> Path:
    return _plot_time_bar(
        df, "inference_time_ms_per_sample",
        ylabel="Inference time (ms / sample)",
        title="Inference time per model",
        out_path=out_path, log=False,
    )


def plot_tuning_times(df: pd.DataFrame, out_path: Path) -> Path | None:
    tdf = df[df["tuning_time_seconds"].notna()].copy()
    if tdf.empty:
        return None
    return _plot_time_bar(
        tdf, "tuning_time_seconds",
        ylabel="Optuna tuning time (s)",
        title="Hyperparameter tuning time per tuned model",
        out_path=out_path, log=False,
    )


def generate_all_plots(run_dir: Path) -> dict[str, Path]:
    """Load results from ``run_dir`` and produce all plots."""
    run_dir = Path(run_dir)
    manager = ResultsManager(run_dir)
    df = manager.to_dataframe()
    if df.empty:
        print(f"  No results in {manager.metrics_dir}; skipping plots.")
        return {}

    viz = manager.viz_dir
    paths: dict[str, Path] = {}
    paths["qwk_accuracy_bar"] = plot_qwk_accuracy_bar(df, viz / "qwk_accuracy_bar.png")
    paths["qwk_vs_accuracy"] = plot_qwk_vs_accuracy(df, viz / "qwk_vs_accuracy.png")
    paths["training_times"] = plot_training_times(df, viz / "training_times.png")
    paths["inference_times"] = plot_inference_times(df, viz / "inference_times.png")
    tp = plot_tuning_times(df, viz / "tuning_times.png")
    if tp is not None:
        paths["tuning_times"] = tp

    summary_cols = [
        "model_id", "model_name", "algorithm", "feature_set", "backbone",
        "tuned", "smote",
        "accuracy", "qwk", "f1_macro",
        "training_time_seconds", "tuning_time_seconds",
        "inference_time_ms_per_sample",
        "n_features", "n_train_samples", "n_val_samples",
        "hostname",
    ]
    cols = [c for c in summary_cols if c in df.columns]
    summary_path = viz / "metrics_summary.csv"
    df.sort_values("model_id")[cols].to_csv(summary_path, index=False)
    paths["summary_csv"] = summary_path

    print(f"  Plots written to: {viz}")
    for name, p in paths.items():
        print(f"    - {name}: {p.name}")
    return paths


def generate_plots_from_merged(results_root: Path,
                               output_dir: Path | None = None) -> dict[str, Path]:
    """Generate plots for merged multi-machine results."""
    results_root = Path(results_root)
    output_dir = Path(output_dir) if output_dir else results_root / "_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = merge_results(results_root)
    if df.empty:
        print(f"  No results under {results_root}; nothing to plot.")
        return {}

    csv_path = output_dir / "merged_results.csv"
    df.to_csv(csv_path, index=False)

    df_plot = (df.sort_values("timestamp")
                 .drop_duplicates(subset=["model_id", "hostname"], keep="last"))

    paths: dict[str, Path] = {"merged_csv": csv_path}
    paths["qwk_accuracy_bar"] = plot_qwk_accuracy_bar(
        df_plot, output_dir / "qwk_accuracy_bar.png")
    paths["qwk_vs_accuracy"] = plot_qwk_vs_accuracy(
        df_plot, output_dir / "qwk_vs_accuracy.png")
    paths["training_times"] = plot_training_times(
        df_plot, output_dir / "training_times.png")
    paths["inference_times"] = plot_inference_times(
        df_plot, output_dir / "inference_times.png")
    tp = plot_tuning_times(df_plot, output_dir / "tuning_times.png")
    if tp is not None:
        paths["tuning_times"] = tp

    print(f"  Merged plots written to: {output_dir}")
    return paths