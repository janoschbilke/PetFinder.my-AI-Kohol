"""Experimental pipeline orchestrator for the 4-stage paper study.

Stages:
  1. Baseline (RF, LGBM) on tabular features only (no image embeddings)
  2. Baseline + Breed PCA (same cache — the tabular parquet already contains PCA breed cols)
  3. Add image embeddings (AlexNet / ResNet50 / EfficientNet-B0), PCA-reduced
  4. Best-embedding config + SMOTE oversampling

For every experiment we record:
  - CV fold scores (QWK), mean & std
  - OOF accuracy, F1 macro / weighted
  - Confusion matrix
  - Model config (params, feature count)

All artefacts are written under `documentation/experimental_results/`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

# Threading knobs for macOS + Optuna interactions (mirror main.py)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# Silence sklearn/optuna noise during the sweep
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src import image_embeddings, lgbm, preprocessing, random_forest

RESULTS_DIR = Path("documentation/experimental_results")
CACHE_DIR = Path("cache")
MODE = "all_multiclass"           # paper focus: 5-class task
EMBEDDING_PCA = 64
BACKBONES = ["alexnet", "resnet50", "efficientnet_b0"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suffix_for(backbone: str | None) -> str:
    """Return the parquet-file suffix produced by preprocessing.run()."""
    mode_suffix = preprocessing.MODES[MODE]["cache_suffix"]  # "" for all_multiclass
    if backbone is None:
        return f"{mode_suffix}_noembed"
    return f"{mode_suffix}_{backbone}_pca{EMBEDDING_PCA}"


def _ensure_preprocessed(backbone: str | None, force: bool = False) -> str:
    """Run preprocessing (if needed) for a given backbone. Returns the suffix."""
    if backbone is None:
        print(f"\n[preprocessing] mode={MODE}, no embeddings")
        preprocessing.run(force=force, mode=MODE, backbone="alexnet", embedding_pca=0)
    else:
        # Ensure embeddings are available for this backbone
        _ensure_embeddings(backbone, force=False)
        print(f"\n[preprocessing] mode={MODE}, backbone={backbone}, pca={EMBEDDING_PCA}")
        preprocessing.run(force=force, mode=MODE, backbone=backbone, embedding_pca=EMBEDDING_PCA)
    return _suffix_for(backbone)


def _ensure_embeddings(backbone: str, force: bool = False) -> None:
    train_emb_path = CACHE_DIR / f"train_embeddings_{backbone}.npy"
    if not force and train_emb_path.exists():
        print(f"[embeddings] cache found for {backbone}")
        return
    print(f"[embeddings] computing for {backbone} (may take a while)...")
    image_embeddings.run(force=force, backbone=backbone)


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _rerun_id(base: str) -> str:
    """Prefix experiment IDs so caches don't collide with older runs."""
    return f"_exp_{base}"


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def _run_rf(feature_suffix: str, experiment_id: str, use_smote: bool = False,
            tune: bool = False, force: bool = False) -> dict:
    print(f"\n>>> RF  id={experiment_id}  suffix='{feature_suffix}'  smote={use_smote}  tune={tune}")
    t0 = time.time()
    metrics = random_forest.run(
        force=force, tune=tune, mode=MODE,
        feature_suffix=feature_suffix, experiment_id=experiment_id,
        use_smote=use_smote,
    )
    metrics["elapsed_sec"] = round(time.time() - t0, 2)
    metrics["experiment_id"] = experiment_id
    return metrics


def _run_lgbm(feature_suffix: str, experiment_id: str, use_smote: bool = False,
              tune: bool = False, n_trials: int = 30, force: bool = False) -> dict:
    print(f"\n>>> LGBM id={experiment_id}  suffix='{feature_suffix}'  smote={use_smote}  tune={tune}")
    t0 = time.time()
    metrics = lgbm.run(
        force=force, tune=tune, mode=MODE, n_trials=n_trials,
        feature_suffix=feature_suffix, experiment_id=experiment_id,
        use_smote=use_smote,
    )
    metrics["elapsed_sec"] = round(time.time() - t0, 2)
    metrics["experiment_id"] = experiment_id
    return metrics


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_1_baseline(tune: bool = False, force: bool = False) -> dict:
    """Baseline: tabular features only, no CNN embeddings (breed PCA still present)."""
    print("\n" + "="*70)
    print(" STAGE 1 — Baseline (tabular only, no image embeddings)")
    print("="*70)
    stage_dir = RESULTS_DIR / "stage_1_baseline"
    stage_dir.mkdir(parents=True, exist_ok=True)

    suffix = _ensure_preprocessed(backbone=None, force=force)

    rf_id = _rerun_id("s1_rf_baseline")
    lgbm_id = _rerun_id("s1_lgbm_baseline")

    rf_m = _run_rf(suffix, rf_id, force=force)
    lgbm_m = _run_lgbm(suffix, lgbm_id, tune=tune, force=force)

    _save_json(rf_m, stage_dir / "rf_baseline.json")
    _save_json(lgbm_m, stage_dir / "lgbm_baseline.json")
    _write_stage_comparison(stage_dir / "comparison.csv", [rf_m, lgbm_m], "Stage 1 Baseline")
    return {"rf": rf_m, "lgbm": lgbm_m, "suffix": suffix}


def stage_2_pca(tune: bool = False, force: bool = False) -> dict:
    """Stage 2 = Stage 1 with an explicit note that Breed PCA is included.

    Since our preprocessing pipeline *always* applies breed-column PCA, Stage 1
    and Stage 2 use the same feature set.  We still run the models under a
    distinct experiment_id so we get a clean comparison row and so the paper's
    ablation table shows the (redundant) PCA-only condition explicitly.
    """
    print("\n" + "="*70)
    print(" STAGE 2 — Baseline + Breed PCA (feature set == Stage 1 in this pipeline)")
    print("="*70)
    stage_dir = RESULTS_DIR / "stage_2_pca"
    stage_dir.mkdir(parents=True, exist_ok=True)

    suffix = _ensure_preprocessed(backbone=None, force=force)

    rf_id = _rerun_id("s2_rf_pca")
    lgbm_id = _rerun_id("s2_lgbm_pca")

    rf_m = _run_rf(suffix, rf_id, force=force)
    lgbm_m = _run_lgbm(suffix, lgbm_id, tune=tune, force=force)

    _save_json(rf_m, stage_dir / "rf_pca.json")
    _save_json(lgbm_m, stage_dir / "lgbm_pca.json")
    _write_stage_comparison(stage_dir / "comparison.csv", [rf_m, lgbm_m], "Stage 2 PCA")
    return {"rf": rf_m, "lgbm": lgbm_m, "suffix": suffix}


def stage_3_embeddings(tune: bool = False, force: bool = False, backbones=None) -> dict:
    """Stage 3: add CNN image embeddings for each backbone."""
    backbones = backbones or BACKBONES
    print("\n" + "="*70)
    print(f" STAGE 3 — Image Embeddings ({len(backbones)} backbones)")
    print("="*70)
    stage_dir = RESULTS_DIR / "stage_3_embeddings"
    stage_dir.mkdir(parents=True, exist_ok=True)

    per_backbone: dict[str, dict] = {}
    all_metrics: list[dict] = []

    for backbone in backbones:
        suffix = _ensure_preprocessed(backbone=backbone, force=force)
        rf_id = _rerun_id(f"s3_rf_{backbone}")
        lgbm_id = _rerun_id(f"s3_lgbm_{backbone}")

        rf_m = _run_rf(suffix, rf_id, force=force)
        lgbm_m = _run_lgbm(suffix, lgbm_id, tune=tune, force=force)

        rf_m["backbone"] = backbone
        lgbm_m["backbone"] = backbone
        per_backbone[backbone] = {"rf": rf_m, "lgbm": lgbm_m, "suffix": suffix}

        _save_json(rf_m, stage_dir / f"rf_{backbone}.json")
        _save_json(lgbm_m, stage_dir / f"lgbm_{backbone}.json")
        all_metrics.extend([rf_m, lgbm_m])

    _write_stage_comparison(stage_dir / "embedding_backbone_comparison.csv",
                            all_metrics, "Stage 3 Image Embeddings")

    # Determine best backbone (highest LGBM QWK — LGBM is the stronger baseline)
    best_backbone = max(backbones, key=lambda b: per_backbone[b]["lgbm"].get("qwk", 0.0))
    print(f"\n[stage 3] Best backbone by LGBM QWK: {best_backbone}")

    with open(stage_dir / "best_backbone.json", "w") as f:
        json.dump({"best_backbone": best_backbone,
                   "lgbm_qwk_by_backbone": {b: per_backbone[b]["lgbm"].get("qwk") for b in backbones}}, f, indent=2)

    return {"per_backbone": per_backbone, "best_backbone": best_backbone}


def stage_4_smote(best_backbone: str, tune: bool = False, force: bool = False) -> dict:
    """Stage 4: best embedding config + SMOTE oversampling."""
    print("\n" + "="*70)
    print(f" STAGE 4 — Best Backbone ({best_backbone}) + SMOTE")
    print("="*70)
    stage_dir = RESULTS_DIR / "stage_4_smote"
    stage_dir.mkdir(parents=True, exist_ok=True)

    suffix = _ensure_preprocessed(backbone=best_backbone, force=force)
    rf_id = _rerun_id(f"s4_rf_{best_backbone}_smote")
    lgbm_id = _rerun_id(f"s4_lgbm_{best_backbone}_smote")

    rf_m = _run_rf(suffix, rf_id, use_smote=True, force=force)
    lgbm_m = _run_lgbm(suffix, lgbm_id, use_smote=True, tune=tune, force=force)

    rf_m["backbone"] = best_backbone
    lgbm_m["backbone"] = best_backbone

    _save_json(rf_m, stage_dir / "rf_smote.json")
    _save_json(lgbm_m, stage_dir / "lgbm_smote.json")
    _write_stage_comparison(stage_dir / "comparison.csv", [rf_m, lgbm_m],
                            f"Stage 4 SMOTE ({best_backbone})")
    return {"rf": rf_m, "lgbm": lgbm_m, "suffix": suffix, "backbone": best_backbone}


# ---------------------------------------------------------------------------
# Comparison writers
# ---------------------------------------------------------------------------

def _row_from_metrics(m: dict, stage: str, feature_set: str) -> dict:
    return {
        "stage": stage,
        "model": m.get("model", "lightgbm" if "lgbm" in m.get("experiment_id", "") else "random_forest"),
        "experiment_id": m.get("experiment_id", ""),
        "feature_set": feature_set,
        "backbone": m.get("backbone", "-"),
        "use_smote": m.get("use_smote", False),
        "qwk": m.get("qwk"),
        "auc": m.get("auc"),
        "accuracy": m.get("accuracy"),
        "f1_macro": m.get("f1_macro"),
        "f1_weighted": m.get("f1_weighted"),
        "cv_mean": m.get("cv_mean"),
        "cv_std": m.get("cv_std"),
        "n_features": m.get("n_features"),
        "n_samples": m.get("n_samples"),
        "elapsed_sec": m.get("elapsed_sec"),
    }


def _write_stage_comparison(path: Path, metrics_list: list[dict], title: str) -> None:
    rows = []
    for m in metrics_list:
        rows.append(_row_from_metrics(m, stage=title, feature_set=m.get("experiment_id", "")))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Wrote comparison CSV: {path}")


def build_ablation_table(all_results: dict) -> pd.DataFrame:
    rows: list[dict] = []

    def add(stage: str, feature_set: str, res_dict: dict):
        for model_key, m in res_dict.items():
            if model_key not in ("rf", "lgbm") or not isinstance(m, dict):
                continue
            rows.append(_row_from_metrics(m, stage=stage, feature_set=feature_set))

    add("Stage 1 — Baseline (no embeddings)",
        "tabular_only", all_results.get("stage_1", {}))
    add("Stage 2 — Baseline + Breed PCA",
        "tabular_pca", all_results.get("stage_2", {}))

    stage_3 = all_results.get("stage_3", {})
    for backbone, per in stage_3.get("per_backbone", {}).items():
        add(f"Stage 3 — Embeddings ({backbone})",
            f"tabular_pca + {backbone}_pca{EMBEDDING_PCA}", per)

    stage_4 = all_results.get("stage_4", {})
    if stage_4:
        add(f"Stage 4 — Best backbone + SMOTE",
            f"tabular_pca + {stage_4.get('backbone')}_pca{EMBEDDING_PCA} + SMOTE", stage_4)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def make_visualisations(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150})

    # ---- QWK bar chart ----
    plot_df = df.dropna(subset=["qwk"]).copy()
    if not plot_df.empty:
        plot_df["label"] = plot_df["stage"] + " — " + plot_df["model"]
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ["#4C72B0" if m == "random_forest" else "#DD8452" for m in plot_df["model"]]
        ax.bar(range(len(plot_df)), plot_df["qwk"], color=colors)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("QWK (Quadratic Weighted Kappa)")
        ax.set_title("QWK across all experiments (RF blue, LGBM orange)")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(plot_df["qwk"]):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "qwk_comparison.png")
        plt.close(fig)
        print(f"  Wrote {out_dir / 'qwk_comparison.png'}")

    # ---- F1 bar chart ----
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(plot_df))
        w = 0.4
        ax.bar(x - w/2, plot_df["f1_macro"],   width=w, label="F1 macro",    color="#4C72B0")
        ax.bar(x + w/2, plot_df["f1_weighted"], width=w, label="F1 weighted", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1 score")
        ax.set_title("F1 (macro vs weighted) across experiments")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "f1_comparison.png")
        plt.close(fig)
        print(f"  Wrote {out_dir / 'f1_comparison.png'}")

    # ---- Progressive improvement plot ----
    if not plot_df.empty:
        rf = plot_df[plot_df["model"] == "random_forest"].reset_index(drop=True)
        lg = plot_df[plot_df["model"] == "lightgbm"].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        if not rf.empty:
            ax.plot(rf["stage"], rf["qwk"], marker="o", label="Random Forest", color="#4C72B0")
        if not lg.empty:
            ax.plot(lg["stage"], lg["qwk"], marker="s", label="LightGBM", color="#DD8452")
        ax.set_ylabel("QWK")
        ax.set_title("Progressive QWK improvement across stages")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "progression.png")
        plt.close(fig)
        print(f"  Wrote {out_dir / 'progression.png'}")


# ---------------------------------------------------------------------------
# README generator
# ---------------------------------------------------------------------------

def write_readme(df: pd.DataFrame, out_path: Path, best_backbone: str | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Executive summary values
    if "qwk" in df.columns and df["qwk"].notna().any():
        best_row = df.loc[df["qwk"].idxmax()]
        best_str = (
            f"**Best overall model:** {best_row['model']} in *{best_row['stage']}* — "
            f"QWK **{best_row['qwk']:.4f}** (accuracy {best_row['accuracy']:.4f}, "
            f"F1-macro {best_row['f1_macro']:.4f})."
        )
    else:
        best_str = "_No QWK results available yet._"

    lines: list[str] = []
    lines.append("# Experimental Results — 4-Stage Ablation Study\n")
    lines.append("This directory contains the full results of the 4-stage experimental "
                 "pipeline used for the paper.\n")
    lines.append("## Setup\n")
    lines.append("- **Task:** 5-class adoption speed prediction (`all_multiclass`)\n")
    lines.append("- **Primary metric:** Quadratic Weighted Kappa (QWK)\n")
    lines.append("- **Cross-validation:** 5-fold Stratified (identical splits across all "
                 "experiments, seed=42)\n")
    lines.append("- **Models:** Random Forest, LightGBM\n")
    lines.append("- **Image backbones:** AlexNet, ResNet50, EfficientNet-B0\n")
    lines.append("- **Embedding compression:** PCA to 64 components (fitted on train only)\n")
    lines.append("- **Imbalance handling:** class_weight='balanced' (all stages) + SMOTE "
                 "in stage 4 (training folds only, k_neighbors=5)\n")
    lines.append("\n## Executive summary\n")
    lines.append(best_str + "\n")
    if best_backbone:
        lines.append(f"\n**Best image backbone (by LGBM QWK):** `{best_backbone}`.\n")

    lines.append("\n## Ablation table\n")
    if not df.empty:
        display_cols = ["stage", "model", "backbone", "use_smote", "qwk",
                        "accuracy", "f1_macro", "f1_weighted", "cv_mean", "cv_std",
                        "n_features"]
        display_cols = [c for c in display_cols if c in df.columns]
        md_table = df[display_cols].to_markdown(index=False, floatfmt=".4f")
        lines.append(md_table + "\n")
    else:
        lines.append("_(No results yet.)_\n")

    lines.append("\n## Per-stage findings\n")
    lines.append("### Stage 1 — Baseline (tabular only)\n")
    lines.append(
        "Baseline models trained on hand-engineered tabular features (numeric, one-hot "
        "categorical, multi-hot colour, sentiment and metadata aggregates) plus a 15-component "
        "PCA of the sparse breed multi-hot columns.  Serves as the reference for all later stages.\n"
    )
    lines.append("### Stage 2 — Baseline + Breed PCA\n")
    lines.append(
        "Because our preprocessing pipeline always applies breed PCA, this stage is included "
        "for completeness — its feature set matches Stage 1 and the numbers should agree "
        "up to noise.  Its main purpose in the paper is to make the ablation explicit.\n"
    )
    lines.append("### Stage 3 — CNN Image Embeddings\n")
    lines.append(
        "Pre-trained (ImageNet) backbones are used as fixed feature extractors; the mean of "
        "each pet's image embeddings is PCA-reduced to 64 dimensions and appended to the "
        "tabular features.  We evaluate three backbones (AlexNet, ResNet50, EfficientNet-B0) "
        "to test whether backbone choice matters for this dataset.\n"
    )
    lines.append("### Stage 4 — Best backbone + SMOTE\n")
    lines.append(
        "The best-performing backbone from Stage 3 is combined with SMOTE oversampling of the "
        "training fold only (k_neighbors=5).  This isolates the effect of imbalance handling "
        "on the strongest feature set.\n"
    )

    lines.append("\n## Statistical detail\n")
    if not df.empty and "cv_std" in df.columns and df["cv_std"].notna().any():
        lines.append(
            "The CV standard deviations reported above are the per-fold spread in the primary "
            "metric (QWK for multiclass).  Fold-level scores are saved in each `*.json` "
            "artefact under the `fold_scores` key.\n"
        )

    lines.append("\n## Files\n")
    lines.append("- `stage_1_baseline/` – Stage 1 JSON metrics and comparison CSV\n")
    lines.append("- `stage_2_pca/` – Stage 2 JSON metrics and comparison CSV\n")
    lines.append("- `stage_3_embeddings/` – per-backbone metrics + best-backbone summary\n")
    lines.append("- `stage_4_smote/` – SMOTE experiment metrics\n")
    lines.append("- `comparisons/` – aggregated ablation CSV, all-models comparison, and plots\n")

    out_path.write_text("\n".join(lines))
    print(f"  Wrote README: {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_all(tune: bool = False, force: bool = False,
            backbones: list[str] | None = None,
            skip_stages: list[int] | None = None) -> None:
    skip = set(skip_stages or [])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    if 1 not in skip:
        all_results["stage_1"] = stage_1_baseline(tune=tune, force=force)
    if 2 not in skip:
        all_results["stage_2"] = stage_2_pca(tune=tune, force=force)
    if 3 not in skip:
        all_results["stage_3"] = stage_3_embeddings(tune=tune, force=force, backbones=backbones)

    best_backbone = None
    if "stage_3" in all_results:
        best_backbone = all_results["stage_3"].get("best_backbone")
    elif backbones:
        best_backbone = backbones[0]
    else:
        best_backbone = BACKBONES[0]

    if 4 not in skip:
        all_results["stage_4"] = stage_4_smote(best_backbone, tune=tune, force=force)

    # Aggregate + write comparison files & README
    print("\n" + "="*70)
    print(" Aggregating results …")
    print("="*70)
    df = build_ablation_table(all_results)
    comp_dir = RESULTS_DIR / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(comp_dir / "ablation_study.csv", index=False)
    df.to_csv(comp_dir / "all_models_comparison.csv", index=False)
    print(f"  Wrote {comp_dir / 'ablation_study.csv'}")
    make_visualisations(df, comp_dir)
    write_readme(df, RESULTS_DIR / "README.md", best_backbone=best_backbone)

    # Also dump the raw results dict for reproducibility
    _save_json({k: {kk: (vv if not isinstance(vv, dict) else
                          {mk: mv for mk, mv in vv.items() if not isinstance(mv, np.ndarray)})
                    for kk, vv in (v.items() if isinstance(v, dict) else [])}
                for k, v in all_results.items()},
               RESULTS_DIR / "detailed_metrics" / "all_results.json")

    print("\n[experimental_pipeline] DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 4-stage experimental pipeline.")
    parser.add_argument("--tune", action="store_true",
                        help="Enable LGBM Optuna tuning (and RF grid search) per stage")
    parser.add_argument("--force", action="store_true",
                        help="Ignore caches and re-run everything")
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Which CNN backbones to test in stage 3 (default: all 3)")
    parser.add_argument("--skip", nargs="*", type=int, default=None,
                        help="Skip these stage numbers (e.g. --skip 3 to skip stage 3)")
    args = parser.parse_args()

    run_all(
        tune=args.tune,
        force=args.force,
        backbones=args.backbones,
        skip_stages=args.skip,
    )
