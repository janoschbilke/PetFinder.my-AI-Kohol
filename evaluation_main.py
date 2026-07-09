"""Entry point for the model evaluation workflow.

Runs all 7 model configurations, saves per-model metrics to
``results/<timestamp>/`` and generates comparison visualisations.

Examples
--------

Run everything on one machine:

    python evaluation_main.py --n-trials 50 --run-tag desktop

Split between two machines:

    # Machine A - lighter models
    python evaluation_main.py --run-tag machineA \\
        --only model_1_rf_ohe_default \\
        --only model_2_rf_ohe_tuned \\
        --only model_3_lgbm_ohe_tuned \\
        --only model_4_rf_breedpca_tuned \\
        --only model_5_lgbm_breedpca_tuned

    # Machine B - embedding models
    python evaluation_main.py --run-tag machineB \\
        --only model_6_lgbm_embeddings_tuned \\
        --only model_7_lgbm_embeddings_smote_tuned

    # After copying both `results/` folders into a shared location:
    python -m src.evaluation.merge_results --results-dir results
"""
from __future__ import annotations

# Optuna + macOS friendliness (see main.py)
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse

from src.evaluation.workflow import MODEL_SPECS, run_workflow


def _parse_args() -> argparse.Namespace:
    valid_ids = [s.model_id for s in MODEL_SPECS]

    parser = argparse.ArgumentParser(
        description="Train + benchmark all 7 model configurations."
    )
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials per tuned model (default: 50).")
    parser.add_argument("--mode", default="all_multiclass",
                        help="Preprocessing mode (default: all_multiclass).")
    parser.add_argument("--backbone", default=None,
                        choices=["alexnet", "resnet50", "efficientnet_b0"],
                        help="Single CNN backbone (legacy; prefer --backbones).")
    parser.add_argument("--backbones", default=None,
                        help="Comma-separated CNN backbones to compare for the "
                             "embedding-based models, e.g. "
                             "'alexnet,resnet50,efficientnet_b0'. Non-embedding "
                             "models run once regardless. Default: alexnet.")
    parser.add_argument("--embedding-pca", type=int, default=64,
                        help="PCA components for image embeddings (default: 64).")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation split fraction (default: 0.2).")
    parser.add_argument("--no-force-preprocess", action="store_true",
                        help="Reuse existing preprocessing cache instead of recomputing.")
    parser.add_argument("--no-force-embeddings", action="store_true",
                        help="Reuse existing image-embedding cache.")
    parser.add_argument("--only", action="append", default=None,
                        metavar="MODEL_ID",
                        help=(f"Restrict to specific model(s); can be given "
                              f"multiple times. Valid ids: {valid_ids}"))
    parser.add_argument("--run-tag", default=None,
                        help="Suffix for the run directory (e.g. hostname).")
    return parser.parse_args()


def _parse_backbones(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    valid = {"alexnet", "resnet50", "efficientnet_b0"}
    for p in parts:
        if p not in valid:
            raise SystemExit(
                f"Invalid backbone '{p}'. Choose from: {sorted(valid)}"
            )
    return parts or None


def main() -> None:
    args = _parse_args()
    backbones = _parse_backbones(args.backbones)

    run_dir = run_workflow(
        n_trials=args.n_trials,
        mode=args.mode,
        backbone=args.backbone,
        backbones=backbones,
        embedding_pca=args.embedding_pca,
        val_size=args.val_size,
        force_preprocess=not args.no_force_preprocess,
        force_embeddings=not args.no_force_embeddings,
        only=args.only,
        run_tag=args.run_tag,
    )
    print(f"\nAll done. Results at: {run_dir}")


if __name__ == "__main__":
    main()