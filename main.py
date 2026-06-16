import os

# For optuna and macOs
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse

from src import (
    data,
    data_analysis,
    image_embeddings,
    lgbm,
    preprocessing,
)

from src.lgbm import N_TUNING_TRIALS
from src.preprocessing import MODES


def main(
    force: bool = False,
    mode: str = "all_multiclass",
    tune: bool = False,
    n_trials: int = N_TUNING_TRIALS,
    backbone: str = "alexnet",
    embedding_pca: int = preprocessing.DEFAULT_EMBEDDING_PCA_COMPONENTS,
    use_smote: bool = False,
    imbalance_strategy: str = "balanced",
) -> None:

    print("Step 0: Download Data")
    data.run(force=force)

    print("\nStep 1: Data Analysis")
    data_analysis.run(force=force)

    if embedding_pca > 0:
        print(f"\nStep 2: Image Embeddings (backbone={backbone})")
        image_embeddings.run(force=force, backbone=backbone)
    else:
        print("\nStep 2: Skipping image embeddings (embedding_pca=0)")

    print(f"\nStep 3: Preprocessing (mode={mode}, backbone={backbone}, embedding_pca={embedding_pca})")
    preprocessing.run(force=force, mode=mode, backbone=backbone, embedding_pca=embedding_pca)

    # Build the feature_suffix so lgbm loads the right parquet
    mode_suffix = preprocessing.MODES[mode]["cache_suffix"]
    if embedding_pca > 0:
        feat_suffix = f"{mode_suffix}_{backbone}_pca{embedding_pca}"
    else:
        feat_suffix = f"{mode_suffix}_noembed"

    print("\nStep 4: LightGBM Training")
    lgbm.run(
        force=force,
        tune=tune,
        mode=mode,
        n_trials=n_trials,
        feature_suffix=feat_suffix,
        experiment_id=feat_suffix,
        use_smote=use_smote,
        imbalance_strategy=imbalance_strategy,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetFinder adoption speed prediction pipeline")
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute all steps")
    parser.add_argument(
        "--mode",
        choices=list(preprocessing.MODES.keys()),
        default="all_multiclass",
        help="Data subset/task mode (default: all_multiclass)",
    )
    parser.add_argument(
        "--backbone",
        choices=["alexnet", "resnet50", "efficientnet_b0"],
        default="alexnet",
        help="CNN backbone for image embeddings (default: alexnet)",
    )
    parser.add_argument(
        "--embedding-pca",
        type=int,
        default=preprocessing.DEFAULT_EMBEDDING_PCA_COMPONENTS,
        help="PCA components for CNN embeddings, 0=skip (default: 64)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning for LightGBM",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            f"Number of Optuna trials. "
            f"Default: 15 when --experiments, {N_TUNING_TRIALS} otherwise."
        ),
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE oversampling inside each CV fold (requires --tune)",
    )
    parser.add_argument(
        "--imbalance-strategy",
        choices=["balanced", "custom_weights"],
        default="balanced",
        help=(
            "Class imbalance strategy for Optuna tuning. "
            "'balanced': tune class_weight (balanced vs None). "
            "'custom_weights': Optuna tunes class_0_weight in [1.0, 20.0]. "
            "(default: balanced)"
        ),
    )
    args = parser.parse_args()

    # Resolve n_trials default based on mode
    if args.n_trials is not None:
        n_trials = args.n_trials
    else:
        n_trials = N_TUNING_TRIALS  # 50

    main(
        force=args.force,
        mode=args.mode,
        tune=args.tune,
        n_trials=n_trials,
        backbone=args.backbone,
        embedding_pca=args.embedding_pca,
        use_smote=args.use_smote,
        imbalance_strategy=args.imbalance_strategy,
    )
