import argparse

from src import (
    data,
    data_analysis,
    evaluate,
    image_embeddings,
    lgbm,
    model,
    preprocessing,
)

from src.lgbm import N_TUNING_TRIALS
from src.preprocessing import MODES


def main(
    force: bool = False,
    mode: str = "all_multiclass",
    tune: bool = False,
    n_trials: int = N_TUNING_TRIALS,
) -> None:
    print("Step 0: Download Data")
    data.run(force=force)

    print("\nStep 1: Data Analysis")
    data_analysis.run(force=force)

    print("\nStep 2: Preprocessing")
    preprocessing.run(force=force, mode=mode)

    # print("\nStep 3: Image Embeddings")
    # image_embeddings.run(force=force)

    # print("\nStep 4: Model Training & Prediction (tabular + CNN embeddings)")
    # model.run(force=force)

    print("\nStep 4b: LightGBM Training (tabular features only)")
    lgbm.run(force=force, tune=tune, mode=mode, n_trials=n_trials)

    # print("\nStep 5: Evaluation")
    # evaluate.run(force=force)

    print("\nPipeline complete.")

    print("\nStep 3: Image Embeddings")
    image_embeddings.run(force=force)

    print("\nStep 4: Model Training & Prediction")
    model.run(force=force)

    print("\nStep 5: Evaluation")
    evaluate.run(force=force)

    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetFinder adoption speed prediction pipeline")
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute all steps")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="all_multiclass",
        help="Data subset/task mode for preprocessing and LightGBM (default: all_multiclass)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning for LightGBM",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TUNING_TRIALS,
        help=f"Number of Optuna trials for LightGBM tuning (default: {N_TUNING_TRIALS})",
    )
    args = parser.parse_args()
    main(force=args.force, mode=args.mode, tune=args.tune, n_trials=args.n_trials)