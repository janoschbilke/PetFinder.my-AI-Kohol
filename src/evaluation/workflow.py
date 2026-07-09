"""Main orchestrator: run all 7 model configurations end-to-end."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src import image_embeddings, preprocessing
from src.evaluation.features import FeatureBundle, build_all_feature_sets
from src.evaluation.results_manager import ModelResult, ResultsManager
from src.evaluation.trainers import (
    DEFAULT_LGBM_PARAMS,
    DEFAULT_RF_PARAMS,
    apply_smote,
    train_lightgbm,
    train_random_forest,
)
from src.evaluation.tuning import tune_lightgbm, tune_random_forest

RESULTS_ROOT = Path("results")


@dataclass
class ModelSpec:
    model_id: str
    model_name: str
    algorithm: str        # "RandomForest" | "LightGBM"
    feature_set: str      # "ohe_raw" | "breed_pca" | "embeddings_pca64"
    tuned: bool
    smote: bool = False


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("model_1_rf_ohe_default",
              "Random Forest (OHE raw, default)",
              "RandomForest", "ohe_raw", tuned=False),
    ModelSpec("model_2_rf_ohe_tuned",
              "Random Forest (OHE raw, tuned)",
              "RandomForest", "ohe_raw", tuned=True),
    ModelSpec("model_3_lgbm_ohe_tuned",
              "LightGBM (OHE raw, tuned)",
              "LightGBM", "ohe_raw", tuned=True),
    ModelSpec("model_4_rf_breedpca_tuned",
              "Random Forest (Breed PCA, tuned)",
              "RandomForest", "breed_pca", tuned=True),
    ModelSpec("model_5_lgbm_breedpca_tuned",
              "LightGBM (Breed PCA, tuned)",
              "LightGBM", "breed_pca", tuned=True),
    ModelSpec("model_6_lgbm_embeddings_tuned",
              "LightGBM (Image Embeddings PCA-64, tuned)",
              "LightGBM", "embeddings_pca64", tuned=True),
    ModelSpec("model_7_lgbm_embeddings_smote_tuned",
              "LightGBM (Image Embeddings PCA-64 + SMOTE, tuned)",
              "LightGBM", "embeddings_pca64", tuned=True, smote=True),
]


def make_run_dir(root: Path = RESULTS_ROOT, tag: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        stamp = f"{stamp}_{tag}"
    run_dir = Path(root) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_config(run_dir: Path, mode: str, backbone: str,
                     embedding_pca: int, n_trials: int, val_size: float) -> None:
    with open(run_dir / "run_config.json", "w") as f:
        json.dump({
            "mode": mode,
            "backbone": backbone,
            "embedding_pca": embedding_pca,
            "n_tuning_trials": n_trials,
            "val_size": val_size,
            "created_at": datetime.now().isoformat(),
            "models": [s.model_id for s in MODEL_SPECS],
        }, f, indent=2)


def _prepare_features(force_preprocess: bool, force_embeddings: bool,
                      mode: str, backbone: str, embedding_pca: int) -> None:
    print("\n=== [Step] CNN image embeddings ===")
    image_embeddings.run(force=force_embeddings, backbone=backbone)

    print("\n=== [Step] Preprocessing: no-embed parquet ===")
    preprocessing.run(force=force_preprocess, mode=mode,
                      backbone=backbone, embedding_pca=0)
    print("\n=== [Step] Preprocessing: with-embed parquet ===")
    preprocessing.run(force=force_preprocess, mode=mode,
                      backbone=backbone, embedding_pca=embedding_pca)


def _run_single_model(spec: ModelSpec,
                      feature_bundles: dict[str, FeatureBundle],
                      manager: ResultsManager,
                      n_trials: int,
                      tmp_dir: Path) -> ModelResult:
    print(f"\n{'='*70}\n>  {spec.model_id}  |  {spec.model_name}\n{'='*70}")
    bundle = feature_bundles[spec.feature_set]

    train_bundle = bundle
    if spec.smote:
        print("  Applying SMOTE to training data...")
        X_res, y_res = apply_smote(bundle.X_train, bundle.y_train)
        print(f"  SMOTE: {len(bundle.y_train)} -> {len(y_res)} training samples")
        train_bundle = FeatureBundle(
            name=f"{bundle.name}_smote",
            X_train=X_res, y_train=y_res,
            X_val=bundle.X_val, y_val=bundle.y_val,
            feature_names=bundle.feature_names,
        )

    result = ModelResult(
        model_id=spec.model_id,
        model_name=spec.model_name,
        algorithm=spec.algorithm,
        feature_set=spec.feature_set,
        tuned=spec.tuned,
        smote=spec.smote,
    )

    best_params: dict = {}
    tuning_time: float | None = None
    log_path = manager.tuning_dir / f"{spec.model_id}_tuning.json"

    if spec.tuned:
        print(f"  Tuning ({n_trials} trials)...")
        if spec.algorithm == "RandomForest":
            best_params, _, tuning_time = tune_random_forest(
                train_bundle.X_train, train_bundle.y_train,
                train_bundle.X_val, train_bundle.y_val,
                n_trials=n_trials, log_path=log_path,
            )
        else:
            fixed_cw: str | None = None if spec.smote else "unset"
            best_params, _, tuning_time = tune_lightgbm(
                train_bundle.X_train, train_bundle.y_train,
                train_bundle.X_val, train_bundle.y_val,
                n_trials=n_trials, log_path=log_path,
                fixed_class_weight=fixed_cw,
            )
        print(f"  Tuning done in {tuning_time:.1f}s -- best params: {best_params}")

    if spec.algorithm == "RandomForest":
        rf_params = ({**DEFAULT_RF_PARAMS, **best_params}
                     if spec.tuned else None)
        result = train_random_forest(train_bundle, result,
                                     params=rf_params, tmp_dir=tmp_dir)
    else:
        lgbm_params = ({**DEFAULT_LGBM_PARAMS, **best_params}
                       if spec.tuned else None)
        if spec.smote and lgbm_params is not None:
            lgbm_params["class_weight"] = None
        result = train_lightgbm(train_bundle, result,
                                params=lgbm_params, tmp_dir=tmp_dir)

    result.tuning_time_seconds = tuning_time
    result.n_tuning_trials = n_trials if spec.tuned else None

    out = manager.save_result(result)
    print(
        f"  OK {spec.model_id}  "
        f"acc={result.accuracy:.4f}  qwk={result.qwk:.4f}  "
        f"train={result.training_time_seconds:.2f}s  "
        f"inf={result.inference_time_ms_per_sample:.3f}ms/sample  -> {out.name}"
    )
    return result


def run_workflow(
    n_trials: int = 50,
    mode: str = "all_multiclass",
    backbone: str = "alexnet",
    embedding_pca: int = 64,
    val_size: float = 0.2,
    force_preprocess: bool = True,
    force_embeddings: bool = True,
    only: list[str] | None = None,
    run_tag: str | None = None,
) -> Path:
    """Run the full evaluation. Returns path to the timestamped run dir."""
    run_dir = make_run_dir(tag=run_tag)
    print(f"\n*  Evaluation run directory: {run_dir}")

    _save_run_config(run_dir, mode, backbone, embedding_pca, n_trials, val_size)

    _prepare_features(
        force_preprocess=force_preprocess,
        force_embeddings=force_embeddings,
        mode=mode, backbone=backbone, embedding_pca=embedding_pca,
    )

    print("\n=== [Step] Building feature bundles + train/val split ===")
    feature_bundles = build_all_feature_sets(
        run_dir=run_dir, mode=mode, backbone=backbone,
        embedding_pca=embedding_pca, val_size=val_size,
    )
    for name, b in feature_bundles.items():
        print(f"  [{name}]  X_train={b.X_train.shape}  X_val={b.X_val.shape}  "
              f"n_features={b.n_features}")

    manager = ResultsManager(run_dir)
    tmp_dir = run_dir / "tmp"

    active_specs = MODEL_SPECS if only is None else [
        s for s in MODEL_SPECS if s.model_id in set(only)
    ]
    print(f"\n=== [Step] Training {len(active_specs)} models ===")

    for spec in active_specs:
        _run_single_model(spec, feature_bundles, manager, n_trials, tmp_dir)

    print("\n=== [Step] Exporting merged CSV ===")
    csv_path = manager.export_csv()
    print(f"  Written: {csv_path}")

    try:
        from src.evaluation.visualize import generate_all_plots
        generate_all_plots(run_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  (visualisations skipped: {e})")

    print(f"\n*  Done. Results at: {run_dir}")
    return run_dir