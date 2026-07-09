"""Main orchestrator: run all model configurations end-to-end.

When multiple ``backbones`` are supplied, the embedding-based models
(models 6 & 7) are duplicated per backbone so their performance and
timings can be compared directly. Non-embedding models run once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src import image_embeddings, preprocessing
from src.evaluation.features import FeatureBundle, build_all_feature_sets
from src.evaluation.results_manager import ModelResult, ResultsManager
from src.evaluation.trainers import (
    DEFAULT_LGBM_PARAMS, DEFAULT_RF_PARAMS,
    train_lightgbm, train_random_forest,
)
from src.evaluation.tuning import tune_lightgbm, tune_random_forest

RESULTS_ROOT = Path("results")


@dataclass
class ModelSpec:
    model_id: str
    model_name: str
    algorithm: str
    feature_set: str
    tuned: bool
    smote: bool = False


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("model_1_rf_ohe_default", "Random Forest (OHE raw, default)",
              "RandomForest", "ohe_raw", tuned=False),
    ModelSpec("model_2_rf_ohe_tuned", "Random Forest (OHE raw, tuned)",
              "RandomForest", "ohe_raw", tuned=True),
    ModelSpec("model_3_lgbm_ohe_tuned", "LightGBM (OHE raw, tuned)",
              "LightGBM", "ohe_raw", tuned=True),
    ModelSpec("model_4_rf_breedpca_tuned", "Random Forest (Breed PCA, tuned)",
              "RandomForest", "breed_pca", tuned=True),
    ModelSpec("model_5_lgbm_breedpca_tuned", "LightGBM (Breed PCA, tuned)",
              "LightGBM", "breed_pca", tuned=True),
    ModelSpec("model_6_lgbm_embeddings_tuned",
              "LightGBM (Image Embeddings PCA-64, tuned)",
              "LightGBM", "embeddings_pca64", tuned=True),
    ModelSpec("model_7_lgbm_embeddings_smote_tuned",
              "LightGBM (Image Embeddings PCA-64 + SMOTE, tuned)",
              "LightGBM", "embeddings_pca64", tuned=True, smote=True),
]

EMBEDDING_FEATURE_SETS: set[str] = {"embeddings_pca64"}


def make_run_dir(root: Path = RESULTS_ROOT, tag: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        stamp = f"{stamp}_{tag}"
    run_dir = Path(root) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_config(run_dir, mode, backbones, embedding_pca, n_trials, val_size):
    with open(run_dir / "run_config.json", "w") as f:
        json.dump({
            "mode": mode, "backbones": backbones,
            "embedding_pca": embedding_pca, "n_tuning_trials": n_trials,
            "val_size": val_size,
            "created_at": datetime.now().isoformat(),
            "models": [s.model_id for s in MODEL_SPECS],
        }, f, indent=2)


def _prepare_features(force_preprocess, force_embeddings, mode, backbone,
                      embedding_pca, needs_embeddings):
    if needs_embeddings:
        print(f"\n=== [Step] CNN image embeddings ({backbone}) ===")
        image_embeddings.run(force=force_embeddings, backbone=backbone)
    print("\n=== [Step] Preprocessing: no-embed parquet ===")
    preprocessing.run(force=force_preprocess, mode=mode,
                      backbone=backbone, embedding_pca=0)
    if needs_embeddings:
        print(f"\n=== [Step] Preprocessing: with-embed ({backbone}) ===")
        preprocessing.run(force=force_preprocess, mode=mode,
                          backbone=backbone, embedding_pca=embedding_pca)


def _run_single_model(spec, feature_bundles, manager, n_trials, tmp_dir,
                      backbone=None,
                      model_id_override=None, model_name_override=None):
    model_id = model_id_override or spec.model_id
    model_name = model_name_override or spec.model_name
    print(f"\n{'='*70}\n>  {model_id}  |  {model_name}\n{'='*70}")
    bundle = feature_bundles[spec.feature_set]

    result = ModelResult(
        model_id=model_id, model_name=model_name,
        algorithm=spec.algorithm, feature_set=spec.feature_set,
        tuned=spec.tuned, smote=spec.smote, backbone=backbone,
    )

    best_params: dict = {}
    tuning_time: float | None = None
    log_path = manager.tuning_dir / f"{model_id}_tuning.json"

    if spec.tuned:
        print(f"  Tuning ({n_trials} trials)...")
        if spec.algorithm == "RandomForest":
            best_params, _, tuning_time = tune_random_forest(
                bundle.X_train, bundle.y_train,
                n_trials=n_trials, log_path=log_path,
            )
        else:
            best_params, _, tuning_time = tune_lightgbm(
                bundle.X_train, bundle.y_train, bundle.feature_names,
                n_trials=n_trials, use_smote=spec.smote, log_path=log_path,
            )
        print(f"  Tuning done in {tuning_time:.1f}s")

    if spec.algorithm == "RandomForest":
        rf_params = ({**DEFAULT_RF_PARAMS, **best_params}
                     if spec.tuned else None)
        result = train_random_forest(bundle, result, params=rf_params,
                                     tmp_dir=tmp_dir)
    else:
        lgbm_params: dict | None = (
            {**DEFAULT_LGBM_PARAMS, **best_params} if spec.tuned else None
        )
        result = train_lightgbm(bundle, result, params=lgbm_params,
                                tmp_dir=tmp_dir, use_smote=spec.smote)

    result.model_id = model_id
    result.model_name = model_name
    result.backbone = backbone
    result.tuning_time_seconds = tuning_time
    result.n_tuning_trials = n_trials if spec.tuned else None

    out = manager.save_result(result)
    print(
        f"  OK {model_id}  acc={result.accuracy:.4f}  qwk={result.qwk:.4f}  "
        f"train={result.training_time_seconds:.2f}s  "
        f"inf={result.inference_time_ms_per_sample:.3f}ms/sample -> {out.name}"
    )
    return result


def _select_specs(only):
    if only is None:
        return list(MODEL_SPECS)
    only_set = set(only)
    matched: list[ModelSpec] = []
    for spec in MODEL_SPECS:
        if spec.model_id in only_set:
            matched.append(spec)
            continue
        for tok in only_set:
            if tok.startswith(spec.model_id + "_"):
                matched.append(spec)
                break
    return matched


def run_workflow(
    n_trials: int = 50,
    mode: str = "all_multiclass",
    backbone: str | None = None,
    backbones: list[str] | None = None,
    embedding_pca: int = 64,
    val_size: float = 0.2,
    force_preprocess: bool = True,
    force_embeddings: bool = True,
    only: list[str] | None = None,
    run_tag: str | None = None,
) -> Path:
    """Run the full evaluation. Returns the run directory."""
    if backbones is None:
        backbones = [backbone] if backbone else ["alexnet"]

    run_dir = make_run_dir(tag=run_tag)
    print(f"\n*  Evaluation run directory: {run_dir}")
    _save_run_config(run_dir, mode, backbones, embedding_pca, n_trials, val_size)

    manager = ResultsManager(run_dir)
    tmp_dir = run_dir / "tmp"

    active_specs = _select_specs(only)
    non_emb = [s for s in active_specs
               if s.feature_set not in EMBEDDING_FEATURE_SETS]
    emb = [s for s in active_specs
           if s.feature_set in EMBEDDING_FEATURE_SETS]

    print("\n=== Plan ===")
    print(f"  Backbones: {backbones}")
    print(f"  Non-embedding models: {[s.model_id for s in non_emb]}")
    print(f"  Embedding models (per backbone): {[s.model_id for s in emb]}")

    # 1) Non-embedding models: run once, using the first backbone as anchor
    if non_emb:
        anchor = backbones[0]
        _prepare_features(
            force_preprocess=force_preprocess, force_embeddings=False,
            mode=mode, backbone=anchor, embedding_pca=embedding_pca,
            needs_embeddings=False,
        )
        print("\n=== [Step] Building non-embedding feature bundles ===")
        bundles = build_all_feature_sets(
            run_dir=run_dir, mode=mode, backbone=anchor,
            embedding_pca=embedding_pca, val_size=val_size,
        )
        for name, b in bundles.items():
            if name in EMBEDDING_FEATURE_SETS:
                continue
            print(f"  [{name}]  X_train={b.X_train.shape}  "
                  f"X_val={b.X_val.shape}  n_features={b.n_features}")
        print(f"\n=== [Step] Training {len(non_emb)} non-embedding models ===")
        for spec in non_emb:
            _run_single_model(spec, bundles, manager, n_trials, tmp_dir)

    # 2) Embedding models: run once per backbone
    for backbone_name in backbones:
        if not emb:
            break
        _prepare_features(
            force_preprocess=force_preprocess,
            force_embeddings=force_embeddings,
            mode=mode, backbone=backbone_name,
            embedding_pca=embedding_pca, needs_embeddings=True,
        )
        print(f"\n=== [Step] Building embedding feature bundle "
              f"({backbone_name}) ===")
        bundles = build_all_feature_sets(
            run_dir=run_dir, mode=mode, backbone=backbone_name,
            embedding_pca=embedding_pca, val_size=val_size,
        )
        b = bundles["embeddings_pca64"]
        print(f"  [embeddings_pca64/{backbone_name}]  "
              f"X_train={b.X_train.shape}  X_val={b.X_val.shape}  "
              f"n_features={b.n_features}")

        print(f"\n=== [Step] Training {len(emb)} embedding models "
              f"({backbone_name}) ===")
        # Only use the per-backbone suffix when comparing multiple backbones,
        # so single-backbone runs keep the original model_id.
        use_suffix = len(backbones) > 1
        for spec in emb:
            if use_suffix:
                mid = f"{spec.model_id}_{backbone_name}"
                mname = f"{spec.model_name} [{backbone_name}]"
            else:
                mid, mname = spec.model_id, spec.model_name
            _run_single_model(
                spec, bundles, manager, n_trials, tmp_dir,
                backbone=backbone_name,
                model_id_override=mid, model_name_override=mname,
            )

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
