"""Feature-set builders and shared train/val split.

Produces three feature matrices from the same underlying training data
so that all 7 model configurations are compared on identical rows:

    * ``ohe_raw``           – full one-hot (breed multi-hot, no PCA, no images)
    * ``breed_pca``         – one-hot + breed-PCA (no images)
    * ``embeddings_pca64``  – one-hot + breed-PCA + image-embeddings PCA-64

We rely on ``src.preprocessing`` (which produces breed-PCA + optional image
embeddings) and, for ``ohe_raw``, rebuild the breed multi-hot columns
directly to avoid the PCA step.

All feature matrices share the same row ordering, so a single train/val
split can be applied to all of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import get_data_root
from src.preprocessing import (
    CACHE_DIR,
    MODES,
    build_features,
)

RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.2

FEATURE_SETS = ("ohe_raw", "breed_pca", "embeddings_pca64")


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------
@dataclass
class FeatureBundle:
    """A feature matrix + labels, aligned by index."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


# ----------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------
def _split_path(run_dir: Path) -> Path:
    return Path(run_dir) / "split_indices.npz"


def build_or_load_split(
    n_samples: int,
    y: np.ndarray,
    run_dir: Path,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx). Persist to ``run_dir`` so all models see
    the same split, and other machines can reuse it if desired."""
    path = _split_path(run_dir)
    if path.exists():
        data = np.load(path)
        return data["train_idx"], data["val_idx"]

    idx = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


# ----------------------------------------------------------------------
# Feature-set 1: raw OHE (no breed-PCA, no embeddings)
# ----------------------------------------------------------------------
def build_ohe_raw_features(mode: str = "all_multiclass") -> pd.DataFrame:
    """Rebuild the training feature matrix using breed multi-hot (no PCA)
    and no image embeddings.

    Returns a DataFrame indexed 0..N-1 with a trailing ``AdoptionSpeed``
    column so it can be split identically to the other feature sets.
    """
    mode_config = MODES[mode]
    type_filter = mode_config["type_filter"]

    data_root = get_data_root()
    train_csv = data_root / "train" / "train.csv"
    sentiment_dir_train = data_root / "train_sentiment"
    metadata_dir_train = data_root / "train_metadata"

    train_df = pd.read_csv(train_csv)

    if type_filter is not None:
        train_df = train_df[train_df["Type"] == type_filter].reset_index(drop=True)

    classes = mode_config["classes"]
    if classes != [0, 1, 2, 3, 4]:
        train_df = train_df[train_df["AdoptionSpeed"].isin(classes)].reset_index(drop=True)
        class_map = {orig: new for new, orig in enumerate(sorted(classes))}
        train_df["AdoptionSpeed"] = train_df["AdoptionSpeed"].map(class_map)

    y_train = train_df["AdoptionSpeed"].astype(int).copy()

    features, breed_mh = build_features(
        train_df,
        sentiment_dir_train,
        metadata_dir_train,
        data_root,
        exclude_type_col=(type_filter is not None),
        type_filter=type_filter,
    )
    # Concat raw breed multi-hot (skip breed-PCA path entirely)
    features = pd.concat([features, breed_mh], axis=1)
    features["AdoptionSpeed"] = y_train.values
    return features


# ----------------------------------------------------------------------
# Feature-sets 2 & 3: reuse preprocessing.run() cache
# ----------------------------------------------------------------------
def _load_preprocessing_features(
    mode: str, backbone: str, embedding_pca: int
) -> pd.DataFrame:
    """Load the parquet produced by ``preprocessing.run``."""
    mode_suffix = MODES[mode]["cache_suffix"]
    if embedding_pca > 0:
        embed_suffix = f"_{backbone}_pca{embedding_pca}"
    else:
        embed_suffix = "_noembed"
    suffix = mode_suffix + embed_suffix
    path = CACHE_DIR / f"train_features{suffix}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected preprocessing cache at {path}. "
            "Run preprocessing first (see evaluation.workflow)."
        )
    return pd.read_parquet(path)


# ----------------------------------------------------------------------
# Bundle-building helpers
# ----------------------------------------------------------------------
def _to_bundle(
    df: pd.DataFrame,
    name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> FeatureBundle:
    y = df["AdoptionSpeed"].values.astype(int)
    feature_cols = [c for c in df.columns if c not in ("AdoptionSpeed", "PetID")]
    X = df[feature_cols].values.astype(np.float32)
    return FeatureBundle(
        name=name,
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        feature_names=feature_cols,
    )


def build_all_feature_sets(
    run_dir: Path,
    mode: str = "all_multiclass",
    backbone: str = "alexnet",
    embedding_pca: int = 64,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, FeatureBundle]:
    """Build all three feature sets sharing the same train/val split.

    Assumes ``preprocessing.run`` has already produced the parquet caches
    for both ``embedding_pca=0`` (breed-PCA only) and
    ``embedding_pca=<embedding_pca>`` (breed-PCA + image embeddings).
    """
    run_dir = Path(run_dir)

    # 1) Load breed-PCA features (no embeddings). This preserves row order
    #    and gives us the label vector.
    breed_pca_df = _load_preprocessing_features(mode, backbone, embedding_pca=0)
    n_samples = len(breed_pca_df)
    y_all = breed_pca_df["AdoptionSpeed"].values.astype(int)

    # 2) Build a shared split
    train_idx, val_idx = build_or_load_split(
        n_samples=n_samples,
        y=y_all,
        run_dir=run_dir,
        val_size=val_size,
        random_state=random_state,
    )

    # 3) OHE-raw
    ohe_raw_df = build_ohe_raw_features(mode=mode)
    if len(ohe_raw_df) != n_samples:
        raise RuntimeError(
            f"Row-count mismatch between OHE-raw ({len(ohe_raw_df)}) "
            f"and breed-PCA ({n_samples}) features."
        )
    ohe_bundle = _to_bundle(ohe_raw_df, "ohe_raw", train_idx, val_idx)

    # 4) Breed-PCA
    breed_bundle = _to_bundle(breed_pca_df, "breed_pca", train_idx, val_idx)

    # 5) Image-embeddings PCA
    emb_df = _load_preprocessing_features(mode, backbone, embedding_pca=embedding_pca)
    emb_bundle = _to_bundle(emb_df, "embeddings_pca64", train_idx, val_idx)

    return {
        "ohe_raw": ohe_bundle,
        "breed_pca": breed_bundle,
        "embeddings_pca64": emb_bundle,
    }