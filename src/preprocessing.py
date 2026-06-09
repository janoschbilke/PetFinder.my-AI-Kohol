import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.data import get_data_root

CACHE_DIR = Path("cache")

NUMERIC_COLS = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
OHE_COLS = [
    "Type", "Gender", "State",
    "Vaccinated", "Dewormed", "Sterilized", "Health", "MaturitySize", "FurLength",
]
COLOR_COLS = ["Color1", "Color2", "Color3"]
BREED_COLS = ["Breed1", "Breed2"]
DROP_COLS = ["Name", "Description", "AdoptionSpeed", "RescuerID"]

BREED_PCA_COMPONENTS = 15

OHE_DROP_CATEGORIES: dict[str, int] = {
    "Vaccinated":   3,
    "Dewormed":     3,
    "Sterilized":   3,
    "Health":       1,
    "Type":         2,
    "Gender":       3,
    "MaturitySize": 4,
    "FurLength":    3,
    "State":        41326,
}

# Explicit category lists for OHE columns whose full value range may not appear in
# every split (e.g. "0 = Not Specified" values that are rare or absent).
# State categories are loaded at runtime from state_labels.csv.
OHE_CATEGORIES: dict[str, list[int]] = {
    "Type":         [1, 2],
    "Gender":       [1, 2, 3],
    "Vaccinated":   [1, 2, 3],
    "Dewormed":     [1, 2, 3],
    "Sterilized":   [1, 2, 3],
    "Health":       [0, 1, 2, 3],
    "MaturitySize": [0, 1, 2, 3, 4],
    "FurLength":    [0, 1, 2, 3],
}

# --- Mode configurations ---
# Each mode defines a subset of the data and target classes for experimentation.
# type_filter: 1=Dogs, 2=Cats, None=All
# classes: which AdoptionSpeed values to keep (relabeled to 0,1,... for the model)
MODES = {
    "all_multiclass": {
        "description": "All pets, all 5 adoption speed classes (original competition task)",
        "type_filter": None,
        "classes": [0, 1, 2, 3, 4],
        "cache_suffix": "",
    },
    "dogs_extreme": {
        "description": "Dogs only — same day (0) vs >100 days (4)",
        "type_filter": 1,
        "classes": [0, 4],
        "cache_suffix": "_dogs_extreme",
    },
    "dogs_month_vs_100": {
        "description": "Dogs only — 8-30 days (2) vs >100 days (4)",
        "type_filter": 1,
        "classes": [2, 4],
        "cache_suffix": "_dogs_month_vs_100",
    },
    "dogs_adjacent": {
        "description": "Dogs only — 8-30 days (2) vs 31-90 days (3)",
        "type_filter": 1,
        "classes": [2, 3],
        "cache_suffix": "_dogs_adjacent",
    },
    "cats_month_vs_100": {
        "description": "Cats only — 8-30 days (2) vs >100 days (4)",
        "type_filter": 2,
        "classes": [2, 4],
        "cache_suffix": "_cats_month_vs_100",
    },
}


def load_sentiment(pet_id: str, sentiment_dir: Path) -> dict:
    path = sentiment_dir / f"{pet_id}.json"
    if not path.exists():
        return {
            "sentiment_score": 0.0,
            "sentiment_magnitude": 0.0,
            "avg_sentence_score": 0.0,
            "avg_sentence_magnitude": 0.0,
            "num_sentences": 0,
        }
    with open(path) as f:
        data = json.load(f)
    doc = data.get("documentSentiment", {})
    sentences = data.get("sentences", [])
    sent_scores = [s["sentiment"]["score"] for s in sentences if "sentiment" in s]
    sent_mags = [s["sentiment"]["magnitude"] for s in sentences if "sentiment" in s]
    return {
        "sentiment_score": doc.get("score", 0.0),
        "sentiment_magnitude": doc.get("magnitude", 0.0),
        "avg_sentence_score": float(np.mean(sent_scores)) if sent_scores else 0.0,
        "avg_sentence_magnitude": float(np.mean(sent_mags)) if sent_mags else 0.0,
        "num_sentences": len(sentences),
    }


def load_metadata(pet_id: str, metadata_dir: Path) -> dict:
    image_idx = 1
    label_scores_all = []
    dom_r, dom_g, dom_b, dom_frac = [], [], [], []
    crop_conf = []

    while True:
        path = metadata_dir / f"{pet_id}-{image_idx}.json"
        if not path.exists():
            break
        with open(path) as f:
            data = json.load(f)

        labels = data.get("labelAnnotations", [])
        if labels:
            scores = [l["score"] for l in labels]
            label_scores_all.extend(scores)

        colors = (
            data.get("imagePropertiesAnnotation", {})
            .get("dominantColors", {})
            .get("colors", [])
        )
        if colors:
            top = colors[0]
            c = top.get("color", {})
            dom_r.append(c.get("red", 0))
            dom_g.append(c.get("green", 0))
            dom_b.append(c.get("blue", 0))
            dom_frac.append(top.get("pixelFraction", 0.0))

        hints = data.get("cropHintsAnnotation", {}).get("cropHints", [])
        if hints:
            crop_conf.append(hints[0].get("confidence", 0.0))

        image_idx += 1

    return {
        "meta_top_label_score": float(max(label_scores_all)) if label_scores_all else 0.0,
        "meta_mean_label_score": float(np.mean(label_scores_all)) if label_scores_all else 0.0,
        "meta_num_labels": len(label_scores_all),
        "meta_dom_color_R": float(np.mean(dom_r)) if dom_r else 0.0,
        "meta_dom_color_G": float(np.mean(dom_g)) if dom_g else 0.0,
        "meta_dom_color_B": float(np.mean(dom_b)) if dom_b else 0.0,
        "meta_dom_color_frac": float(np.mean(dom_frac)) if dom_frac else 0.0,
        "meta_crop_confidence": float(np.mean(crop_conf)) if crop_conf else 0.0,
    }


def multi_hot_encode(
    df: pd.DataFrame,
    cols: list[str],
    prefix: str,
    skip_zero: bool = True,
    known_values: set[int] | None = None,
) -> pd.DataFrame:
    """Create a multi-hot encoded DataFrame from multiple columns sharing the same value space.

    If *known_values* is provided it defines the complete set of columns to emit,
    ensuring that values never observed in *df* still produce an all-zero column.
    """
    if known_values is not None:
        all_values: set[int] = set(known_values)
    else:
        all_values = set()
        for col in cols:
            if col in df.columns:
                all_values.update(df[col].dropna().astype(int).unique())
    if skip_zero:
        all_values.discard(0)

    result_dict: dict[str, pd.Series] = {}
    for val in sorted(all_values):
        col_name = f"{prefix}_{val}"
        mask = pd.Series(0.0, index=df.index)
        for col in cols:
            if col in df.columns:
                mask = mask.where(df[col].astype(int) != val, 1.0)
        result_dict[col_name] = mask

    return pd.DataFrame(result_dict, index=df.index)


def apply_breed_pca(
    train_breed_df: pd.DataFrame,
    test_breed_df: pd.DataFrame,
    n_components: int = BREED_PCA_COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Apply PCA to breed multi-hot columns to reduce sparsity.

    PCA is fitted on the training data only and applied to both train and test
    to avoid data leakage.
    """
    # Ensure test has the same columns as train (fill missing with 0)
    missing_cols = set(train_breed_df.columns) - set(test_breed_df.columns)
    if missing_cols:
        for col in missing_cols:
            test_breed_df[col] = 0.0
    extra_cols = set(test_breed_df.columns) - set(train_breed_df.columns)
    if extra_cols:
        test_breed_df = test_breed_df.drop(columns=list(extra_cols))

    # Align column order
    test_breed_df = test_breed_df[train_breed_df.columns]

    # Clamp n_components to the number of available columns
    actual_components = min(n_components, train_breed_df.shape[1], train_breed_df.shape[0])

    # Fit PCA on training data only
    pca = PCA(n_components=actual_components, random_state=42)
    train_pca = pca.fit_transform(train_breed_df.values)
    test_pca = pca.transform(test_breed_df.values)

    # Create DataFrames with named columns
    pca_col_names = [f"Breed_PCA_{i}" for i in range(actual_components)]
    train_pca_df = pd.DataFrame(train_pca, index=train_breed_df.index, columns=pca_col_names)
    test_pca_df = pd.DataFrame(test_pca, index=test_breed_df.index, columns=pca_col_names)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Breed PCA: {train_breed_df.shape[1]} cols → {actual_components} components "
          f"(explained variance: {explained_var:.1%})")

    return train_pca_df, test_pca_df, pca


def build_features(
    df: pd.DataFrame,
    sentiment_dir: Path,
    metadata_dir: Path,
    data_root: Path,
    exclude_type_col: bool = False,
    type_filter: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature DataFrame from raw data.

    Args:
        df: Raw dataframe (with PetID, AdoptionSpeed already handled by caller)
        sentiment_dir: Path to sentiment JSON files
        metadata_dir: Path to metadata JSON files
        data_root: Root data directory (for breed/color/state labels)
        exclude_type_col: If True, don't include Type in OHE (useful when all are same type)
        type_filter: If set, only include breeds for this type (1=dog, 2=cat)

    Returns:
        Tuple of (features_df, breed_multihot_df)
    """
    df = df.copy()

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    pet_ids = df["PetID"].tolist()

    sentiment_rows = [
        load_sentiment(pid, sentiment_dir)
        for pid in tqdm(pet_ids, desc="  Loading sentiment", leave=False)
    ]
    sentiment_df = pd.DataFrame(sentiment_rows, index=df.index)

    metadata_rows = [
        load_metadata(pid, metadata_dir)
        for pid in tqdm(pet_ids, desc="  Loading metadata", leave=False)
    ]
    metadata_df = pd.DataFrame(metadata_rows, index=df.index)

    df = df.drop(columns=["PetID"])

    # Load label CSVs for known value sets
    breed_labels_df = pd.read_csv(data_root / "breed_labels.csv")
    color_ids: set[int] = set(
        pd.read_csv(data_root / "color_labels.csv")["ColorID"].astype(int).tolist()
    )
    state_ids: list[int] = sorted(
        pd.read_csv(data_root / "state_labels.csv")["StateID"].astype(int).tolist()
    )

    # Filter breed IDs to only relevant type (dog=1, cat=2) if specified
    if type_filter is not None:
        breed_ids: set[int] = set(
            breed_labels_df[breed_labels_df["Type"] == type_filter]["BreedID"].astype(int).tolist()
        )
    else:
        breed_ids: set[int] = set(breed_labels_df["BreedID"].astype(int).tolist())

    color_cols_present = [c for c in COLOR_COLS if c in df.columns]
    breed_cols_present = [c for c in BREED_COLS if c in df.columns]
    color_mh = multi_hot_encode(df, color_cols_present, "Color", known_values=color_ids)
    breed_mh = multi_hot_encode(df, breed_cols_present, "Breed", known_values=breed_ids)

    # Determine which OHE columns to use
    ohe_cols_to_use = list(OHE_COLS)
    if exclude_type_col:
        ohe_cols_to_use = [c for c in ohe_cols_to_use if c != "Type"]
        # Drop the Type column from the dataframe
        if "Type" in df.columns:
            df = df.drop(columns=["Type"])

    # Cast each OHE column to pd.Categorical with its full known category list
    for col, cats in OHE_CATEGORIES.items():
        if col in df.columns and col in ohe_cols_to_use:
            df[col] = pd.Categorical(df[col].astype(int), categories=cats)
    if "State" in df.columns and "State" in ohe_cols_to_use:
        df["State"] = pd.Categorical(df["State"].astype(int), categories=state_ids)

    ohe_cols_present = [c for c in ohe_cols_to_use if c in df.columns]
    df = df.drop(columns=color_cols_present + breed_cols_present)
    df = pd.get_dummies(df, columns=ohe_cols_present, dtype=float)

    # Drop the reference category column for each OHE column to avoid multicollinearity.
    ref_cols_to_drop = [
        f"{col}_{val}"
        for col, val in OHE_DROP_CATEGORIES.items()
        if f"{col}_{val}" in df.columns
    ]
    if ref_cols_to_drop:
        df = df.drop(columns=ref_cols_to_drop)

    # NOTE: breed_mh is returned separately for PCA processing in run()
    df = pd.concat([df, color_mh, sentiment_df, metadata_df], axis=1)

    return df, breed_mh


def run(force: bool = False, mode: str = "all_multiclass") -> None:
    """Run preprocessing pipeline.

    Args:
        force: If True, ignore cache and recompute
        mode: One of the keys in MODES dict (e.g. "all_multiclass", "dogs_extreme", etc.)
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODES.keys())}")

    mode_config = MODES[mode]
    suffix = mode_config["cache_suffix"]

    CACHE_DIR.mkdir(exist_ok=True)

    train_out = CACHE_DIR / f"train_features{suffix}.parquet"
    test_out = CACHE_DIR / f"test_features{suffix}.parquet"

    if not force and train_out.exists() and test_out.exists():
        print(f"Preprocessing cache found ({mode} mode), skipping. Use --force to recompute.")
        return

    print(f"\n{'='*60}")
    print(f"  Preprocessing Mode: {mode}")
    print(f"  {mode_config['description']}")
    print(f"{'='*60}\n")

    data_root = get_data_root()
    train_csv = data_root / "train" / "train.csv"
    test_csv = data_root / "test" / "test.csv"
    sentiment_dir_train = data_root / "train_sentiment"
    sentiment_dir_test = data_root / "test_sentiment"
    metadata_dir_train = data_root / "train_metadata"
    metadata_dir_test = data_root / "test_metadata"

    # --- Load and filter training data ---
    print("Loading train CSV...")
    train_df = pd.read_csv(train_csv)

    # Filter by type if specified
    type_filter = mode_config["type_filter"]
    if type_filter is not None:
        n_before = len(train_df)
        train_df = train_df[train_df["Type"] == type_filter].reset_index(drop=True)
        print(f"  Filtered to Type=={type_filter}: {n_before} → {len(train_df)} samples")

    # Filter by classes if not all classes
    classes = mode_config["classes"]
    if classes != [0, 1, 2, 3, 4]:
        n_before = len(train_df)
        train_df = train_df[train_df["AdoptionSpeed"].isin(classes)].reset_index(drop=True)
        print(f"  Filtered to classes {classes}: {n_before} → {len(train_df)} samples")

        # Relabel classes to 0, 1, 2, ... for the model
        class_map = {orig: new for new, orig in enumerate(sorted(classes))}
        print(f"  Class mapping: {class_map}")
        train_df["AdoptionSpeed"] = train_df["AdoptionSpeed"].map(class_map)

    y_train = train_df["AdoptionSpeed"].copy()
    print(f"  Training samples: {len(train_df)}")
    print(f"  Class distribution:\n{y_train.value_counts().sort_index().to_string()}")

    # Determine if we should exclude the Type column
    exclude_type = type_filter is not None

    print("\nBuilding train features (sentiment + metadata)...")
    train_features, train_breed_mh = build_features(
        train_df, sentiment_dir_train, metadata_dir_train, data_root,
        exclude_type_col=exclude_type,
        type_filter=type_filter,
    )
    train_features["AdoptionSpeed"] = y_train.values

    # --- Load and filter test data ---
    print("\nLoading test CSV...")
    test_df = pd.read_csv(test_csv)

    # Apply same type filter to test (for consistent feature space)
    if type_filter is not None:
        test_df = test_df[test_df["Type"] == type_filter].reset_index(drop=True)
        print(f"  Filtered test to Type=={type_filter}: {len(test_df)} samples")

    test_pet_ids = test_df["PetID"].copy()

    print("Building test features (sentiment + metadata)...")
    test_features, test_breed_mh = build_features(
        test_df, sentiment_dir_test, metadata_dir_test, data_root,
        exclude_type_col=exclude_type,
        type_filter=type_filter,
    )
    test_features["PetID"] = test_pet_ids.values

    # Apply PCA to breed multi-hot columns (fit on train, transform both)
    print("\nApplying PCA to breed features...")
    train_breed_pca, test_breed_pca, breed_pca = apply_breed_pca(
        train_breed_mh, test_breed_mh, n_components=BREED_PCA_COMPONENTS
    )

    # Save the fitted PCA object for reproducibility
    pca_path = CACHE_DIR / f"breed_pca{suffix}.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump(breed_pca, f)
    print(f"  Saved PCA transformer: {pca_path}")

    # Concatenate PCA breed features with the rest
    train_features = pd.concat([train_features, train_breed_pca], axis=1)
    test_features = pd.concat([test_features, test_breed_pca], axis=1)

    # Align columns between train and test
    train_cols = set(train_features.columns) - {"AdoptionSpeed"}
    test_cols = set(test_features.columns) - {"PetID"}
    missing_cols = sorted(train_cols - test_cols)
    if missing_cols:
        test_features = pd.concat(
            [test_features, pd.DataFrame(0.0, index=test_features.index, columns=missing_cols)],
            axis=1,
        )
    extra_cols = test_cols - train_cols
    if extra_cols:
        test_features = test_features.drop(columns=list(extra_cols))

    feature_cols = sorted(train_cols)
    test_features = test_features[feature_cols + ["PetID"]]
    train_features = train_features[feature_cols + ["AdoptionSpeed"]]

    print(f"\nTrain feature matrix: {train_features.shape}")
    print(f"Test feature matrix:  {test_features.shape}")

    train_features.to_parquet(train_out, index=False)
    test_features.to_parquet(test_out, index=False)
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PetFinder data for model training."
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute")
    parser.add_argument("--mode", choices=list(MODES.keys()), default="all_multiclass",
                        help="Data subset/task mode (default: all_multiclass)")
    args = parser.parse_args()
    run(force=args.force, mode=args.mode)
