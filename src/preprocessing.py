import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def build_features(df: pd.DataFrame, sentiment_dir: Path, metadata_dir: Path, data_root: Path) -> pd.DataFrame:
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

    breed_ids: set[int] = set(
        pd.read_csv(data_root / "breed_labels.csv")["BreedID"].astype(int).tolist()
    )
    color_ids: set[int] = set(
        pd.read_csv(data_root / "color_labels.csv")["ColorID"].astype(int).tolist()
    )
    state_ids: list[int] = sorted(
        pd.read_csv(data_root / "state_labels.csv")["StateID"].astype(int).tolist()
    )

    color_cols_present = [c for c in COLOR_COLS if c in df.columns]
    breed_cols_present = [c for c in BREED_COLS if c in df.columns]
    color_mh = multi_hot_encode(df, color_cols_present, "Color", known_values=color_ids)
    breed_mh = multi_hot_encode(df, breed_cols_present, "Breed", known_values=breed_ids)

    # Cast each OHE column to pd.Categorical with its full known category list so that
    # pd.get_dummies emits a column for every value, including those absent from this
    # split (e.g. "0 = Not Specified" for Health/MaturitySize/FurLength, or Perlis for State).
    for col, cats in OHE_CATEGORIES.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col].astype(int), categories=cats)
    if "State" in df.columns:
        df["State"] = pd.Categorical(df["State"].astype(int), categories=state_ids)

    ohe_cols_present = [c for c in OHE_COLS if c in df.columns]
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

    df = pd.concat([df, color_mh, breed_mh, sentiment_df, metadata_df], axis=1)

    return df


def run(force: bool = False) -> None:
    CACHE_DIR.mkdir(exist_ok=True)

    train_out = CACHE_DIR / "train_features.parquet"
    test_out = CACHE_DIR / "test_features.parquet"

    if not force and train_out.exists() and test_out.exists():
        print("Preprocessing cache found, skipping. Use --force to recompute.")
        return

    data_root = get_data_root()
    train_csv = data_root / "train" / "train.csv"
    test_csv = data_root / "test" / "test.csv"
    sentiment_dir_train = data_root / "train_sentiment"
    sentiment_dir_test = data_root / "test_sentiment"
    metadata_dir_train = data_root / "train_metadata"
    metadata_dir_test = data_root / "test_metadata"

    print("Loading train CSV...")
    train_df = pd.read_csv(train_csv)
    y_train = train_df["AdoptionSpeed"].copy()

    print("Building train features (sentiment + metadata)...")
    train_features = build_features(train_df, sentiment_dir_train, metadata_dir_train, data_root)
    train_features["AdoptionSpeed"] = y_train.values

    print("Loading test CSV...")
    test_df = pd.read_csv(test_csv)
    test_pet_ids = test_df["PetID"].copy()

    print("Building test features (sentiment + metadata)...")
    test_features = build_features(test_df, sentiment_dir_test, metadata_dir_test, data_root)
    test_features["PetID"] = test_pet_ids.values

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

    print(f"Train feature matrix: {train_features.shape}")
    print(f"Test feature matrix:  {test_features.shape}")

    train_features.to_parquet(train_out, index=False)
    test_features.to_parquet(test_out, index=False)
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)