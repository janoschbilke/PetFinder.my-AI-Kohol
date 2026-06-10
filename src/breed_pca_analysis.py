"""Analyse the fitted breed-PCA components and try to interpret them.

Loads the cached PCA transformer (`cache/breed_pca.pkl`), reconstructs the
training breed multi-hot matrix and produces:

- A textual report listing the top positive / negative breed loadings per
  component, correlations of each PC score with interpretable raw features
  (Type, MaturitySize, FurLength, Age, Fee, AdoptionSpeed, ...) and example
  pets at each extreme.
- A heatmap of those correlations.
- Bar charts of the top loadings for the first few components.
- A scree plot of explained variance.
- Automatic "label suggestions" based on strong correlations.

Outputs go into `outputs/pca_analysis/`.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import get_data_root
from src.preprocessing import BREED_COLS, multi_hot_encode

OUTPUT_DIR = Path("outputs/pca_analysis")
CACHE_DIR = Path("cache")

TOP_N_LOADINGS = 10
TOP_N_EXAMPLES = 5
LOADING_PLOT_COMPONENTS = 6

INTERPRETABLE_FEATURES = [
    "Type",
    "MaturitySize",
    "FurLength",
    "Age",
    "Fee",
    "PhotoAmt",
    "Quantity",
    "Vaccinated",
    "Sterilized",
    "Health",
    "AdoptionSpeed",
]

CORR_THRESHOLD = 0.30


def _format_breed_name(row: pd.Series) -> str:
    type_str = {1: "Dog", 2: "Cat"}.get(int(row["Type"]), "?")
    return f"{row['BreedName']} ({type_str})"


def _suggest_label(corrs: pd.Series) -> str:
    parts: list[str] = []
    for feat, val in corrs.items():
        if pd.isna(val):
            continue
        if abs(val) >= CORR_THRESHOLD:
            sign = "+" if val > 0 else "-"
            parts.append(f"{sign}{feat}({val:+.2f})")
    return ", ".join(parts) if parts else "no strong signal"


def run(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pca_path = CACHE_DIR / "breed_pca.pkl"
    if not pca_path.exists():
        raise FileNotFoundError(
            f"{pca_path} not found. Run preprocessing first."
        )

    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    n_components = pca.n_components_
    print(
        f"Loaded PCA with {n_components} components, "
        f"{pca.components_.shape[1]} input breed columns"
    )

    data_root = get_data_root()
    train_df = pd.read_csv(data_root / "train" / "train.csv")
    breed_labels = pd.read_csv(data_root / "breed_labels.csv")

    breed_ids = set(breed_labels["BreedID"].astype(int).tolist())
    breed_cols_present = [c for c in BREED_COLS if c in train_df.columns]
    breed_mh = multi_hot_encode(
        train_df, breed_cols_present, "Breed", known_values=breed_ids
    )

    expected_n_cols = pca.components_.shape[1]
    if breed_mh.shape[1] != expected_n_cols:
        raise RuntimeError(
            f"Breed multi-hot has {breed_mh.shape[1]} columns but PCA expects "
            f"{expected_n_cols}. Did the breed list change since fitting?"
        )

    scores = pca.transform(breed_mh.values)
    score_df = pd.DataFrame(
        scores,
        index=train_df.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    breed_col_names = breed_mh.columns.tolist()
    breed_id_per_col = [int(c.split("_")[1]) for c in breed_col_names]
    name_lookup = breed_labels.set_index("BreedID")[["BreedName", "Type"]]
    loading_meta = name_lookup.loc[breed_id_per_col].reset_index(drop=True)
    loading_meta["display"] = loading_meta.apply(_format_breed_name, axis=1)

    corr_df = pd.DataFrame(
        index=score_df.columns, columns=INTERPRETABLE_FEATURES, dtype=float
    )
    for pc in score_df.columns:
        for feat in INTERPRETABLE_FEATURES:
            if feat in train_df.columns:
                corr_df.loc[pc, feat] = score_df[pc].corr(train_df[feat])

    report_lines: list[str] = []
    report_lines.append("Breed PCA Component Interpretation Report")
    report_lines.append("=" * 20)
    report_lines.append(f"Components:       {n_components}")
    report_lines.append(f"Input breeds:     {expected_n_cols}")
    report_lines.append(f"Training samples: {len(train_df)}")
    report_lines.append(
        f"Total explained variance: "
        f"{pca.explained_variance_ratio_.sum() * 100:.1f}%"
    )
    report_lines.append("")

    for i in range(n_components):
        pc_name = f"PC{i + 1}"
        loadings = pca.components_[i]
        evr = pca.explained_variance_ratio_[i] * 100

        order = np.argsort(loadings)
        top_pos_idx = order[::-1][:TOP_N_LOADINGS]
        top_neg_idx = order[:TOP_N_LOADINGS]

        report_lines.append("-" * 20)
        report_lines.append(
            f"{pc_name}  (explained variance: {evr:.1f}%)  "
            f"suggested label: {_suggest_label(corr_df.loc[pc_name])}"
        )
        report_lines.append("-" * 20)

        report_lines.append(f"  Top +{TOP_N_LOADINGS} breed loadings (positive end):")
        for idx in top_pos_idx:
            report_lines.append(
                f"    {loadings[idx]:+.3f}  {loading_meta.loc[idx, 'display']}"
            )
        report_lines.append(f"  Top -{TOP_N_LOADINGS} breed loadings (negative end):")
        for idx in top_neg_idx:
            report_lines.append(
                f"    {loadings[idx]:+.3f}  {loading_meta.loc[idx, 'display']}"
            )

        report_lines.append("  Correlation with raw features:")
        for feat, val in corr_df.loc[pc_name].items():
            mark = "  *" if (not pd.isna(val) and abs(val) >= CORR_THRESHOLD) else ""
            report_lines.append(f"    {feat:>14}: {val:+.3f}{mark}")

        sorted_pets = score_df[pc_name].sort_values()
        bottom_idx = sorted_pets.index[:TOP_N_EXAMPLES]
        top_idx = sorted_pets.index[-TOP_N_EXAMPLES:][::-1]

        def _pet_line(idx: int) -> str:
            row = train_df.loc[idx]
            b1 = name_lookup["BreedName"].get(int(row["Breed1"]), "Unknown")
            b2 = name_lookup["BreedName"].get(int(row["Breed2"]), "")
            type_s = {1: "Dog", 2: "Cat"}.get(int(row["Type"]), "?")
            breeds = b1 if not b2 or b2 == "Unknown" else f"{b1} / {b2}"
            return (
                f"    score={score_df.loc[idx, pc_name]:+.3f}  "
                f"PetID={row['PetID']}  Type={type_s}  Breeds={breeds}"
            )

        report_lines.append(f"  Example pets at HIGH end of {pc_name}:")
        for idx in top_idx:
            report_lines.append(_pet_line(idx))
        report_lines.append(f"  Example pets at LOW end of {pc_name}:")
        for idx in bottom_idx:
            report_lines.append(_pet_line(idx))
        report_lines.append("")

    report_lines.append("=" * 78)
    report_lines.append("Summary - suggested axis labels")
    report_lines.append("=" * 78)
    for i in range(n_components):
        pc_name = f"PC{i + 1}"
        evr = pca.explained_variance_ratio_[i] * 100
        report_lines.append(
            f"  {pc_name} ({evr:5.1f}% var): {_suggest_label(corr_df.loc[pc_name])}"
        )

    report_path = OUTPUT_DIR / "breed_pca_analysis.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"Wrote report:        {report_path}")

    # Heatmap of correlations
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * n_components)))
    sns.heatmap(
        corr_df.astype(float),
        annot=True,
        fmt="+.2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Breed-PCA components vs. interpretable raw features")
    plt.tight_layout()
    corr_path = OUTPUT_DIR / "breed_pca_correlations.png"
    plt.savefig(corr_path, dpi=150)
    plt.close(fig)
    print(f"Wrote heatmap:       {corr_path}")

    # Top-loading bar charts for the first N components
    n_plot = min(LOADING_PLOT_COMPONENTS, n_components)
    n_cols_plot = 2
    n_rows_plot = (n_plot + n_cols_plot - 1) // n_cols_plot
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(14, 4 * n_rows_plot))
    axes = np.atleast_2d(axes).reshape(n_rows_plot, n_cols_plot)
    for i in range(n_plot):
        ax = axes[i // n_cols_plot, i % n_cols_plot]
        loadings = pca.components_[i]
        order = np.argsort(loadings)
        sel = np.concatenate([order[:TOP_N_LOADINGS], order[::-1][:TOP_N_LOADINGS]])
        sel_loadings = loadings[sel]
        sel_labels = loading_meta.loc[sel, "display"].tolist()
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in sel_loadings]
        y_pos = np.arange(len(sel_loadings))
        ax.barh(y_pos, sel_loadings, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sel_labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        evr = pca.explained_variance_ratio_[i] * 100
        ax.set_title(f"PC{i + 1}  ({evr:.1f}% var)")
        ax.set_xlabel("Loading")
    # hide unused axes
    for j in range(n_plot, n_rows_plot * n_cols_plot):
        axes[j // n_cols_plot, j % n_cols_plot].axis("off")
    fig.suptitle("Breed-PCA — top loadings per component", fontsize=14)
    plt.tight_layout()
    loadings_path = OUTPUT_DIR / "breed_pca_loadings.png"
    plt.savefig(loadings_path, dpi=150)
    plt.close(fig)
    print(f"Wrote loadings:      {loadings_path}")

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 5))
    evr = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(evr)
    x = np.arange(1, n_components + 1)
    ax.bar(x, evr, color="steelblue", label="Per-component")
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="darkorange", marker="o", label="Cumulative")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance (%)")
    ax2.set_ylabel("Cumulative (%)")
    ax.set_title("Breed-PCA scree plot")
    ax.set_xticks(x)
    plt.tight_layout()
    scree_path = OUTPUT_DIR / "breed_pca_explained_variance.png"
    plt.savefig(scree_path, dpi=150)
    plt.close(fig)
    print(f"Wrote scree plot:    {scree_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpret the fitted breed-PCA components."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)
