from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

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

def _load_pca(pca_path: Path) -> PCA:
    if not pca_path.exists():
        raise FileNotFoundError(
            f"{pca_path} not found. Run preprocessing first."
        )
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    print(
        f"Loaded PCA with {pca.n_components_} components, "
        f"{pca.components_.shape[1]} input breed columns"
    )
    return pca

def _load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = get_data_root()
    train_df = pd.read_csv(data_root / "train" / "train.csv")
    breed_labels = pd.read_csv(data_root / "breed_labels.csv")
    return train_df, breed_labels


def _build_breed_matrix(
    train_df: pd.DataFrame, breed_labels: pd.DataFrame
) -> pd.DataFrame:
    breed_ids = set(breed_labels["BreedID"].astype(int).tolist())
    breed_cols_present = [c for c in BREED_COLS if c in train_df.columns]
    return multi_hot_encode(
        train_df, breed_cols_present, "Breed", known_values=breed_ids
    )


def _compute_scores(pca: PCA, breed_mh: pd.DataFrame) -> pd.DataFrame:
    expected_n_cols = pca.components_.shape[1]
    if breed_mh.shape[1] != expected_n_cols:
        raise RuntimeError(
            f"Breed multi-hot has {breed_mh.shape[1]} columns but PCA expects "
            f"{expected_n_cols}. Did the breed list change since fitting?"
        )
    scores = pca.transform(breed_mh.values)
    return pd.DataFrame(
        scores,
        index=breed_mh.index,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )


def _build_loading_meta(
    breed_mh: pd.DataFrame, breed_labels: pd.DataFrame
) -> pd.DataFrame:
    breed_col_names = breed_mh.columns.tolist()
    breed_id_per_col = [int(c.split("_")[1]) for c in breed_col_names]
    name_lookup = breed_labels.set_index("BreedID")[["BreedName", "Type"]]
    loading_meta = name_lookup.loc[breed_id_per_col].reset_index(drop=False)
    loading_meta = loading_meta.rename(columns={"BreedID": "breed_id"})
    loading_meta["display"] = loading_meta.apply(_format_breed_name, axis=1)
    return loading_meta


def _compute_correlations(
    score_df: pd.DataFrame, train_df: pd.DataFrame
) -> pd.DataFrame:
    corr_df = pd.DataFrame(
        index=score_df.columns, columns=INTERPRETABLE_FEATURES, dtype=float
    )
    for pc in score_df.columns:
        for feat in INTERPRETABLE_FEATURES:
            if feat in train_df.columns:
                corr_df.loc[pc, feat] = score_df[pc].corr(train_df[feat])
    return corr_df

def _format_pet_line(
    idx: int,
    pc_name: str,
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    name_lookup: pd.DataFrame,
) -> str:
    row = train_df.loc[idx]
    b1 = name_lookup["BreedName"].get(int(row["Breed1"]), "Unknown")
    b2 = name_lookup["BreedName"].get(int(row["Breed2"]), "")
    type_s = {1: "Dog", 2: "Cat"}.get(int(row["Type"]), "?")
    breeds = b1 if not b2 or b2 == "Unknown" else f"{b1} / {b2}"
    return (
        f"    score={score_df.loc[idx, pc_name]:+.3f}  "
        f"PetID={row['PetID']}  Type={type_s}  Breeds={breeds}"
    )


def _format_component_section(
    i: int,
    pca: PCA,
    corr_df: pd.DataFrame,
    loading_meta: pd.DataFrame,
    score_df: pd.DataFrame,
    train_df: pd.DataFrame,
    name_lookup: pd.DataFrame,
) -> list[str]:
    pc_name = f"PC{i + 1}"
    loadings = pca.components_[i]
    evr = pca.explained_variance_ratio_[i] * 100

    order = np.argsort(loadings)
    top_pos_idx = order[::-1][:TOP_N_LOADINGS]
    top_neg_idx = order[:TOP_N_LOADINGS]

    lines: list[str] = []
    lines.append("-" * 20)
    lines.append(
        f"{pc_name}  (explained variance: {evr:.1f}%)  "
        f"suggested label: {_suggest_label(corr_df.loc[pc_name])}"
    )
    lines.append("-" * 20)

    lines.append(f"  Top +{TOP_N_LOADINGS} breed loadings (positive end):")
    for idx in top_pos_idx:
        lines.append(
            f"    {loadings[idx]:+.3f}  {loading_meta.loc[idx, 'display']}"
        )
    lines.append(f"  Top -{TOP_N_LOADINGS} breed loadings (negative end):")
    for idx in top_neg_idx:
        lines.append(
            f"    {loadings[idx]:+.3f}  {loading_meta.loc[idx, 'display']}"
        )

    lines.append("  Correlation with raw features:")
    for feat, val in corr_df.loc[pc_name].items():
        mark = "  *" if (not pd.isna(val) and abs(val) >= CORR_THRESHOLD) else ""
        lines.append(f"    {feat:>14}: {val:+.3f}{mark}")

    sorted_pets = score_df[pc_name].sort_values()
    bottom_idx = sorted_pets.index[:TOP_N_EXAMPLES]
    top_idx = sorted_pets.index[-TOP_N_EXAMPLES:][::-1]

    lines.append(f"  Example pets at HIGH end of {pc_name}:")
    for idx in top_idx:
        lines.append(_format_pet_line(idx, pc_name, train_df, score_df, name_lookup))
    lines.append(f"  Example pets at LOW end of {pc_name}:")
    for idx in bottom_idx:
        lines.append(_format_pet_line(idx, pc_name, train_df, score_df, name_lookup))
    lines.append("")
    return lines


def _format_summary(pca: PCA, corr_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("Summary - suggested axis labels")
    lines.append("=" * 78)
    for i in range(pca.n_components_):
        pc_name = f"PC{i + 1}"
        evr = pca.explained_variance_ratio_[i] * 100
        lines.append(
            f"  {pc_name} ({evr:5.1f}% var): {_suggest_label(corr_df.loc[pc_name])}"
        )
    return lines


def _build_report(
    pca: PCA,
    corr_df: pd.DataFrame,
    loading_meta: pd.DataFrame,
    score_df: pd.DataFrame,
    train_df: pd.DataFrame,
    name_lookup: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("Breed PCA Component Interpretation Report")
    lines.append("=" * 20)
    lines.append(f"Components:       {pca.n_components_}")
    lines.append(f"Input breeds:     {pca.components_.shape[1]}")
    lines.append(f"Training samples: {len(train_df)}")
    lines.append(
        f"Total explained variance: "
        f"{pca.explained_variance_ratio_.sum() * 100:.1f}%"
    )
    lines.append("")

    for i in range(pca.n_components_):
        lines.extend(
            _format_component_section(
                i, pca, corr_df, loading_meta, score_df, train_df, name_lookup
            )
        )

    lines.extend(_format_summary(pca, corr_df))
    return "\n".join(lines)


def _plot_correlation_heatmap(
    corr_df: pd.DataFrame, n_components: int, output_dir: Path
) -> Path:
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
    path = output_dir / "breed_pca_correlations.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_top_loadings(
    pca: PCA, loading_meta: pd.DataFrame, output_dir: Path
) -> Path:
    n_components = pca.n_components_
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
    path = output_dir / "breed_pca_loadings.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_component_csvs(
    pca: PCA, loading_meta: pd.DataFrame, output_dir: Path
) -> Path:
    csv_dir = output_dir / "components"
    csv_dir.mkdir(parents=True, exist_ok=True)
    type_str = loading_meta["Type"].map({1: "Dog", 2: "Cat"}).fillna("?")
    for i in range(pca.n_components_):
        loadings = pca.components_[i]
        df = pd.DataFrame(
            {
                "breed_id": loading_meta["breed_id"].astype(int),
                "breed_name": loading_meta["BreedName"],
                "type": type_str,
                "loading": loadings,
            }
        )
        df = df.sort_values("loading", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
        df["loading"] = df["loading"].round(6)
        df.to_csv(csv_dir / f"pc{i + 1:02d}_loadings.csv", index=False)
    return csv_dir


def _plot_scree(pca: PCA, output_dir: Path) -> Path:
    n_components = pca.n_components_
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
    path = output_dir / "breed_pca_explained_variance.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path

def run(
    output_dir: Path = OUTPUT_DIR, cache_dir: Path = CACHE_DIR
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pca = _load_pca(cache_dir / "breed_pca.pkl")
    train_df, breed_labels = _load_training_data()
    breed_mh = _build_breed_matrix(train_df, breed_labels)
    score_df = _compute_scores(pca, breed_mh)
    loading_meta = _build_loading_meta(breed_mh, breed_labels)
    name_lookup = breed_labels.set_index("BreedID")[["BreedName", "Type"]]
    corr_df = _compute_correlations(score_df, train_df)

    report = _build_report(
        pca, corr_df, loading_meta, score_df, train_df, name_lookup
    )
    report_path = output_dir / "breed_pca_analysis.txt"
    report_path.write_text(report)
    print(f"Wrote report:        {report_path}")

    corr_path = _plot_correlation_heatmap(corr_df, pca.n_components_, output_dir)
    print(f"Wrote heatmap:       {corr_path}")

    loadings_path = _plot_top_loadings(pca, loading_meta, output_dir)
    print(f"Wrote loadings:      {loadings_path}")

    scree_path = _plot_scree(pca, output_dir)
    print(f"Wrote scree plot:    {scree_path}")

    csv_dir = _write_component_csvs(pca, loading_meta, output_dir)
    print(f"Wrote component CSVs: {csv_dir}")

    print("\nDone.")


if __name__ == "__main__":
    run()
