"""Generate a summary of experiment results with visualizations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("documentation")


# Experiment results from our runs
RESULTS = {
    "dogs_extreme": {
        "description": "Dogs: Same day vs >100 days",
        "classes": ["Same day", ">100 days"],
        "samples": 2584,
        "balance": "170 vs 2414",
        "auc": 0.798,
        "accuracy": 0.935,
        "f1_macro": 0.500,
        "f1_macro_opt": 0.646,
        "optimal_threshold": 0.78,
    },
    "dogs_month_vs_100": {
        "description": "Dogs: 8-30 days vs >100 days",
        "classes": ["8-30 days", ">100 days"],
        "samples": 4578,
        "balance": "2164 vs 2414",
        "auc": 0.788,
        "accuracy": 0.720,
        "f1_macro": 0.719,
        "f1_macro_opt": 0.720,
        "optimal_threshold": 0.51,
    },
    "cats_month_vs_100": {
        "description": "Cats: 8-30 days vs >100 days",
        "classes": ["8-30 days", ">100 days"],
        "samples": 3656,
        "balance": "1873 vs 1783",
        "auc": 0.739,
        "accuracy": 0.682,
        "f1_macro": 0.682,
        "f1_macro_opt": 0.684,
        "optimal_threshold": 0.46,
    },
    "dogs_adjacent": {
        "description": "Dogs: 8-30 days vs 31-90 days",
        "classes": ["8-30 days", "31-90 days"],
        "samples": 4113,
        "balance": "2164 vs 1949",
        "auc": 0.576,
        "accuracy": 0.559,
        "f1_macro": 0.554,
        "f1_macro_opt": 0.554,
        "optimal_threshold": 0.50,
    },
}


def plot_auc_comparison():
    """Bar chart comparing AUC-ROC across all experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    modes = list(RESULTS.keys())
    aucs = [RESULTS[m]["auc"] for m in modes]
    descriptions = [RESULTS[m]["description"] for m in modes]

    colors = ["#2ecc71" if a > 0.7 else "#f39c12" if a > 0.6 else "#e74c3c" for a in aucs]

    bars = ax.barh(descriptions, aucs, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=11, fontweight="bold")

    # Reference line at 0.5 (random)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Random (0.5)")

    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_title("LightGBM Binary Classification — AUC-ROC by Experiment", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "results_auc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'results_auc_comparison.png'}")


def plot_class_distance_vs_auc():
    """Scatter plot showing how class distance correlates with AUC."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Class distance: difference between the original AdoptionSpeed class numbers
    experiments = [
        ("dogs_adjacent", 1, 0.576, "Dogs: 2 vs 3"),
        ("cats_month_vs_100", 2, 0.739, "Cats: 2 vs 4"),
        ("dogs_month_vs_100", 2, 0.788, "Dogs: 2 vs 4"),
        ("dogs_extreme", 4, 0.798, "Dogs: 0 vs 4"),
    ]

    distances = [e[1] for e in experiments]
    aucs = [e[2] for e in experiments]
    labels = [e[3] for e in experiments]

    ax.scatter(distances, aucs, s=200, c=aucs, cmap="RdYlGn", vmin=0.5, vmax=0.85,
              edgecolors="black", linewidths=1.5, zorder=5)

    for d, a, label in zip(distances, aucs, labels):
        ax.annotate(label, (d, a), textcoords="offset points", xytext=(10, -5),
                   fontsize=10)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Random")
    ax.set_xlabel("Class Distance (difference in AdoptionSpeed categories)", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("Model Performance vs. Class Separation", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 5)
    ax.set_ylim(0.45, 0.85)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "results_distance_vs_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'results_distance_vs_auc.png'}")


def plot_metrics_summary():
    """Grouped bar chart of key metrics per experiment."""
    fig, ax = plt.subplots(figsize=(10, 5))

    modes = list(RESULTS.keys())
    descriptions = [RESULTS[m]["description"] for m in modes]
    x = np.arange(len(modes))
    width = 0.25

    aucs = [RESULTS[m]["auc"] for m in modes]
    accs = [RESULTS[m]["accuracy"] for m in modes]
    f1s = [RESULTS[m]["f1_macro_opt"] for m in modes]

    bars1 = ax.bar(x - width, aucs, width, label="AUC-ROC", color="#3498db")
    bars2 = ax.bar(x, accs, width, label="Accuracy", color="#2ecc71")
    bars3 = ax.bar(x + width, f1s, width, label="F1-macro (opt. threshold)", color="#9b59b6")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Performance Metrics by Experiment", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(descriptions, fontsize=9, ha="center")
    ax.legend(loc="upper right")
    ax.set_ylim(0.4, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "results_metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'results_metrics_summary.png'}")


def print_summary_table():
    """Print a markdown summary table."""
    print("\n" + "=" * 80)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    print("\n| Mode | Description | Samples | Balance | AUC | Acc | F1-macro |")
    print("|------|-------------|---------|---------|-----|-----|----------|")
    for mode, r in RESULTS.items():
        print(f"| `{mode}` | {r['description']} | {r['samples']} | {r['balance']} | "
              f"**{r['auc']:.3f}** | {r['accuracy']:.3f} | {r['f1_macro_opt']:.3f} |")

    print("\n### Key Findings:")
    print("1. **Strong signal for distant classes**: Dogs 0 vs 4 → AUC 0.80, Dogs 2 vs 4 → AUC 0.79")
    print("2. **Cats slightly harder to predict**: Cats 2 vs 4 → AUC 0.74 (vs Dogs 0.79)")
    print("3. **Adjacent classes near-random**: Dogs 2 vs 3 → AUC 0.58 (barely above random)")
    print("4. **Tabular features carry meaningful signal** for distinguishing adoption speed extremes")
    print("5. **Breed PCA**: 241 dog breeds → 15 components (79% variance), 66 cat breeds → 15 (92%)")
    print()


def run():
    """Generate all result plots and summary."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating result visualizations...")
    plot_auc_comparison()
    plot_class_distance_vs_auc()
    plot_metrics_summary()
    print_summary_table()
    print("Done! Plots saved to documentation/")


if __name__ == "__main__":
    run()