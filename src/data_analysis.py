import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_ROOT = Path("~/.cache/kagglehub/competitions/petfinder-adoption-prediction").expanduser()
TRAIN_CSV = DATA_ROOT / "train" / "train.csv"
TRAIN_IMAGES_DIR = DATA_ROOT / "train_images"

ADOPTION_SPEED_LABELS = {
    0: "Same day",
    1: "1-7 days",
    2: "8-30 days",
    3: "31-90 days",
    4: ">100 days",
}

TYPE_LABELS = {1: "Dog", 2: "Cat"}
GENDER_LABELS = {1: "Male", 2: "Female", 3: "Mixed"}
HEALTH_LABELS = {0: "Not specified", 1: "Healthy", 2: "Minor injury", 3: "Serious injury"}
VACCINATED_LABELS = {1: "Yes", 2: "No", 3: "Not sure"}

def run(force: bool = False) -> None:
    df = pd.read_csv(TRAIN_CSV)
    n = len(df)
    print(f"Total training samples: {n}")

    print(f"  Adoption Speed Distribution \n")
    speed_counts = df["AdoptionSpeed"].value_counts().sort_index()
    for cls, count in speed_counts.items():
        label = ADOPTION_SPEED_LABELS[cls]
        pct = count / n * 100
        print(f"  Class {cls} ({label:>12}): {count:5d}  ({pct:5.1f}%)")

    print(f"  Animal Type\n")
    type_counts = df["Type"].value_counts().sort_index()
    for t, count in type_counts.items():
        label = TYPE_LABELS.get(t, str(t))
        print(f"  {label:>6}: {count:5d}  ({count/n*100:.1f}%)")

    print(f"  Gender Distribution\n")
    gender_counts = df["Gender"].value_counts().sort_index()
    for g, count in gender_counts.items():
        label = GENDER_LABELS.get(g, str(g))
        print(f"  {label:>8}: {count:5d}  ({count/n*100:.1f}%)")

    print(f"  Health Condition\n")
    health_counts = df["Health"].value_counts().sort_index()
    for h, count in health_counts.items():
        label = HEALTH_LABELS.get(h, str(h))
        print(f"  {label:>16}: {count:5d}  ({count/n*100:.1f}%)")

    print(f"  Vaccination Status\n")
    vacc_counts = df["Vaccinated"].value_counts().sort_index()
    for v, count in vacc_counts.items():
        label = VACCINATED_LABELS.get(v, str(v))
        print(f"  {label:>10}: {count:5d}  ({count/n*100:.1f}%)")

    print(f"  Age Statistics (months)\n")
    print(df["Age"].describe().to_string())

    print(f"  Adoption Fee Statistics \n")
    free = (df["Fee"] == 0).sum()
    print(f"  Free (Fee=0): {free:5d}  ({free/n*100:.1f}%)")
    print(f"  Paid (Fee>0): {n-free:5d}  ({(n-free)/n*100:.1f}%)")
    print(f"  Fee range: {df['Fee'].min()} – {df['Fee'].max()}")
    print(f"  Mean fee (paid only): {df.loc[df['Fee']>0,'Fee'].mean():.1f}")

    print(f"  Photo & Video Counts\n")
    print(f"  Photos — mean: {df['PhotoAmt'].mean():.1f}  max: {df['PhotoAmt'].max()}")
    print(f"  Videos — mean: {df['VideoAmt'].mean():.1f}  max: {df['VideoAmt'].max()}")
    print(f"  No photos: {(df['PhotoAmt']==0).sum()}")

    print(f"  Adoption Speed by Animal Type\n")
    cross = pd.crosstab(df["Type"], df["AdoptionSpeed"])
    cross.index = [TYPE_LABELS.get(i, str(i)) for i in cross.index]
    cross.columns = [ADOPTION_SPEED_LABELS[c] for c in cross.columns]
    print(cross.to_string())

    print(f"  Generating Plots\n")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("PetFinder Dataset — Exploratory Analysis", fontsize=14)

    speed_series = speed_counts.rename(index=ADOPTION_SPEED_LABELS)
    axes[0, 0].bar(speed_series.index, speed_series.values, color="steelblue")
    axes[0, 0].set_title("Adoption Speed Distribution")
    axes[0, 0].set_xlabel("Class")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=20)

    type_series = type_counts.rename(index=TYPE_LABELS)
    axes[0, 1].bar(type_series.index, type_series.values, color=["#e07b54", "#5b8db8"])
    axes[0, 1].set_title("Animal Type")
    axes[0, 1].set_ylabel("Count")

    axes[0, 2].hist(df["Age"].clip(upper=60), bins=30, color="mediumseagreen", edgecolor="white")
    axes[0, 2].set_title("Age Distribution (months, clipped at 60)")
    axes[0, 2].set_xlabel("Age (months)")
    axes[0, 2].set_ylabel("Count")

    axes[1, 0].hist(df["PhotoAmt"], bins=20, color="mediumpurple", edgecolor="white")
    axes[1, 0].set_title("Number of Photos per Pet")
    axes[1, 0].set_xlabel("Photo count")
    axes[1, 0].set_ylabel("Count")

    fee_paid = df[df["Fee"] > 0]["Fee"].clip(upper=500)
    axes[1, 1].hist(fee_paid, bins=30, color="goldenrod", edgecolor="white")
    axes[1, 1].set_title("Adoption Fee Distribution (paid only, clipped at 500)")
    axes[1, 1].set_xlabel("Fee")
    axes[1, 1].set_ylabel("Count")

    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    cross_pct.plot(kind="bar", ax=axes[1, 2], colormap="tab10", edgecolor="white")
    axes[1, 2].set_title("Adoption Speed by Animal Type (%)")
    axes[1, 2].set_xlabel("Type")
    axes[1, 2].set_ylabel("Percentage")
    axes[1, 2].tick_params(axis="x", rotation=0)
    axes[1, 2].legend(title="Speed", fontsize=7)

    plt.tight_layout()
    plt.savefig("data_analysis.png", dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)