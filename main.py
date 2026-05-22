import argparse

from src import (
    data,
    data_analysis,
    preprocessing,
)


def main(force: bool = False) -> None:
    print("Step 0: Download Data")
    data.run(force=force)

    print("\nStep 1: Data Analysis")
    data_analysis.run(force=force)

    print("\nStep 2: Preprocessing")
    preprocessing.run(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetFinder adoption speed prediction pipeline")
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute all steps")
    args = parser.parse_args()
    main(force=args.force)