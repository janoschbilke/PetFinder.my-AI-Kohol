from pathlib import Path

import kagglehub

COMPETITION = "petfinder-adoption-prediction"

def get_data_root() -> Path:
    path = kagglehub.competition_download(COMPETITION)
    data_root = Path(path)
    print(f"Data root: {data_root}")
    return data_root


def run(force: bool = False) -> Path:
    return get_data_root()


if __name__ == "__main__":
    root = get_data_root()