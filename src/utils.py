import numpy as np
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def print_qwk(fold: int, score: float) -> None:
    print(f"Fold {fold}  QWK: {score:.4f}")


def print_cv_summary(scores: list[float]) -> None:
    arr = np.array(scores)
    print(f"CV QWK: {arr.mean():.4f} +/- {arr.std():.4f}")