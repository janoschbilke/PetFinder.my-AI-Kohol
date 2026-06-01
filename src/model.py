import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from src.utils import print_cv_summary, print_qwk, quadratic_weighted_kappa

from src.data import get_data_root

CACHE_DIR = Path("cache")
N_FOLDS = 5
RANDOM_STATE = 42

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_logloss",
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "n_estimators": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    print("  Loading tabular features...")
    train_feat = pd.read_parquet(CACHE_DIR / "train_features.parquet")
    test_feat = pd.read_parquet(CACHE_DIR / "test_features.parquet")

    print("  Loading CNN embeddings...")
    train_emb = np.load(CACHE_DIR / "train_embeddings.npy")
    test_emb = np.load(CACHE_DIR / "test_embeddings.npy")

    train_emb_ids = np.load(CACHE_DIR / "train_pet_ids.npy", allow_pickle=True).tolist()
    test_emb_ids = np.load(CACHE_DIR / "test_pet_ids.npy", allow_pickle=True).tolist()

    y = train_feat["AdoptionSpeed"].values.astype(int)
    test_pet_ids = test_feat["PetID"].tolist()

    tab_cols = [c for c in train_feat.columns if c != "AdoptionSpeed"]
    tab_cols_no_pid = [c for c in tab_cols if c != "PetID"]
    X_tab = train_feat[tab_cols_no_pid].values.astype(np.float32)
    X_tab_test = test_feat[tab_cols_no_pid].values.astype(np.float32)

    print("  Aligning embeddings to tabular row order...")
    data_root = get_data_root()
    train_id_to_idx = {pid: i for i, pid in enumerate(train_emb_ids)}
    test_id_to_idx = {pid: i for i, pid in enumerate(test_emb_ids)}

    train_tab_pet_ids = pd.read_csv(data_root / "train" / "train.csv")["PetID"].tolist()
    train_emb_aligned = np.zeros((len(X_tab), train_emb.shape[1]), dtype=np.float32)
    for i, pid in enumerate(tqdm(train_tab_pet_ids, desc="  Aligning train embeddings", leave=False)):
        if pid in train_id_to_idx:
            train_emb_aligned[i] = train_emb[train_id_to_idx[pid]]

    test_tab_pet_ids = pd.read_csv(data_root / "test" / "test.csv")["PetID"].tolist()
    test_emb_aligned = np.zeros((len(X_tab_test), test_emb.shape[1]), dtype=np.float32)
    for i, pid in enumerate(tqdm(test_tab_pet_ids, desc="  Aligning test embeddings", leave=False)):
        if pid in test_id_to_idx:
            test_emb_aligned[i] = test_emb[test_id_to_idx[pid]]

    X_train = np.concatenate([X_tab, train_emb_aligned], axis=1)
    X_test = np.concatenate([X_tab_test, test_emb_aligned], axis=1)

    emb_col_names = [f"embed_{i}" for i in range(train_emb.shape[1])]
    all_col_names = tab_cols_no_pid + emb_col_names

    return X_train, y, X_test, test_pet_ids, all_col_names


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y), dtype=int)
    qwk_scores = []
    models = []

    fold_iter = tqdm(enumerate(skf.split(X, y), start=1), total=N_FOLDS, desc="CV folds")
    for fold, (train_idx, val_idx) in fold_iter:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        sample_weights = compute_sample_weight("balanced", y_tr)

        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        val_preds = clf.predict(X_val)
        oof_preds[val_idx] = val_preds
        score = quadratic_weighted_kappa(y_val, val_preds)
        qwk_scores.append(score)
        fold_iter.set_postfix({"QWK": f"{score:.4f}"})
        print_qwk(fold, score)
        models.append(clf)

    print_cv_summary(qwk_scores)
    best_model = models[int(np.argmax(qwk_scores))]
    return best_model, oof_preds


def run(force: bool = False) -> None:
    submission_path = Path("submission.csv")
    model_path = CACHE_DIR / "model.pkl"

    if not force and submission_path.exists() and model_path.exists():
        print("Model and submission already exist. Use --force to retrain.")
        return

    print("Loading cached features and embeddings...")
    X_train, y_train, X_test, test_pet_ids, col_names = load_data()
    print(f"Feature matrix: train={X_train.shape}  test={X_test.shape}")

    print(f"\nTraining LightGBM with {N_FOLDS}-fold CV...")
    best_model, oof_preds = train_and_evaluate(X_train, y_train)

    print("\nSaving model and OOF predictions...")
    CACHE_DIR.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    np.save(CACHE_DIR / "oof_predictions.npy", oof_preds)
    np.save(CACHE_DIR / "oof_labels.npy", y_train)
    np.save(CACHE_DIR / "col_names.npy", np.array(col_names, dtype=object))
    print(f"Saved: {model_path}")

    print("\nGenerating test predictions...")
    test_preds = best_model.predict(X_test)

    submission = pd.DataFrame({"PetID": test_pet_ids, "AdoptionSpeed": test_preds.astype(int)})
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)