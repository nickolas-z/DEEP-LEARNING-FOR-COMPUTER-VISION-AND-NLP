# -*- coding: utf-8 -*-
"""
PetFinder.my Adoption Speed – Prize‑level Multimodal Solution
===========================================================

A single Python file that can be run locally or as a Kaggle notebook. It
covers the full pipeline:

1.  Load metadata CSV, image folder & split folds
2.  Generate/cached multimodal embeddings (Sentence‑BERT for description,
    CLIP ViT‑B/32 for photos)
3.  Concatenate tabular + reduced multimodal features
4.  Hyper‑parameter optimisation with Optuna on LightGBMRegressor using
    StratifiedGroupKFold and quadratic weighted kappa (QWK)
5.  Train final model on all data with best params, search optimal
    rounding thresholds, produce submission CSV

Assumptions:
------------
* train.csv, test.csv, train_images/, test_images/ follow the official
  PetFinder format.
* Execution environment has GPU (CUDA) for CLIP & Sentence‑BERT.
* pip install -q pandas numpy pillow scikit-learn sentence-transformers
  git+https://github.com/openai/CLIP.git lightgbm optuna joblib umap-learn

Run:
-----
$ python petfinder_prize_solution.py --data_dir /kaggle/input/petfinder-pawpularity-score
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import umap
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------- CONFIG -------------------------------------------------
N_JOBS = os.cpu_count()
IMG_EMBED_CACHE = "clip_img_embeds.pkl"
TXT_EMBED_CACHE = "txt_embeds.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)

# --------------------------- DATA ---------------------------------------------------


def read_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


# --------------------------- TEXT EMBEDDINGS ---------------------------------------


def get_text_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)


def encode_text(descriptions: List[str], model, batch_size: int = 64):
    return model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# --------------------------- IMAGE EMBEDDINGS --------------------------------------


def get_clip():
    import clip  # lazy import

    return clip.load("ViT-B/32", device=DEVICE)


def embed_one_pet(pet_id: str, img_dir: Path, clip_model, preprocess):
    imgs = list((img_dir).glob(f"{pet_id}-*.jpg"))[:3]
    vecs = []
    for p in imgs:
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vec = clip_model.encode_image(img)
        vecs.append(vec.cpu().numpy())
    if len(vecs) == 0:
        return np.zeros(512)
    return np.mean(np.vstack(vecs), axis=0)


def encode_images(ids: List[str], img_dir: Path):
    clip_model, preprocess = get_clip()
    func = delayed(embed_one_pet)
    embeds = Parallel(n_jobs=N_JOBS, backend="threading")(
        func(pid, img_dir, clip_model, preprocess) for pid in ids
    )
    return np.vstack(embeds)


# --------------------------- FEATURE ENGINEERING -----------------------------------


def build_features(df: pd.DataFrame, txt_embeds: np.ndarray, img_embeds: np.ndarray):
    tab_cols = [
        c for c in df.columns if c not in {"PetID", "Description", "AdoptionSpeed"}
    ]
    tab_data = df[tab_cols]

    # one‑hot for low‑cardinality categoricals
    cat_cols = [
        c
        for c in tab_data.columns
        if tab_data[c].dtype == "object" or tab_data[c].nunique() < 20
    ]
    num_cols = [c for c in tab_cols if c not in cat_cols]

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cats = enc.fit_transform(tab_data[cat_cols])

    scaler = StandardScaler()
    nums = scaler.fit_transform(tab_data[num_cols])

    # dimensionality reduction
    pca_txt = PCA(n_components=128, random_state=SEED).fit_transform(txt_embeds)
    umap_img = umap.UMAP(n_components=64, random_state=SEED).fit_transform(img_embeds)

    X = np.hstack([nums, cats, pca_txt, umap_img])
    return X


# --------------------------- METRIC & CV -------------------------------------------


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


qwk_scorer = make_scorer(qwk, greater_is_better=True)

# --------------------------- OPTUNA OPTIMISATION -----------------------------------


def objective(trial, X, y, folds):
    params = {
        "n_estimators": 4000,
        "learning_rate": trial.suggest_loguniform("lr", 1e-3, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 31, 1023),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample", 0.6, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "random_state": SEED,
        "objective": "regression",
        "metric": "rmse",
        "n_jobs": -1,
    }
    model = LGBMRegressor(**params)
    preds = cross_val_predict(model, X, y, cv=folds, n_jobs=-1, verbose=0)
    score = qwk(y, np.round(preds).astype(int))
    return score


# --------------------------- THRESHOLD OPTIMISER -----------------------------------


def optimise_rounding(preds: np.ndarray, y_true: np.ndarray):
    best_score, best_cut = -1, None
    for t1 in np.arange(0.5, 1.5, 0.05):
        for t2 in np.arange(1.5, 2.5, 0.05):
            for t3 in np.arange(2.5, 3.5, 0.05):
                cuts = [t1, t2, t3]
                y_pred = np.digitize(preds, cuts)
                s = qwk(y_true, y_pred)
                if s > best_score:
                    best_score, best_cut = s, cuts
    return best_cut


# --------------------------- MAIN ---------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_csv", default="submission.csv")
    args = parser.parse_args()

    train, test = read_data(args.data_dir)

    # ---------------------- TEXT ----------------------------------------------------
    if Path(TXT_EMBED_CACHE).exists():
        txt_train = joblib.load(TXT_EMBED_CACHE)
    else:
        txt_model = get_text_model()
        txt_train = encode_text(train["Description"].fillna("").tolist(), txt_model)
        joblib.dump(txt_train, TXT_EMBED_CACHE)

    # test embeddings (lazily, may reuse same cache key)
    txt_test = encode_text(test["Description"].fillna("").tolist(), get_text_model())

    # ---------------------- IMAGES --------------------------------------------------
    img_dir_train = args.data_dir / "train_images"
    img_dir_test = args.data_dir / "test_images"

    if Path(IMG_EMBED_CACHE).exists():
        img_train = joblib.load(IMG_EMBED_CACHE)
    else:
        img_train = encode_images(train.PetID.tolist(), img_dir_train)
        joblib.dump(img_train, IMG_EMBED_CACHE)

    img_test = encode_images(test.PetID.tolist(), img_dir_test)

    # ---------------------- FEATURES ----------------------------------------------
    X_train = build_features(train, txt_train, img_train)
    X_test = build_features(test, txt_test, img_test)
    y = train["AdoptionSpeed"].values

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        X_train, y, groups=train["RescuerID"]
    )

    # ---------------------- OPTUNA -------------------------------------------------
    study = optuna.create_study(direction="maximize", study_name="lgb_qwk")
    study.optimize(lambda t: objective(t, X_train, y, folds), n_trials=40, timeout=7200)
    best_params = study.best_trial.params

    # ---------------------- FINAL MODEL -------------------------------------------
    best_params.update(
        {
            "n_estimators": 4000,
            "objective": "regression",
            "metric": "rmse",
            "random_state": SEED,
            "n_jobs": -1,
        }
    )
    model = LGBMRegressor(**best_params)
    model.fit(X_train, y)

    preds_val = model.predict(X_train)
    cuts = optimise_rounding(preds_val, y)
    print("Optimal cuts", cuts)

    preds_test = model.predict(X_test)
    y_test_round = np.digitize(preds_test, cuts)

    sub = pd.DataFrame({"PetID": test.PetID, "AdoptionSpeed": y_test_round.astype(int)})
    sub.to_csv(args.out_csv, index=False)
    print("Submission saved to", args.out_csv)

    # Save artifacts for reproducibility
    with open("best_params.json", "w") as fp:
        json.dump(best_params, fp, indent=2)


if __name__ == "__main__":
    main()
