"""Pipelines, timing helpers and cross-validated hyper-parameter tuning."""
from __future__ import annotations

import os
import tempfile
import time

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_utils import classify_columns


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def build_preprocessor(kind: str, columns: list[str]) -> ColumnTransformer:
    """ColumnTransformer for one of three model families.

    kind = 'linear'      : impute + scale numerics, one-hot categoricals
    kind = 'tree_nan'    : pass-through, NaN handled natively (LightGBM/XGBoost/HistGB)
    kind = 'tree_impute' : constant -1 imputation (sklearn RandomForest)
    """
    g = classify_columns(columns)
    num, cat, binary = g["num"], g["cat"], g["bin"]
    if kind == "linear":
        ct = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                              ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=200))]), cat),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ])
    elif kind == "tree_nan":
        ct = ColumnTransformer([("all", "passthrough", num + cat + binary)],
                               verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
    elif kind == "tree_impute":
        ct = ColumnTransformer([("all", SimpleImputer(strategy="constant", fill_value=-1),
                                 num + cat + binary)], verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind}")
    return ct


def make_pipeline(model, kind: str, columns: list[str]) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(kind, columns)), ("model", model)])


# ----------------------------------------------------------------------------
# Timing / size
# ----------------------------------------------------------------------------
def fit_timed(pipe, X, y):
    t0 = time.perf_counter()
    pipe.fit(X, y)
    return pipe, time.perf_counter() - t0


def predict_timed(pipe, X, n_ref: int = 1000):
    """Return (proba, milliseconds per 1000 rows)."""
    t0 = time.perf_counter()
    proba = pipe.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - t0
    return proba, 1000 * elapsed * n_ref / len(X)


def model_size_mb(obj) -> float:
    """Serialised size of an object (joblib, compress=0) in MB."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.joblib")
        joblib.dump(obj, path, compress=0)
        return os.path.getsize(path) / 1024 ** 2


def make_cv_splits(X, y, n_splits: int = 3, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


# ----------------------------------------------------------------------------
# Cross-validated boosting (native cv with early stopping)
# ----------------------------------------------------------------------------
def lgbm_cv(params: dict, X, y, splits, max_rounds: int = 2000, patience: int = 100, seed: int = 42):
    """Return (mean_auc, std_auc, best_rounds) from lightgbm.cv."""
    import lightgbm as lgb
    p = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": seed, **params}
    res = lgb.cv(p, lgb.Dataset(X, label=y), num_boost_round=max_rounds, folds=splits,
                 callbacks=[lgb.early_stopping(patience, verbose=False)])
    mean_key = next(k for k in res if k.endswith("auc-mean"))
    std_key = next(k for k in res if k.endswith("auc-stdv"))
    best = int(np.argmax(res[mean_key]))
    return float(res[mean_key][best]), float(res[std_key][best]), best + 1


def xgb_cv(params: dict, X, y, splits, max_rounds: int = 2000, patience: int = 100, seed: int = 42):
    """Return (mean_auc, std_auc, best_rounds) from xgboost.cv."""
    import xgboost as xgb
    p = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
         "seed": seed, **params}
    res = xgb.cv(p, xgb.DMatrix(X, label=y), num_boost_round=max_rounds, folds=splits,
                 early_stopping_rounds=patience, verbose_eval=False)
    best = int(res["test-auc-mean"].values.argmax())
    return (float(res["test-auc-mean"].iloc[best]), float(res["test-auc-std"].iloc[best]), best + 1)


# ----------------------------------------------------------------------------
# Optuna studies
# ----------------------------------------------------------------------------
def lgbm_space(trial) -> dict:
    return {
        "learning_rate": 0.05,
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
    }


def xgb_space(trial) -> dict:
    return {
        "learning_rate": 0.05,
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }


def run_study(kind: str, X, y, splits, n_trials: int = 20, seed: int = 42):
    """Tune 'lgbm' or 'xgb'. Returns (best_params, best_rounds, study)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space, cv_fn = (lgbm_space, lgbm_cv) if kind == "lgbm" else (xgb_space, xgb_cv)

    def objective(trial):
        params = space(trial)
        mean, std, rounds = cv_fn(params, X, y, splits, seed=seed)
        trial.set_user_attr("rounds", rounds)
        trial.set_user_attr("auc_std", std)
        return mean

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = {**space_fixed(kind), **study.best_params}
    return best, int(study.best_trial.user_attrs["rounds"]), study


def space_fixed(kind: str) -> dict:
    """Parameters held fixed during tuning (not suggested by Optuna)."""
    if kind == "lgbm":
        return {"learning_rate": 0.05, "subsample_freq": 1}
    return {"learning_rate": 0.05}
