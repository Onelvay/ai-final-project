"""Data loading, dtype optimisation, missing-value handling and feature helpers."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

MISSING_CODE = -1      # Porto Seguro encodes "missing" as -1 in every column
TARGET = "target"
ID_COL = "id"

FAMILY_DESCRIPTIONS = {
    "ind": "Individual (driver) attributes",
    "reg": "Regional / area attributes",
    "car": "Vehicle attributes",
    "calc": "Calculated insurer features (anonymised)",
    "other": "Engineered features",
}


# ----------------------------------------------------------------------------
# Memory helpers
# ----------------------------------------------------------------------------
def mem_mb(df: pd.DataFrame) -> float:
    """Deep memory usage of a DataFrame in MB."""
    return float(df.memory_usage(deep=True).sum() / 1024 ** 2)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integers to the smallest int type and floats to float32."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[col] = s.astype("float32")
    return out


def load_csv_chunked(path, chunksize: int = 100_000, verbose: bool = True) -> pd.DataFrame:
    """Read a large CSV in chunks, optimising dtypes chunk by chunk.

    Peak memory stays close to (compact result + one raw chunk) instead of
    (raw full frame). Raises a readable error if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    chunks, t0 = [], time.perf_counter()
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        chunks.append(optimize_dtypes(chunk))
        if verbose:
            print(f"  chunk {i + 1}: {len(chunk):>7,} rows", end="\r")
    df = optimize_dtypes(pd.concat(chunks, ignore_index=True))  # unify dtypes across chunks
    if verbose:
        print(f"\nLoaded {len(df):,} rows x {df.shape[1]} cols in {time.perf_counter() - t0:.1f}s "
              f"({mem_mb(df):.1f} MB)")
    return df


# ----------------------------------------------------------------------------
# Column bookkeeping
# ----------------------------------------------------------------------------
def classify_columns(columns) -> dict[str, list[str]]:
    """Split Porto Seguro feature names into bin / cat / num lists by suffix."""
    groups = {"bin": [], "cat": [], "num": []}
    for c in columns:
        if c in (ID_COL, TARGET):
            continue
        if c.endswith("_bin"):
            groups["bin"].append(c)
        elif c.endswith("_cat"):
            groups["cat"].append(c)
        else:
            groups["num"].append(c)
    return groups


def feature_family(col: str) -> str:
    """'ps_car_13' -> 'car'; engineered columns -> 'other'."""
    parts = col.split("_")
    return parts[1] if col.startswith("ps_") and len(parts) > 2 else "other"


def calc_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("ps_calc_")]


# ----------------------------------------------------------------------------
# Missing values
# ----------------------------------------------------------------------------
def count_missing_codes(df: pd.DataFrame) -> pd.Series:
    """Number of -1 codes per feature (only features that have any)."""
    feats = [c for c in df.columns if c not in (ID_COL, TARGET)]
    counts = (df[feats] == MISSING_CODE).sum()
    return counts[counts > 0].sort_values(ascending=False)


def replace_missing_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace -1 with NaN (columns with missing values become float32)."""
    out, report = df.copy(), {}
    for col in [c for c in out.columns if c not in (ID_COL, TARGET)]:
        mask = out[col] == MISSING_CODE
        n = int(mask.sum())
        if n:
            out[col] = out[col].astype("float32").mask(mask)
            report[col] = n
    return out, report


# ----------------------------------------------------------------------------
# Feature engineering (row-wise, stateless -> cannot leak across the split)
# ----------------------------------------------------------------------------
ENGINEERED = ["n_missing", "ind_bin_sum", "car13_x_reg03", "reg_sum"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Needs NaN (not -1) as the missing marker."""
    out = df.copy()
    raw = [c for c in out.columns if c not in (ID_COL, TARGET) and c not in ENGINEERED]
    out["n_missing"] = out[raw].isna().sum(axis=1).astype("int8")
    ind_bin = [c for c in raw if c.startswith("ps_ind_") and c.endswith("_bin")]
    out["ind_bin_sum"] = out[ind_bin].sum(axis=1).astype("int8")
    out["car13_x_reg03"] = (out["ps_car_13"] * out["ps_reg_03"]).astype("float32")
    out["reg_sum"] = (out["ps_reg_01"] + out["ps_reg_02"] + out["ps_reg_03"]).astype("float32")
    return out


def select_feature_set(df: pd.DataFrame, name: str) -> list[str]:
    """Return the column list for a named feature set."""
    base = [c for c in df.columns if c not in (ID_COL, TARGET) and c not in ENGINEERED]
    if name == "raw_all":
        return base
    if name == "drop_calc":
        return [c for c in base if not c.startswith("ps_calc_")]
    if name == "drop_calc+engineered":
        return [c for c in base if not c.startswith("ps_calc_")] + [c for c in ENGINEERED if c in df.columns]
    raise ValueError(f"Unknown feature set: {name}")


# ----------------------------------------------------------------------------
# Univariate signal (used for the feature dictionary and EDA)
# ----------------------------------------------------------------------------
def feature_signal(s: pd.Series, y: pd.Series, max_groups: int = 100, n_folds: int = 5,
                   seed: int = 42) -> tuple[float, float]:
    """Univariate normalised Gini (out-of-fold) and chi-square p-value of one feature vs target.

    Each distinct value (or quantile bin if > max_groups values; the bins use only X, not y) is
    mapped to its target rate **estimated on the other folds**; the Gini is 2*AUC-1 of that score.
    Cross-fitting removes the in-sample optimism that would otherwise favour high-cardinality
    features, so pure-noise features land near 0.
    """
    from sklearn.model_selection import KFold
    x = s.astype("float64")
    if x.nunique() > max_groups:
        x = pd.Series(pd.qcut(x, max_groups, labels=False, duplicates="drop"), index=x.index)
    key = x.fillna(-999.0).reset_index(drop=True)
    yy = y.reset_index(drop=True)
    score = np.full(len(yy), yy.mean())
    for tr, va in KFold(n_folds, shuffle=True, random_state=seed).split(key):
        rate = yy.iloc[tr].groupby(key.iloc[tr]).mean()
        score[va] = key.iloc[va].map(rate).fillna(yy.iloc[tr].mean()).to_numpy()
    gini = 2 * roc_auc_score(yy, score) - 1
    table = pd.crosstab(key, yy)
    p = stats.chi2_contingency(table)[1] if table.shape[0] > 1 else 1.0
    return float(gini), float(p)


def build_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per raw feature: family, kind, missingness, cardinality, signal."""
    groups = classify_columns(df.columns)
    kind_of = {c: k for k, cols in groups.items() for c in cols}
    y = df[TARGET]
    rows = []
    for col in [c for c in df.columns if c not in (ID_COL, TARGET)]:
        gini, p = feature_signal(df[col], y)
        rows.append({
            "feature": col, "family": feature_family(col), "kind": kind_of[col],
            "n_missing": int(df[col].isna().sum()),
            "pct_missing": round(100 * df[col].isna().mean(), 3),
            "n_unique": int(df[col].nunique()),
            "univariate_gini": round(gini, 5), "chi2_pvalue": p,
        })
    return pd.DataFrame(rows)
