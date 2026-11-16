"""Evaluation helpers: Gini, PR-AUC, bootstrap CIs, lift tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)


def gini_normalized(y_true, y_score) -> float:
    """Normalised Gini = 2*AUC - 1 (the Kaggle competition metric)."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def evaluate_predictions(y_true, proba) -> dict:
    """All probability-based metrics used in the comparison tables."""
    proba = np.asarray(proba)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "gini": gini_normalized(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y_true, proba),
    }


def bootstrap_auc_ci(y_true, proba, n_boot: int = 200, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI for ROC-AUC. Returns (low, high)."""
    y, p = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.quantile(aucs, [alpha / 2, 1 - alpha / 2]))


def paired_bootstrap_diff(y_true, p_new, p_old, n_boot: int = 200, seed: int = 42):
    """Bootstrap the AUC difference (new - old) on the same resamples."""
    y, a, b = np.asarray(y_true), np.asarray(p_new), np.asarray(p_old)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), tuple(np.quantile(diffs, [0.025, 0.975])), float((diffs <= 0).mean())


def lift_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Decile table, bin 1 = highest predicted risk."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["bin"] = pd.qcut(df["p"].rank(method="first", ascending=False), n_bins, labels=False) + 1
    base = df["y"].mean()
    t = df.groupby("bin").agg(n=("y", "size"), claims=("y", "sum"), mean_pred=("p", "mean")).reset_index()
    t["claim_rate"] = t["claims"] / t["n"]
    t["lift"] = t["claim_rate"] / base
    t["cum_capture"] = t["claims"].cumsum() / t["claims"].sum()
    return t


def confusion_at_threshold(y_true, proba, threshold: float) -> dict:
    """Confusion counts and precision/recall/F1 at a probability threshold."""
    y, pred = np.asarray(y_true), (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "flagged_pct": 100 * pred.mean()}
