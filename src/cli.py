"""Interactive console menu: load the saved model bundle and score policyholders."""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_utils import add_features

MENU = """
==== Safe Driver - claim risk predictor ====
 1 - Predict: enter values for key features
 2 - Predict: random policyholder from the hold-out sample
 3 - Model card (metrics, size, speed)
 4 - Top features (permutation importance)
 5 - Exit
"""


def load_bundle(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}. Run Part 2/3 of the notebook first.")
    return joblib.load(path)


def parse_value(text: str) -> float:
    """'na' / empty -> NaN (missing); otherwise float. Raises ValueError on garbage."""
    text = text.strip().lower()
    if text in ("", "na", "nan", "none", "-1"):
        return math.nan
    return float(text)


def predict_proba(bundle: dict, overrides: dict | None = None, row: dict | None = None) -> float:
    """Score one policyholder. Unspecified raw features use training medians/modes."""
    values = dict(bundle["raw_defaults"]) if row is None else dict(row)
    values.update(overrides or {})
    frame = add_features(pd.DataFrame([values]).astype("float32"))
    return float(bundle["pipeline"].predict_proba(frame[bundle["columns"]])[0, 1])


def risk_band(p: float, cutoffs: dict) -> str:
    """Map a probability to a band using hold-out score quantiles."""
    if p >= cutoffs["p90"]:
        return "VERY HIGH (top 10% of risk)"
    if p >= cutoffs["p75"]:
        return "HIGH (top 10-25%)"
    if p >= cutoffs["p50"]:
        return "MEDIUM (25-50%)"
    return "LOW (bottom 50%)"


def _ask_value(name: str, default, input_fn, print_fn, attempts: int = 3):
    for _ in range(attempts):
        text = input_fn(f"  {name} [default {default:.4g}, 'na' = missing]: ")
        if text.strip() == "":
            return default
        try:
            return parse_value(text)
        except ValueError:
            print_fn("  ! Please enter a number or 'na'.")
    print_fn("  Too many invalid entries - using the default.")
    return default


def run_menu(bundle_path, input_fn=input, print_fn=print) -> None:
    """Menu loop. `input_fn` can be replaced by a scripted iterator for demos/tests."""
    try:
        bundle = load_bundle(bundle_path)
    except FileNotFoundError as err:
        print_fn(str(err))
        return

    def safe_input(prompt: str) -> str:
        try:
            return input_fn(prompt)
        except (EOFError, StopIteration, KeyboardInterrupt):
            return "5"

    rng = np.random.default_rng(0)
    while True:
        print_fn(MENU)
        choice = safe_input("Select an option (1-5): ").strip()
        if choice == "1":
            overrides = {}
            for feat in bundle["key_features"]:
                overrides[feat] = _ask_value(feat, bundle["raw_defaults"][feat], safe_input, print_fn)
            p = predict_proba(bundle, overrides)
            print_fn(f"\n  Predicted claim probability: {100 * p:.2f}%  ->  {risk_band(p, bundle['cutoffs'])}")
        elif choice == "2":
            sample = bundle["sample"]
            i = int(rng.integers(0, len(sample)))
            row = sample.iloc[i]
            p = predict_proba(bundle, row={k: row[k] for k in bundle["raw_defaults"]})
            print_fn(f"\n  Hold-out policyholder id={int(row['id'])}: predicted {100 * p:.2f}% -> "
                     f"{risk_band(p, bundle['cutoffs'])}; actual claim = {int(row['target'])}")
        elif choice == "3":
            print_fn("\n  Model card")
            for k, v in bundle["model_card"].items():
                print_fn(f"   {k:<24}{v:.4f}" if isinstance(v, float) else f"   {k:<24}{v}")
        elif choice == "4":
            print_fn("\n  Top features by permutation importance (drop in ROC-AUC):")
            for name, val in bundle["importance"][:10]:
                print_fn(f"   {name:<20}{val:.5f}")
        elif choice == "5":
            print_fn("Goodbye.")
            return
        else:
            print_fn("  ! Unknown option, choose 1-5.")
