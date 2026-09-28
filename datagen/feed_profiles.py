"""
Random feed profile generation for in silico data generation.

Ranges calibrated against all available experiments (BC12, BC15, BC18, BC19,
VR5, VR6, VR7) rather than BC12 alone.

Key design choices
------------------
S2_0 in (0, 75): covers both experimental regimes —
  - BC12: xylose loaded in batch (S2_0 ≈ 63 g/L)
  - BC15, BC18, BC19, VR5-7: xylose only in feed (S2_0 ≈ 0)

S1_feed in (70, 115): BC15 uses 108.76 g/L (was OOD at old ceiling of 100).
S2_feed in (220, 340): BC15/BC18/VR5-7 use 309–334 g/L (was OOD at 300).
t_total in (85, 175):  BC15 ends at 97.6 h (was OOD below old floor of 120).
X0 in (0.5, 4.5):      VR5-7 start at 3.1–4.0 g/L (was OOD above old ceil of 1.5).

Two profile types are generated with equal probability:
  1. Constant rate: single feed rate throughout
  2. Step change:   rate changes at a random time mid-experiment

The synthetic DataFrames produced are compatible with
kinetic_models.feed_fun.get_feed_values() and create_feed_rate_function().
"""

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

# --- Parameter ranges (calibrated to full experimental dataset) ---

FEED_RANGES = {
    "t_start":        (20.0,  45.0),   # h: when feed begins
    "F_initial":      (1.0,   3.5),    # mL/h: initial feed rate
    "F_step":         (0.8,   4.5),    # mL/h: feed rate after step change
    "t_step_offset":  (20.0,  65.0),   # h after t_start: when step change occurs
    "S1_feed":        (70.0,  115.0),  # g/L: glucose in feed  [BC15=108.76 → extended from 100]
    "S2_feed":        (220.0, 340.0),  # g/L: xylose in feed   [BC15=333.96 → extended from 300]
    "t_total":        (85.0,  175.0),  # h: run duration       [BC15=97.6h  → lowered from 120]
}

IC_RANGES = {
    "X0":    (0.5,   4.5),    # g/L: initial biomass  [VR5-7=3.1-4.0 → extended from 1.5]
    "S1_0":  (15.0,  25.0),   # g/L: initial glucose
    "S2_0":  (0.0,   75.0),   # g/L: initial xylose   [BC15/18/19/VR=0 → lowered from 50]
    "V0_mL": (250.0, 300.0),  # mL: initial reactor volume
}
# P0 is always 0 (no product at start)


def sample_feed_and_ic(n: int, seed: int = None) -> dict:
    """
    Sample n sets of (feed profile params, initial conditions) using LHS.

    Parameters
    ----------
    n    : number of samples
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys from FEED_RANGES + IC_RANGES, each value is (n,) array.
    Additional keys:
      "P0"      : zeros array
      "has_step": boolean array, True → step change in feed rate
    """
    all_ranges = {**FEED_RANGES, **IC_RANGES}
    keys = list(all_ranges.keys())
    dim = len(keys)

    sampler = LatinHypercube(d=dim, seed=seed)
    unit = sampler.random(n)          # (n, dim) in [0, 1]

    lbs = np.array([all_ranges[k][0] for k in keys])
    ubs = np.array([all_ranges[k][1] for k in keys])
    samples = lbs + unit * (ubs - lbs)

    result = {keys[i]: samples[:, i] for i in range(dim)}
    result["P0"] = np.zeros(n)

    rng = np.random.default_rng(seed)
    result["has_step"] = rng.integers(0, 2, size=n).astype(bool)

    return result


def make_exter_df(fp: dict, idx: int, label: str = "SYNTH") -> pd.DataFrame:
    """
    Build a synthetic experiment DataFrame for one simulation, compatible
    with get_feed_values() and create_feed_rate_function().

    Parameters
    ----------
    fp    : single-simulation dict (scalar values), from sample_feed_and_ic
    idx   : simulation index, used to create a unique experiment label
    label : base label prefix

    Returns
    -------
    pd.DataFrame with columns:
      Experiment Label, Sample name, Time (h), Feed rate (mL/h),
      Feed added (mL), Glucose, Xylose
    """
    t_start  = float(fp["t_start"])
    F_init   = float(fp["F_initial"])
    F_step   = float(fp["F_step"])
    t_step   = t_start + float(fp["t_step_offset"])
    has_step = bool(fp["has_step"])
    S1_feed  = float(fp["S1_feed"])
    S2_feed  = float(fp["S2_feed"])
    t_total  = float(fp["t_total"])

    exp_label = f"{label}{idx}"
    rows = []

    if (not has_step) or (t_step >= t_total):
        # --- Constant rate feed ---
        # Single anchor row: 1 h after feed start.
        # create_feed_rate_function infers t_feed_start = anchor_t - feed_added/rate
        #   = (t_start + 1) - F_init/F_init = t_start  ✓
        rows.append({
            "Experiment Label": exp_label,
            "Sample name":      f"{exp_label}_0",
            "Time (h)":         t_start + 1.0,
            "Feed rate (mL/h)": F_init,
            "Feed added (mL)":  F_init * 1.0,
            "Glucose":          np.nan,
            "Xylose":           np.nan,
        })
    else:
        # --- Step change at t_step ---
        # Row 0: anchor at t_step with rate F_init up to that point.
        #   t_feed_start = t_step - F_init*(t_step-t_start)/F_init = t_start  ✓
        fa_at_step = F_init * (t_step - t_start)
        rows.append({
            "Experiment Label": exp_label,
            "Sample name":      f"{exp_label}_0",
            "Time (h)":         t_step,
            "Feed rate (mL/h)": F_init,
            "Feed added (mL)":  fa_at_step,
            "Glucose":          np.nan,
            "Xylose":           np.nan,
        })
        # Row 1: 0.01 h after step — rate switches to F_step.
        # Feed added unchanged (instantaneous rate change).
        rows.append({
            "Experiment Label": exp_label,
            "Sample name":      f"{exp_label}_1",
            "Time (h)":         t_step + 0.01,
            "Feed rate (mL/h)": F_step,
            "Feed added (mL)":  fa_at_step,
            "Glucose":          np.nan,
            "Xylose":           np.nan,
        })

    # Feed concentration row (no time info — identified by Sample name)
    rows.append({
        "Experiment Label": exp_label,
        "Sample name":      f"{exp_label}_Feed",
        "Time (h)":         np.nan,
        "Feed rate (mL/h)": np.nan,
        "Feed added (mL)":  np.nan,
        "Glucose":          S1_feed,
        "Xylose":           S2_feed,
    })

    return pd.DataFrame(rows)
