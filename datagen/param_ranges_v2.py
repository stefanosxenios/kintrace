"""
Parameter ranges for the 11 kinetic models (21-31).
Used with Latin Hypercube Sampling in generate_v2.py.

Inhibition constant notes
--------------------------
KSI1  (100, 500):  S2 cross-inhibits mu1 via (S2^2)/KSI1 in denominator
KPI1  (50, 100):   xylitol inhibits mu1 via (1 + P/KPI1) divisor
KSI2  (1, 50):     S2 self-inhibits mu2 (Haldane) via (S2^2)/KSI2 in denominator
KPI2  (1, 25):     xylitol inhibits mu2 via (1 + P/KPI2) divisor

mumax2 ranges
-------------
Models 21-27 (no mu2 inhibition): mumax2 capped at 4.0 h^-1 to prevent
  unrealistic xylose consumption — no KSI2/KPI2 terms to constrain the rate.
Models 28-31 (mu2 inhibited): mumax2 up to 8.0 h^-1 is safe — KSI2 and/or
  KPI2 pull the effective rate back to physiological levels.

mumax1 / YXS1
--------------
Both raised (mumax1 to 2.5, YXS1 to 1.5) to cover realistic biomass
trajectories even when inhibition terms are active on the glucose branch.
"""

# Base: conservative mumax2 — used for models with no mu2 inhibition (21-27)
_BASE = {
    "mumax1": (0.05, 0.8),    # h^-1
    "mumax2": (1.0, 6.0),    # h^-1  — capped: no KSI2/KPI2 to constrain
    "YXS1":   (0.1, 1.2),   # g biomass / g glucose
    "YPS2":   (0.5, 2.0),   # g xylitol / g xylose
    "KS1":    (0.05, 0.2),    # g/L
    "KP":     (40.0, 120.0), # g/L
}

# Extended mumax2 — used for models with mu2 inhibition terms (28-31)
_BASE_MU2_INH = {**_BASE, "mumax2": (1.0, 8.0)}

PARAM_RANGES = {
    # --- No inhibition ---ßß
    21: {**_BASE},

    # --- mu1-inhibited models (mu2 uninhibited → conservative mumax2) ---
    22: {**_BASE, "KPI1": (10.0, 50.0)},                             # PI on mu1
    23: {**_BASE, "KSI1": (20.0, 100.0)},                             # S2 cross-inhibits mu1
    24: {**_BASE, "KSI1": (20.0, 100.0),  "KPI1": (10.0, 50.0)},    # cross-SI + PI on mu1
    25: {**_BASE, "kd":   (0.005, 0.1),   "KPI1": (10.0, 50.0)},    # kd + PI on mu1
    26: {**_BASE, "kd":   (0.005, 0.1),   "KSI1": (20.0, 100.0)},    # kd + cross-SI on mu1
    27: {**_BASE, "kd":   (0.005, 0.1),   "KSI1": (20.0, 100.0),
                                          "KPI1": (10.0, 50.0)},     # kd + cross-SI + PI on mu1

    # --- mu2-inhibited models (KSI2/KPI2 constrain rate → wider mumax2) ---
    28: {**_BASE_MU2_INH, "KPI2": (2.0,  20.0)},                             # PI on mu2
    29: {**_BASE_MU2_INH, "KSI2": (2.0,  50.0)},                             # Haldane on S2
    30: {**_BASE_MU2_INH, "KSI2": (2.0,  50.0), "KPI2": (2.0,  20.0)},     # SI + PI on mu2
    31: {**_BASE_MU2_INH, "kd":   (0.005, 0.1), "KSI2": (2.0,  50.0),
                                                 "KPI2": (2.0,  50.0)},      # kd + SI + PI on mu2
}


def get_param_names(model_id: int) -> list:
    """Return ordered list of parameter names for a given model."""
    if model_id not in PARAM_RANGES:
        raise ValueError(f"Model {model_id} not in PARAM_RANGES. "
                         f"Available: {sorted(PARAM_RANGES.keys())}")
    return list(PARAM_RANGES[model_id].keys())


def get_param_bounds(model_id: int):
    """Return (lower_bounds, upper_bounds) lists for a given model."""
    ranges = PARAM_RANGES[model_id]
    lbs = [v[0] for v in ranges.values()]
    ubs = [v[1] for v in ranges.values()]
    return lbs, ubs
