"""
LHS parameter ranges for the single-substrate "simple benchmark" mechanisms
(see kinetic_models/simple_models.py).

Design notes on identifiability / regime excitation
---------------------------------------------------
The four mechanisms are only distinguishable in specific regimes:

  * substrate_inhibition  shows itself only at HIGH S (the S^2/KI term must
    be non-negligible relative to Ks + S).  KI is therefore kept small enough
    that inhibition bites within the S range the feed sampler actually visits.
  * biomass_inhibition    shows itself only as X approaches Xmax, so Xmax is
    kept within reach of the biomass the feed sampler can produce.
  * contois vs monod      diverge when X is large relative to S (the Kc*X term
    competes with S in the denominator), so Kc is not made vanishingly small.

`Sin` (feed substrate concentration) is a fixed operating condition, not a
sampled kinetic parameter — see SIN_DEFAULT in feed_profiles_simple.py.

These ranges are used for data generation and for the Stage-2 regressor's
sigmoid output bounds. The *parameter-estimation* step (least-squares refinement
from the LSTM guess, and classic fitting) deliberately uses WIDER bounds so it
can reach out-of-distribution values — see ``fitting.generous_bounds``.
"""

# id -> {param_name: (low, high)}   order must match get_param_names()
PARAM_RANGES_SIMPLE = {
    # 1  monod
    1: {
        "mumax": (0.30, 0.90),   # h^-1
        "Ks":    (0.10, 5.00),   # g/L  (wide so the Monod knee is visible)
        "Yxs":   (0.30, 0.60),   # g biomass / g substrate
    },
    # 2  contois
    2: {
        "mumax": (0.30, 0.90),
        "Kc":    (0.05, 1.00),   # dimensionless (Kc*X has units of S)
        "Yxs":   (0.30, 0.60),
    },
    # 3  substrate (Haldane) inhibition
    3: {
        "mumax": (0.30, 0.90),
        "Ks":    (0.10, 5.00),
        "KI":    (0.50, 8.00),   # g/L  (smaller => stronger inhibition)
        "Yxs":   (0.30, 0.60),
    },
    # 4  biomass inhibition (logistic-type)
    4: {
        "mumax": (0.30, 0.90),
        "Ks":    (0.10, 5.00),
        "Xmax":  (12.0, 40.0),   # g/L  (must be reachable given the feed)
        "Yxs":   (0.30, 0.60),
    },
}


def get_param_bounds(model_id: int):
    """Return (lows, highs) lists in canonical parameter order."""
    from kinetic_models.simple_models import get_param_names
    names = get_param_names(model_id)
    ranges = PARAM_RANGES_SIMPLE[model_id]
    lows = [ranges[n][0] for n in names]
    highs = [ranges[n][1] for n in names]
    return lows, highs
