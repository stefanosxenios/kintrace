"""
Feed profiles and initial conditions for the single-substrate benchmark.

The sampler is deliberately designed to *excite the discriminating regimes* of
the four mechanisms so that mechanism identification and parameter recovery are
a fair test rather than an impossible one:

  * a random initial substrate charge S0 spanning low->high concentrations, so
    that substrate-inhibition trajectories actually spend time in the inhibited
    (high-S) region;
  * a piecewise-constant feed with an optional initial batch phase (F=0), so the
    culture first consumes the charge (revealing the low-S growth shape) and then
    receives more substrate (pushing biomass up toward Xmax, revealing biomass
    inhibition, and keeping X large relative to S, revealing Contois behaviour).

Feed rate is expressed in L/h directly (this is a clean in-silico system; there
is no mL<->L bookkeeping like the dual-substrate real-data path).
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Known operating condition — feed substrate concentration (g/L).
SIN_DEFAULT = 120.0

# Feed / IC sampling ranges
_RANGES = {
    "X0":      (1.0, 5.0),     # g/L initial biomass
    "S0":      (1.0, 60.0),    # g/L initial substrate charge (wide: excites inhibition)
    "V0":      (1.0, 2.0),     # L  initial volume
    "t_total": (10.0, 30.0),   # h  experiment length
    "Sin":     (100.0, 150.0), # g/L feed substrate concentration (known condition)
}

N_FEED_SEG = 4          # piecewise-constant feed segments
F_MAX = 0.15            # L/h per-segment max feed rate
P_BATCH_START = 0.5     # probability the first segment is a batch phase (F=0)


class PiecewiseFeed:
    """Picklable piecewise-constant feed callable f(t) -> F [L/h]."""

    def __init__(self, breakpoints, rates):
        self.breakpoints = np.asarray(breakpoints, dtype=float)  # length n_seg+1
        self.rates = np.asarray(rates, dtype=float)              # length n_seg

    def __call__(self, t):
        if t <= self.breakpoints[0]:
            return float(self.rates[0])
        if t >= self.breakpoints[-1]:
            return float(self.rates[-1])
        idx = int(np.searchsorted(self.breakpoints, t, side="right") - 1)
        idx = min(max(idx, 0), len(self.rates) - 1)
        return float(self.rates[idx])


def sample_feed_and_ic(n: int, seed: int = None) -> dict:
    """
    Draw n initial-condition / feed-profile specifications via LHS.

    Returns a dict of arrays (length n) plus the per-sample feed rate matrix.
    """
    rng = np.random.default_rng(seed)

    # LHS over the scalar IC / duration dimensions
    keys = ["X0", "S0", "V0", "t_total", "Sin"]
    sampler = LatinHypercube(d=len(keys), seed=seed)
    unit = sampler.random(n)
    out = {}
    for j, k in enumerate(keys):
        lo, hi = _RANGES[k]
        out[k] = lo + unit[:, j] * (hi - lo)

    # Per-sample piecewise feed rates
    feed_rates = rng.uniform(0.0, F_MAX, size=(n, N_FEED_SEG))
    # optionally force an initial batch phase (F=0) for some samples
    batch_mask = rng.random(n) < P_BATCH_START
    feed_rates[batch_mask, 0] = 0.0
    out["feed_rates"] = feed_rates
    return out


def make_feed_fn(fp: dict) -> PiecewiseFeed:
    """
    Build a PiecewiseFeed from one sample's spec.

    `fp` must contain 't_total' and 'feed_rates' (length N_FEED_SEG).
    """
    t_total = float(fp["t_total"])
    rates = np.asarray(fp["feed_rates"], dtype=float)
    n_seg = len(rates)
    breakpoints = np.linspace(0.0, t_total, n_seg + 1)
    return PiecewiseFeed(breakpoints, rates)
