"""
Dataset-overview figure for the paper (Case Study 1).

Reproduces the ensemble style of scripts/sindy_paper_intro.ipynb, but colours
each mechanism so the reader can see how the four candidate mechanisms spread
under a shared feed profile and inoculum. For each mechanism we sample many
parameter sets by Latin Hypercube Sampling within the training ranges, integrate
under a fixed representative fed-batch feed, and overlay X, S, V and the feed.

Output: results/simple_benchmark/dataset_overview.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import odeint

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from kinetic_models.simple_models import SIMPLE_MODELS, get_simple_model, get_param_names
from datagen.param_ranges_simple import get_param_bounds
from scipy.stats.qmc import LatinHypercube

# fixed representative fed-batch feed (matches the intro-notebook profile)
def feed(t):
    if t < 1.0:  return 0.0
    if t < 2.0:  return 0.05
    if t < 7.0:  return 0.10
    if t < 8.4:  return 0.05
    return 0.0

COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#9467bd", 4: "#2ca02c"}
X0, S0, V0, SIN, TEND = 5.5, 0.1, 1.5, 120.0, 10.0
N_PER = 60


def simulate(mid, params, t):
    rhs = get_simple_model(mid)
    return odeint(rhs, [X0, S0, V0], t, args=(np.asarray(params), feed, SIN),
                  rtol=1e-6, atol=1e-8, mxstep=5000)


def main():
    t = np.linspace(0, TEND, 200)
    fig, (axX, axS) = plt.subplots(1, 2, figsize=(12, 4.6))
    for mid in SIMPLE_MODELS:
        lo, hi = get_param_bounds(mid)
        unit = LatinHypercube(d=len(lo), seed=100 + mid).random(N_PER)
        P = np.array(lo) + unit * (np.array(hi) - np.array(lo))
        c = COLORS[mid]; name = SIMPLE_MODELS[mid][0]
        first = True
        for p in P:
            y = simulate(mid, p, t)
            if np.any(~np.isfinite(y)) or y[:, 0].max() > 200:
                continue
            lbl = name if first else None; first = False
            axX.plot(t, y[:, 0], color=c, alpha=0.25, lw=0.9, label=lbl)
            axS.plot(t, y[:, 1], color=c, alpha=0.25, lw=0.9)

    axX.set_title("Biomass (X)"); axX.set_ylabel("X (g/L)"); axX.set_xlabel("time (h)")
    axX.legend(fontsize=9, framealpha=0.9)
    axS.set_title("Substrate (S)"); axS.set_ylabel("S (g/L)"); axS.set_xlabel("time (h)")
    for a in (axX, axS): a.grid(alpha=0.25)
    fig.suptitle("Benchmark dataset — four mechanisms under a shared feed (colour = mechanism)",
                 y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(_ROOT, "results", "simple_benchmark", "dataset_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
