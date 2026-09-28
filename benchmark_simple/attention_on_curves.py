"""
In-distribution result figure with attention overlaid on the PREDICTED curves.

For one correctly-identified example per mechanism (parameters sampled within the
training ranges), the pipeline is run on the fly:
  nondimensional classifier -> mechanism + confidence + attention weights
  scale-aware regressor      -> parameter guess (dashed curve)
  least-squares refinement   -> refined fit (solid curve)
The biomass trajectory is shown with the noisy observations, the predicted and
refined curves, and the Stage-1 attention shaded underneath.

Output: results/simple_benchmark/attention_on_curves.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from benchmark_simple import data as D, fitting as Fit
from benchmark_simple.models_io import load_classifier, load_regressor
from benchmark_simple.ood_experiment import build_features, make_condition
from datagen.param_ranges_simple import get_param_bounds
from kinetic_models.simple_models import SIMPLE_MODELS, get_param_names

RESULTS = os.path.join(_ROOT, "results", "simple_benchmark_rec")
DEV = "cpu"; T = 100


def sample_in_range(mid, rng):
    lo, hi = get_param_bounds(mid)
    return np.array(lo) + rng.random(len(lo)) * (np.array(hi) - np.array(lo))


def main():
    torch.set_num_threads(4)
    splits_clf = D.make_splits(_ROOT, rep="hybrid", scaling="per_traj", dataset="generated_simple")
    splits_reg = D.make_splits(_ROOT, rep="hybrid", scaling="global",   dataset="generated_simple")
    idim = splits_clf["input_dim"]
    clf = load_classifier(RESULTS, idim, DEV); reg = load_regressor(RESULTS, idim, DEV)

    # a representative in-distribution operating condition
    cond = make_condition(X0=3.0, S0=8.0, V0=1.5, Sin=140.0, t_total=24.0,
                          feed_rates=[0.0, 0.05, 0.08, 0.03])
    rng = np.random.default_rng(3)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8)); axes = axes.ravel()
    for ax, mid in zip(axes, SIMPLE_MODELS):
        # find a correctly-classified in-range example
        for _ in range(12):
            params = sample_in_range(mid, rng)
            reps, Xobs, Sobs, tvec, _ = build_features(mid, params, cond, T, 0.03, int(rng.integers(1e6)))
            xbc = torch.tensor(splits_clf["transform"](reps["hybrid"][None]).astype("float32"))
            xbr = torch.tensor(splits_reg["transform"](reps["hybrid"][None]).astype("float32"))
            with torch.no_grad():
                logits, w = clf(xbc, return_attention=True)
                pred = int(logits.argmax(1)); conf = float(torch.softmax(logits, 1).max())
            if D.CLASS_TO_ID[pred] == mid:
                break
        att = w[0].detach().numpy(); att = att / (att.max() + 1e-9)
        with torch.no_grad():
            th = D.denormalise_params(reg(xbr, mid).numpy()[0][None], mid)[0]
        ref = Fit.fit_from_init(mid, th, cond, Xobs, Sobs, bounds=Fit.generous_bounds(mid))
        pred_curve = Fit.simulate_traj(mid, th, cond, T)
        ref_curve = Fit.simulate_traj(mid, ref["params"], cond, T)

        # attention shading (normalised time index mapped onto real time)
        for k in range(len(tvec) - 1):
            ax.axvspan(tvec[k], tvec[k + 1], color="#d62728", alpha=0.45 * float(att[k]), lw=0)
        ax.scatter(tvec, Xobs, s=13, c="k", alpha=0.55, zorder=5, label="observed")
        ax.plot(tvec, pred_curve[:, 0], "--", color="#ff7f0e", lw=2, label="LSTM prediction")
        ax.plot(tvec, ref_curve[:, 0], "-", color="#1f77b4", lw=2, label="refined fit")
        ax.set_title(f"{SIMPLE_MODELS[mid][0]}  |  predicted: {SIMPLE_MODELS[D.CLASS_TO_ID[pred]][0]} "
                     f"(conf {conf:.2f})", fontsize=10)
        ax.set_xlabel("time (h)"); ax.set_ylabel("Biomass X (g/L)"); ax.grid(alpha=0.2)
        if mid == list(SIMPLE_MODELS)[0]:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("In-distribution: mechanism + parameters predicted on the fly, "
                 "with Stage-1 attention (red shading)", y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(_ROOT, "results", "simple_benchmark", "attention_on_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
