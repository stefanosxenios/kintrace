"""
Paper OOD figures: run both OOD scenarios (a low Monod curve and a high
biomass-inhibition plateau) through the recommended pipeline (nondimensional
classifier + scale-aware regressor from results/simple_benchmark_rec), fitting
every mechanism with rigorous multistart over generous bounds, and produce two
combined figures:

  ood_curves_paper.png : observed vs LSTM-raw / refined / fit-all, per scenario
  ood_flip_paper.png   : per-method mechanism selection, per scenario

Run:  python benchmark_simple/ood_paper_figures.py --restarts 40
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from benchmark_simple import data as D, fitting as Fit
from benchmark_simple.models_io import load_classifier, load_regressor
from benchmark_simple.ood_experiment import build_features, make_condition, mse_scaled
from kinetic_models.simple_models import SIMPLE_MODELS, get_param_names

# Shipped weights: per-trajectory-scaled classifier + bank of per-mechanism regressors.
# Passing the DIRECTORY as the regressor checkpoint loads the per-mechanism bank.
MODELS = os.path.join(_ROOT, "results", "simple_benchmark_rec")
DEF_CLF_CKPT = os.path.join(MODELS, "classifier.pt")
DEF_REG_CKPT = MODELS
OUT = os.path.join(_ROOT, "results", "simple_benchmark")
DEV = "cpu"; T = 100
PALETTE = {1: "#1f77b4", 2: "#ff7f0e", 3: "#9467bd", 4: "#2ca02c"}

SCENARIOS = [
    ("monod_low", dict(mech=1, params=[0.10, 2.0, 0.22],
        cond=dict(X0=1.5, S0=10.0, V0=1.5, Sin=110.0, t_total=30.0,
                  feed_rates=[0.0, 0.02, 0.02, 0.01]))),
    ("biomass_high", dict(mech=4, params=[0.7, 1.0, 65.0, 0.5],
        cond=dict(X0=2.0, S0=10.0, V0=1.5, Sin=220.0, t_total=30.0,
                  feed_rates=[0.02, 0.12, 0.12, 0.08]))),
]


def run(sc, clf, reg, sclf, sreg, n_restarts, seed=0):
    cond = make_condition(**sc["cond"]); tmech = sc["mech"]
    reps, Xobs, Sobs, tvec, _ = build_features(tmech, sc["params"], cond, T, 0.03, 7)
    xbc = torch.tensor(sclf["transform"](reps["hybrid"][None]).astype("float32"))
    xbr = torch.tensor(sreg["transform"](reps["hybrid"][None]).astype("float32"))
    with torch.no_grad():
        probs = clf.predict_proba(xbc)[0].numpy()
    rng = np.random.default_rng(seed)
    rows, curves = {}, {}
    for k, mid in enumerate(D.MODEL_IDS):
        lo, hi = Fit.generous_bounds(mid)
        with torch.no_grad():
            th = D.denormalise_params(reg(xbr, mid).numpy()[0][None], mid)[0]
        raw = mse_scaled(mid, th, cond, Xobs, Sobs)
        ref = Fit.fit_from_init(mid, th, cond, Xobs, Sobs, bounds=(lo, hi))
        best = None
        for _ in range(n_restarts):
            th0 = lo + rng.random(len(lo)) * (hi - lo)
            r = Fit.fit_from_init(mid, th0, cond, Xobs, Sobs, bounds=(lo, hi))
            if best is None or r["sse"] < best["sse"]:
                best = r
        rows[mid] = dict(prob=float(probs[k]), raw=raw,
                         refine=mse_scaled(mid, ref["params"], cond, Xobs, Sobs),
                         fitall=mse_scaled(mid, best["params"], cond, Xobs, Sobs))
        if mid == tmech:
            curves = dict(raw=Fit.simulate_traj(mid, th, cond, T),
                          refine=Fit.simulate_traj(mid, ref["params"], cond, T),
                          fitall=Fit.simulate_traj(mid, best["params"], cond, T))
    return dict(tmech=tmech, Xobs=Xobs, Sobs=Sobs, tvec=tvec, rows=rows, curves=curves)


def fig_curves(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.3 * n))
    for i, (name, R) in enumerate(results):
        tv = R["tvec"]
        for j, (obs, comp, lab) in enumerate([(R["Xobs"], 0, "Biomass X (g/L)"),
                                              (R["Sobs"], 1, "Substrate S (g/L)")]):
            ax = axes[i, j]
            ax.scatter(tv, obs, s=13, c="k", alpha=.5, zorder=5, label="OOD observed")
            ax.plot(tv, R["curves"]["raw"][:, comp], "--", c="#ff7f0e", lw=2, label="LSTM-raw (bounded)")
            ax.plot(tv, R["curves"]["refine"][:, comp], "-", c="#1f77b4", lw=2, label="LSTM-init refined")
            ax.plot(tv, R["curves"]["fitall"][:, comp], "-", c="#2ca02c", lw=2, label="fit-all (generous)")
            ax.set_xlabel("time (h)"); ax.set_ylabel(lab); ax.grid(alpha=.3)
            if j == 0:
                ax.set_title(f"true = {SIMPLE_MODELS[R['tmech']][0]}", loc="left", fontsize=11)
        axes[i, 0].legend(fontsize=8)
    fig.suptitle("Out-of-distribution fits: the bounded regressor cannot express the OOD "
                 "regime; generous-bounds fitting recovers it", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ood_curves_paper.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_flip(results):
    n = len(results)
    panels = [("prob", "Classifier probability", "max"),
              ("raw", "LSTM-raw MSE (bounded)", "min"),
              ("refine", "LSTM-refined MSE", "min"),
              ("fitall", "Fit-all MSE (generous)", "min")]
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.8 * n))
    names = [SIMPLE_MODELS[m][0] for m in D.MODEL_IDS]
    for i, (sname, R) in enumerate(results):
        true_i = list(D.MODEL_IDS).index(R["tmech"])
        for a, (key, title, how) in enumerate(panels):
            ax = axes[i, a]
            vals = np.array([R["rows"][m][key] for m in D.MODEL_IDS])
            colors = ["#2ca02c" if k == true_i else "#a0a0a0" for k in range(len(names))]
            ax.bar(names, vals, color=colors)
            pick = int(np.argmax(vals) if how == "max" else np.argmin(vals))
            ax.bar(names[pick], vals[pick], facecolor="none", edgecolor="#d62728", lw=2.5)
            if how == "min" and np.nanmax(vals) / max(np.nanmin(vals), 1e-9) > 20:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(title, fontsize=10)
            if a == 0:
                ax.set_ylabel(f"true = {SIMPLE_MODELS[R['tmech']][0]}", fontsize=10)
            for l in ax.get_xticklabels():
                l.set_rotation(25); l.set_ha("right")
    fig.suptitle("Out-of-distribution mechanism selection (green = true, red outline = method's pick)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ood_flip_paper.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--clf-ckpt", default=DEF_CLF_CKPT,
                    help="Stage-1 classifier checkpoint (default: paper nondim hybrid)")
    ap.add_argument("--reg-ckpt", default=DEF_REG_CKPT,
                    help="Stage-2 regressor checkpoint (default: paper global hybrid)")
    ap.add_argument("--out-dir", default=OUT,
                    help="where to write the figures (default: paper_benchmark/figures)")
    args = ap.parse_args(); torch.set_num_threads(4)
    OUT = args.out_dir; os.makedirs(OUT, exist_ok=True)
    sclf = D.make_splits(_ROOT, rep="hybrid", scaling="per_traj", dataset="generated_simple")
    sreg = D.make_splits(_ROOT, rep="hybrid", scaling="global", dataset="generated_simple")
    clf = load_classifier(None, sclf["input_dim"], DEV, ckpt=args.clf_ckpt)
    reg = load_regressor(None, sreg["input_dim"], DEV, ckpt=args.reg_ckpt)
    results = [(name, run(sc, clf, reg, sclf, sreg, args.restarts)) for name, sc in SCENARIOS]
    fig_curves(results); fig_flip(results)
    print("saved ood_curves_paper.png and ood_flip_paper.png to", OUT)


if __name__ == "__main__":
    main()
