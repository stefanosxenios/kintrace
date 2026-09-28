"""
Out-of-distribution (OOD) experiment — why a classifier is needed.

We build a trajectory from a TRUE mechanism whose parameters lie OUTSIDE the
training ranges, so the concentrations settle on a plateau never seen during
training. We then select the mechanism four ways and compare:

  1. classifier            — Stage-1 softmax over mechanisms (reads trajectory SHAPE)
  2. lstm_raw   MSE        — simulate each mechanism with its Stage-2 regressor
                             parameters (sigmoid-bounded to TRAINING ranges) and
                             rank by fit MSE. Because the regressor cannot express
                             the OOD plateau, this ranking can be MISLEADING.
  3. lstm_refine MSE       — least-squares refinement STARTING FROM the regressor
                             parameters, with generous bounds (may reach OOD).
  4. fitall     MSE        — conventional baseline: fit every mechanism from
                             naive/LHS-multistart initialisation with generous
                             bounds, rank by MSE (the AIC/BIC-style selector).

Hypothesis: the classifier names the true mechanism from shape; naive
best-regressor-MSE selection can be fooled by the OOD plateau; and refining from
the LSTM shifts the MSE ranking back toward the true mechanism.

Outputs (results/simple_benchmark<TAG>/):
  ood_ranking.csv / ood_ranking.json
  ood_flip.png           per-method bar charts, true mechanism highlighted
  ood_curves.png         observed vs LSTM-raw (bounded) vs refined, true mechanism

Run:
  python benchmark_simple/ood_experiment.py --true-mech 4 --ood-xmax 50
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from benchmark_simple import data as D
from benchmark_simple import fitting as Fit
from benchmark_simple.models_io import load_classifier, load_regressor
from datagen.noise import add_gaussian_noise
from kinetic_models.simple_models import (
    SIMPLE_MODELS, get_param_names, compute_mu, get_simple_model)


# ── OOD trajectory construction ──────────────────────────────────────────────

def make_condition(X0, S0, V0, Sin, t_total, feed_rates):
    row = {"X0": X0, "S0": S0, "V0": V0, "Sin": Sin, "t_total": t_total}
    for j, f in enumerate(feed_rates):
        row[f"feed_rate_{j}"] = f
    return pd.Series(row)


def build_features(true_mid, params, cond_row, T, noise_pct, seed):
    """Simulate the OOD case and assemble the hybrid [X,S,V,F/V,mu,qS,F/V] input."""
    traj = Fit.simulate_traj(true_mid, params, cond_row, T)
    feed_fn = Fit._feed_from_condition(cond_row)
    tvec = np.linspace(0.0, float(cond_row["t_total"]), T)
    V = traj[:, 2]
    FV = np.array([feed_fn(t) / max(float(V[i]), 1e-9) for i, t in enumerate(tvec)])
    mu = compute_mu(true_mid, traj[:, 0], traj[:, 1], params)
    p = {n: float(v) for n, v in zip(get_param_names(true_mid), params)}
    qS = (1.0 / p["Yxs"]) * mu
    conc = np.concatenate([traj, FV[:, None]], axis=1)               # [X,S,V,F/V]
    rates = np.concatenate([np.stack([mu, qS], 1), FV[:, None]], 1)  # [mu,qS,F/V]
    conc_n = add_gaussian_noise(conc, noise_pct, clip_zero=True, seed=seed)
    rate_n = add_gaussian_noise(rates, noise_pct, clip_zero=False, seed=seed + 1)
    hybrid = np.concatenate([conc_n[:, :4], rate_n[:, :2]], axis=1)  # (T,6) F/V once
    reps = {"concentrations": conc_n, "rates": rate_n, "hybrid": hybrid}
    return reps, conc_n[:, 0], conc_n[:, 1], tvec, traj


def mse_scaled(model_id, theta, cond_row, Xobs, Sobs):
    traj = Fit.simulate_traj(model_id, theta, cond_row, len(Xobs))
    if traj is None:
        return 1e6
    sX = max(np.std(Xobs), 1e-3); sS = max(np.std(Sobs), 1e-3)
    return float(np.mean(((traj[:, 0] - Xobs) / sX) ** 2
                         + ((traj[:, 1] - Sobs) / sS) ** 2) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-mech", type=int, default=4, help="true mechanism id (1-4)")
    ap.add_argument("--ood-xmax", type=float, default=50.0,
                    help="OOD Xmax for biomass inhibition (training max 40)")
    ap.add_argument("--scenario", default="", choices=["", "monod_low"],
                    help="monod_low = a true Monod curve far below the training regime")
    ap.add_argument("--tag", default="", help="results dir suffix (blank = T=100 run)")
    ap.add_argument("--rep", default="hybrid",
                    choices=["concentrations", "rates", "hybrid"],
                    help="input representation the loaded models were trained on")
    ap.add_argument("--scaling", default="global", choices=["global", "per_traj"],
                    help="default scaling for both stages (overridden by the two below)")
    ap.add_argument("--clf-scaling", default="", choices=["", "global", "per_traj"],
                    help="scaling the CLASSIFIER was trained with (blank = --scaling)")
    ap.add_argument("--reg-scaling", default="", choices=["", "global", "per_traj"],
                    help="scaling the REGRESSOR was trained with (blank = --scaling)")
    ap.add_argument("--dataset", default="generated_simple",
                    help="dataset whose scaler/ranges the models were trained on")
    ap.add_argument("--clf-ckpt", default="",
                    help="explicit Stage-1 checkpoint (e.g. results/paper_benchmark/"
                         "models/classifier_nondim_hybrid.pt); default <results_dir>/classifier.pt")
    ap.add_argument("--reg-ckpt", default="",
                    help="explicit Stage-2 checkpoint (e.g. results/paper_benchmark/"
                         "models/regressor_global_hybrid.pt); default <results_dir>/regressor.pt")
    ap.add_argument("--out-dir", default="",
                    help="explicit output dir for the ood_* figures/tables "
                         "(default: the results dir derived from --tag)")
    ap.add_argument("--timepoints", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--multistart", type=int, default=5)
    args = ap.parse_args()
    device = "cpu"; torch.set_num_threads(4)

    results_dir = os.path.join(_ROOT, "results",
                               f"simple_benchmark_{args.tag}" if args.tag else "simple_benchmark")
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(_ROOT, args.out_dir)) if args.out_dir else results_dir
    os.makedirs(out_dir, exist_ok=True)
    tmid = args.true_mech

    # OOD ground-truth parameters (outside training ranges) + a feed profile
    if args.scenario == "monod_low":
        # A TRUE Monod curve whose growth is far BELOW the training regime:
        # very low mumax (OOD, <0.30) so biomass stays low and slow-rising.
        tmid = 1
        ood_params = [0.10, 2.0, 0.22]           # [mumax(OOD low), Ks, Yxs(OOD low)]
        cond = make_condition(X0=1.5, S0=10.0, V0=1.5, Sin=110.0, t_total=30.0,
                              feed_rates=[0.0, 0.02, 0.02, 0.01])
    else:
        ood_params = {
            1: [0.7, 1.0, 0.75],                    # Yxs OOD (>0.6)
            2: [0.7, 0.3, 0.75],
            3: [0.7, 1.0, 0.3, 0.5],                # KI OOD (<0.5): very strong inhibition
            4: [0.7, 1.0, args.ood_xmax, 0.5],      # Xmax OOD (>40): unseen biomass plateau
        }[tmid]
        cond = make_condition(X0=2.0, S0=10.0, V0=1.5, Sin=220.0, t_total=30.0,
                              feed_rates=[0.02, 0.12, 0.12, 0.08])
    tname = SIMPLE_MODELS[tmid][0]

    reps, Xobs, Sobs, tvec, traj_clean = build_features(
        tmid, ood_params, cond, args.timepoints, args.noise, args.seed)
    print(f"True mechanism: {tname}  OOD params={ood_params}  rep={args.rep}")
    print(f"Biomass plateau reached: {Xobs.max():.1f} g/L "
          f"(training biomass rarely exceeds ~35-40)")

    # Separate transforms for classifier vs regressor (they may be trained with
    # different scaling, e.g. nondimensional classifier + scale-aware regressor).
    clf_scaling = args.clf_scaling or args.scaling
    reg_scaling = args.reg_scaling or args.scaling
    ds = args.dataset
    splits_clf = D.make_splits(_ROOT, rep=args.rep, seed=42, max_per_model=900,
                               dataset=ds, scaling=clf_scaling)
    splits_reg = D.make_splits(_ROOT, rep=args.rep, seed=42, max_per_model=900,
                               dataset=ds, scaling=reg_scaling)
    input_dim = splits_clf["input_dim"]
    clf = load_classifier(results_dir, input_dim, device, ckpt=(args.clf_ckpt or None))
    reg = load_regressor(results_dir, input_dim, device, ckpt=(args.reg_ckpt or None))

    xb_clf = torch.tensor(splits_clf["transform"](reps[args.rep][None, :, :]).astype(np.float32), device=device)
    xb_reg = torch.tensor(splits_reg["transform"](reps[args.rep][None, :, :]).astype(np.float32), device=device)
    with torch.no_grad():
        probs = clf.predict_proba(xb_clf)[0].cpu().numpy()

    rows = []
    lstm_curves = {}
    for k, mid in enumerate(D.MODEL_IDS):
        with torch.no_grad():
            lstm_norm = reg(xb_reg, mid).cpu().numpy()[0]
        theta_lstm = D.denormalise_params(lstm_norm[None, :], mid)[0]
        mse_raw = mse_scaled(mid, theta_lstm, cond, Xobs, Sobs)

        gb = Fit.generous_bounds(mid)
        ref = Fit.fit_from_init(mid, theta_lstm, cond, Xobs, Sobs, bounds=gb)
        mse_ref = mse_scaled(mid, ref["params"], cond, Xobs, Sobs)

        # fit-all baseline: LHS multistart from naive/random, generous bounds
        rng = np.random.default_rng(args.seed + mid)
        best = None; tot_nfev = 0
        for _ in range(args.multistart):
            th0 = Fit.random_init(mid, rng)
            r = Fit.fit_from_init(mid, th0, cond, Xobs, Sobs, bounds=gb)
            tot_nfev += r["nfev"]
            if best is None or r["sse"] < best["sse"]:
                best = r
        mse_fitall = mse_scaled(mid, best["params"], cond, Xobs, Sobs)

        rows.append(dict(mechanism=SIMPLE_MODELS[mid][0], is_true=(mid == tmid),
                         classifier_prob=float(probs[k]),
                         mse_lstm_raw=mse_raw, mse_lstm_refine=mse_ref,
                         nfev_refine=ref["nfev"], mse_fitall=mse_fitall,
                         nfev_fitall=tot_nfev))
        if mid == tmid:
            lstm_curves["raw"] = Fit.simulate_traj(mid, theta_lstm, cond, len(Xobs))
            lstm_curves["refine"] = Fit.simulate_traj(mid, ref["params"], cond, len(Xobs))
            lstm_curves["fitall"] = Fit.simulate_traj(mid, best["params"], cond, len(Xobs))
            lstm_curves["theta_raw"] = theta_lstm
            lstm_curves["theta_ref"] = ref["params"]

    sfx = f"_{args.scenario}" if args.scenario else ""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"ood_ranking{sfx}.csv"), index=False)

    def _sel(col, best="min"):
        i = df[col].idxmin() if best == "min" else df[col].idxmax()
        return df.loc[i, "mechanism"]
    selection = {"true": tname,
                 "classifier": _sel("classifier_prob", "max"),
                 "lstm_raw_MSE": _sel("mse_lstm_raw"),
                 "lstm_refine_MSE": _sel("mse_lstm_refine"),
                 "fitall_MSE": _sel("mse_fitall")}
    json.dump({"selection": selection, "table": rows,
               "ood_params": ood_params, "plateau": float(Xobs.max())},
              open(os.path.join(out_dir, f"ood_ranking{sfx}.json"), "w"), indent=2)
    print("\n=== mechanism selected by each method ===")
    for k, v in selection.items():
        flag = "  <-- TRUE" if v == tname and k != "true" else ""
        print(f"  {k:18s}: {v}{flag}")
    print("\n", df.round(3).to_string(index=False))

    _plot_flip(df, tname, selection, out_dir, sfx)
    _plot_curves(tvec, Xobs, Sobs, lstm_curves, tname, out_dir, sfx)
    print(f"\nSaved ood_flip{sfx}.png, ood_curves{sfx}.png to {out_dir}")


def _plot_flip(df, tname, selection, out, sfx=""):
    mechs = df["mechanism"].tolist()
    true_i = df.index[df["is_true"]].tolist()[0]
    panels = [("classifier_prob", "Classifier probability", "max"),
              ("mse_lstm_raw", "LSTM-raw fit MSE (bounded)", "min"),
              ("mse_lstm_refine", "LSTM-init refined MSE", "min"),
              ("mse_fitall", "Fit-all MSE (generous bounds)", "min")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (col, title, best) in zip(axes, panels):
        vals = df[col].values
        colors = ["#2ca02c" if i == true_i else "#a0a0a0" for i in range(len(mechs))]
        ax.bar(mechs, vals, color=colors)
        sel_i = (np.argmax(vals) if best == "max" else np.argmin(vals))
        ax.bar(mechs[sel_i], vals[sel_i], facecolor="none", edgecolor="#d62728", lw=2.5)
        if best == "min" and np.nanmax(vals) / max(np.nanmin(vals), 1e-9) > 20:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        for lab in ax.get_xticklabels():
            lab.set_rotation(25); lab.set_ha("right")
    fig.suptitle(f"OOD mechanism selection (true = {tname}). "
                 f"Green = true mechanism, red outline = method's pick.", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ood_flip{sfx}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(tvec, Xobs, Sobs, curves, tname, out, sfx=""):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, obs, comp, lab in [(axs[0], Xobs, 0, "Biomass X (g/L)"),
                               (axs[1], Sobs, 1, "Substrate S (g/L)")]:
        ax.scatter(tvec, obs, s=14, color="k", alpha=0.5, zorder=3, label="OOD observed")
        if curves.get("raw") is not None:
            ax.plot(tvec, curves["raw"][:, comp], "--", color="#ff7f0e", lw=2,
                    label="LSTM-raw (bounded to training)")
        if curves.get("refine") is not None:
            ax.plot(tvec, curves["refine"][:, comp], "-", color="#1f77b4", lw=2,
                    label="LSTM-init refined")
        if curves.get("fitall") is not None:
            ax.plot(tvec, curves["fitall"][:, comp], "-", color="#2ca02c", lw=2,
                    label="fit-all (generous bounds)")
        ax.set_xlabel("time (h)"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"True mechanism {tname}: the bounded regressor cannot express the OOD "
                 f"plateau; only generous-bounds fitting recovers it.")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"ood_curves{sfx}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
