"""
Apply the trained Case Study 2 pipeline to real fed-batch xylitol experiments:
for each experiment, classify the mechanism, regress an initial parameter set,
and refine it by least-squares fitting. Writes a summary JSON (and optionally
per-experiment overlay plots) to results/xylitol_use_case/<tag>/realdata/.

Run:
  python xylitol_case_study/evaluate_realdata.py \
      --tag main \
      --xlsx data/FedBatch_xylitol_in_YP.xlsx \
      --sheet "S. cerevisiae (plasmid)" \
      --experiments BC12 BC15 BC18 BC19
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from xylitol_case_study.realdata import (
    load_models, load_experiments, run_experiment, plot_experiment_fit,
    plot_candidate_fits)


def plot_summary(summary, out_path):
    """Figure 2: predicted mechanism + confidence, and SSE init->fitted, per experiment."""
    rows = [r for r in summary if "error" not in r]
    if not rows:
        return
    names = [r["experiment"] for r in rows]
    x = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4))
    conf = [r["confidence"] for r in rows]
    bars = a1.bar(x, conf, color="#1f77b4")
    for xi, r in zip(x, rows):
        a1.text(xi, r["confidence"] + 0.01, f"M{r['predicted_model']}", ha="center", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=45, ha="right")
    a1.set_ylabel("classifier confidence"); a1.set_ylim(0, 1.05)
    a1.set_title("Identified mechanism & confidence")
    w = 0.4
    si = [r.get("sse_init", np.nan) for r in rows]
    sf = [r.get("sse_fitted", np.nan) for r in rows]
    a2.bar(x - w / 2, si, w, color="#bbbbbb", label="LSTM init")
    a2.bar(x + w / 2, sf, w, color="#1f77b4", label="refined")
    a2.set_yscale("log"); a2.set_xticks(x); a2.set_xticklabels(names, rotation=45, ha="right")
    a2.set_ylabel("SSE (log)"); a2.set_title("Fit error: LSTM init vs refined"); a2.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_baseline_comparison(summary, out_path):
    """Figure 5: final SSE (and cost) of LSTM-init vs naive vs multistart, per experiment."""
    def _ms(r):
        return r.get("sse_multistart", r.get("multistart_sse"))
    rows = [r for r in summary if "error" not in r and _ms(r) is not None]
    if not rows:
        return
    names = [r["experiment"] for r in rows]
    x = np.arange(len(names)); w = 0.27
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4))
    a1.bar(x - w, [r.get("sse_fitted", np.nan) for r in rows], w, color="#1f77b4", label="LSTM-init")
    a1.bar(x, [r.get("naive_sse", np.nan) for r in rows], w, color="#ff7f0e", label="naive")
    a1.bar(x + w, [_ms(r) for r in rows], w, color="#2ca02c", label="multistart")
    a1.set_yscale("log"); a1.set_xticks(x); a1.set_xticklabels(names, rotation=45, ha="right")
    a1.set_ylabel("final SSE (log)"); a1.set_title("Final fit error by initialisation"); a1.legend()
    a2.bar(x - w, [r.get("nfev", np.nan) for r in rows], w, color="#1f77b4", label="LSTM-init")
    a2.bar(x, [r.get("naive_nfev", np.nan) for r in rows], w, color="#ff7f0e", label="naive")
    a2.bar(x + w, [r.get("multistart_nfev", np.nan) for r in rows], w, color="#2ca02c", label="multistart")
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=45, ha="right")
    a2.set_ylabel("function evaluations"); a2.set_title("Fitting cost"); a2.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def save_sse_matrix(summary, out_dir):
    """Experiment x mechanism matrix of the LSTM-predicted parameters' SSE (no refinement).
    Writes sse_matrix.csv, sse_matrix.json and a heatmap sse_matrix.png."""
    import csv
    rows = [r for r in summary if r.get("lstm_sse_all")]
    if not rows:
        return
    mids = sorted({int(k) for r in rows for k in r["lstm_sse_all"]})
    names = [r["experiment"] for r in rows]
    M = np.full((len(rows), len(mids)), np.nan)
    for i, r in enumerate(rows):
        for k, v in r["lstm_sse_all"].items():
            M[i, mids.index(int(k))] = v
    with open(os.path.join(out_dir, "sse_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["experiment"] + [f"model{m}" for m in mids])
        for nm, row in zip(names, M):
            w.writerow([nm] + [f"{x:.4f}" if np.isfinite(x) else "" for x in row])
    json.dump({"experiments": names, "models": mids, "sse": M.tolist()},
              open(os.path.join(out_dir, "sse_matrix.json"), "w"), indent=2)
    fig, ax = plt.subplots(figsize=(1.6 + 0.55 * len(mids), 1.4 + 0.5 * len(rows)))
    im = ax.imshow(np.log10(np.clip(M, 1e-6, None)), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(mids))); ax.set_xticklabels([f"M{m}" for m in mids], rotation=90, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(rows)):
        if np.any(np.isfinite(M[i])):
            j = int(np.nanargmin(M[i]))
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="red", lw=2))
    ax.set_title("LSTM-only SSE: experiment × mechanism (log; red = lowest per row)")
    fig.colorbar(im, fraction=0.03, label=r"log$_{10}$ SSE")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "sse_matrix.png"), dpi=150); plt.close(fig)


# ── parallelism: one worker process per experiment ───────────────────────────
# Each worker loads the models + experiments ONCE (initializer), so torch objects
# are never pickled/shipped; only the experiment name is sent per task, and the
# result dict comes back. Heavy work (ODE fits) is CPU-bound and independent.

_MODELS = _EXPS = _OPTS = None


def _init_worker(tag, xlsx, sheet, opts):
    global _MODELS, _EXPS, _OPTS
    try:
        import torch
        torch.set_num_threads(1)     # avoid oversubscribing cores across processes
    except Exception:
        pass
    _MODELS = load_models(tag=tag)
    _EXPS = load_experiments(xlsx, sheet)
    _OPTS = opts


def _worker(name):
    if name not in _EXPS:
        return {"experiment": name, "error": "not in sheet"}
    try:
        return run_experiment(_EXPS[name], name, **_OPTS, **_MODELS)
    except Exception as e:
        return {"experiment": name, "error": f"{type(e).__name__}: {e}"}


def _print_res(res):
    name = res.get("experiment", "?")
    if "error" in res:
        print(f"  {name}: {res['error']}"); return
    msg = (f"  {name}: model{res['predicted_model']} "
           f"(conf {res['confidence']:.2f})  SSE {res['sse_init']:.2f}")
    if "sse_fitted" in res:
        msg += f" -> {res['sse_fitted']:.2f} ({res['nfev']} evals)"
    cand = " ".join(f"M{c['model']}({c['prob']:.2f})" for c in res.get("candidates", []))
    print(msg + (f"\n      candidates >0.01: {cand}" if cand else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--xlsx", default=os.path.join(_ROOT, "data", "FedBatch_xylitol_in_YP.xlsx"))
    ap.add_argument("--sheet", default="S. cerevisiae (plasmid)")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="experiment names to run (default: all in the sheet)")
    ap.add_argument("--method", default="gpr", choices=["gpr", "spline"])
    ap.add_argument("--no-fit", action="store_true", help="classify+regress only, skip refinement")
    ap.add_argument("--plots", action="store_true", help="write per-experiment overlay + summary figures")
    ap.add_argument("--baselines", action="store_true",
                    help="also fit naive + multistart baselines (Stage-2 comparison, figure 5)")
    ap.add_argument("--multistart", type=int, default=10)
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figures from an existing realdata_summary.json, no inference")
    ap.add_argument("--estimate-conf", type=float, default=0.1,
                    help="do full LSTM/refine/multistart estimation for every mechanism above this confidence")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes, one experiment each (0 = auto: min(#experiments, #cores); 1 = serial)")
    args = ap.parse_args()

    out_dir = os.path.join(_ROOT, "results", "xylitol_use_case", args.tag, "realdata")
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        summary = json.load(open(os.path.join(out_dir, "realdata_summary.json")))
        plot_summary(summary, os.path.join(out_dir, "summary_experiments.png"))
        if any(("sse_multistart" in r or "multistart_sse" in r) for r in summary):
            plot_baseline_comparison(summary, os.path.join(out_dir, "baseline_comparison.png"))
        save_sse_matrix(summary, out_dir)
        # per-experiment overlays re-simulate the ODE from stored params (no neural inference)
        try:
            exps = load_experiments(args.xlsx, args.sheet)
            for r in summary:
                if "predicted_model" in r and r.get("experiment") in exps:
                    plot_experiment_fit(exps[r["experiment"]], r,
                                        os.path.join(out_dir, f"fit_{r['experiment']}.png"))
                    plot_candidate_fits(exps[r["experiment"]], r, out_dir)
        except Exception as e:
            print(f"  (per-experiment overlays skipped: {e})")
        print("regenerated figures from realdata_summary.json — no inference run")
        return

    exps = load_experiments(args.xlsx, args.sheet)     # main process: for names + plotting
    names = args.experiments or list(exps.keys())
    for n in [n for n in names if n not in exps]:
        print(f"  skip {n} (not in sheet)")
    names = [n for n in names if n in exps]

    opts = dict(method=args.method, do_fit=not args.no_fit, do_baselines=args.baselines,
                n_multistart=args.multistart, estimate_conf=args.estimate_conf)
    n_workers = args.workers or min(len(names), os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(names)))

    if n_workers > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        print(f"running {len(names)} experiments across {n_workers} worker processes ...")
        ctx = mp.get_context("spawn")
        results = {}
        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                    initializer=_init_worker,
                                    initargs=(args.tag, args.xlsx, args.sheet, opts)) as ex:
            futs = {ex.submit(_worker, nm): nm for nm in names}
            for fut in cf.as_completed(futs):
                res = fut.result(); results[futs[fut]] = res; _print_res(res)
        summary = [results[nm] for nm in names]        # restore input order
    else:
        models = load_models(tag=args.tag)
        summary = []
        for name in names:
            res = run_experiment(exps[name], name, **opts, **models)
            summary.append(res); _print_res(res)

    json.dump(summary, open(os.path.join(out_dir, "realdata_summary.json"), "w"), indent=2)
    save_sse_matrix(summary, out_dir)   # experiment x mechanism LSTM-SSE matrix (csv/json/png)
    if args.plots:
        for res in summary:
            if "predicted_model" in res and res.get("experiment") in exps:
                plot_experiment_fit(exps[res["experiment"]], res,
                                    os.path.join(out_dir, f"fit_{res['experiment']}.png"))
                plot_candidate_fits(exps[res["experiment"]], res, out_dir)
        plot_summary(summary, os.path.join(out_dir, "summary_experiments.png"))
        if args.baselines:
            plot_baseline_comparison(summary, os.path.join(out_dir, "baseline_comparison.png"))
        print("  wrote figures: fit_<exp>.png, fit_<exp>_M<mid>.png (per candidate), "
              "summary_experiments.png, sse_matrix.png"
              + (", baseline_comparison.png" if args.baselines else ""))
    print(f"\nwrote {os.path.join(out_dir, 'realdata_summary.json')}")


if __name__ == "__main__":
    main()
