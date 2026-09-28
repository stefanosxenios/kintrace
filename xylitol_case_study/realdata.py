"""
Real-data pipeline for Case Study 2: apply the trained (scale-invariant)
classifier + regressor to measured fed-batch xylitol experiments, then refine
the LSTM-initialised parameters by least-squares fitting of the identified
mechanism to the data.

Reuses the existing preprocessing (`dataloader.experimental_inference`,
which handles sparse offline HPLC data + derivative estimation) and the
dual-substrate ODE library (`kinetic_models.base_models`, models 21-31).

Pipeline per experiment:
    preprocess (per-traj scaler)  -> classify mechanism
    preprocess (regressor scaler) -> regress physical parameters (LSTM init)
    least-squares fit (L-BFGS-B) from the LSTM init -> refined parameters

Example
-------
    from xylitol_case_study.realdata import load_models, load_experiments, run_experiment
    models = load_models(tag="main")
    exps = load_experiments("data/FedBatch_xylitol_in_YP.xlsx", "S. cerevisiae (plasmid)")
    res = run_experiment(exps["BC12"], "BC12", **models)
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from xylitol_case_study.models_io import build_classifier, load_permech_bank
from dataloader.dataloader import FedBatchDataLoader
from dataloader.experimental_inference import prepare_experimental_input
from kinetic_models.base_models import get_model

# experimental column -> ODE state index (0=X, 1=S1/Glu_eq, 2=S2/Xylose, 3=P/Xylitol)
_STATE_MAP = [("DW (g/L)", 0), ("Glu_eq", 1), ("Xylose", 2), ("Xylitol", 3)]


# ── model / experiment loading ───────────────────────────────────────────────

def load_models(tag="main", device="cpu"):
    """Load the trained classifier + per-mechanism regressor bank and their scalers/meta."""
    D = os.path.join(_ROOT, "results", "xylitol_use_case", tag)
    cmeta = json.load(open(os.path.join(D, "_classifier.json")))
    clf = build_classifier(cmeta["input_dim"], cmeta["n_classes"])
    clf.load_state_dict(torch.load(os.path.join(D, "classifier.pt"), map_location=device))
    clf.to(device).eval()
    clf_scaler = pickle.load(open(os.path.join(D, "classifier_scaler.pkl"), "rb"))

    rmeta = json.load(open(os.path.join(D, "regressor_meta.json")))
    counts = {int(k): v for k, v in rmeta["model_param_counts"].items()}
    reg = load_permech_bank(D, rmeta["input_dim"], counts, device=device)
    reg_scaler = pickle.load(open(os.path.join(D, "regressor_x_scaler.pkl"), "rb"))

    return dict(clf_model=clf, reg_model=reg, clf_scaler=clf_scaler, reg_scaler=reg_scaler,
                label_names=cmeta["label_names"], reg_meta=rmeta,
                dataset_type=cmeta["rep"], seq_len=cmeta["seq_len"], device=device)


def load_experiments(filepath, sheet_name):
    """Load + clean fed-batch experiments from an Excel workbook."""
    loader = FedBatchDataLoader(filepath=filepath, sheet_name=sheet_name)
    loader.load()
    return loader.clean()


# ── ODE simulation + objective (ported from regressor_v2.ipynb) ──────────────

def _ensure_glu_eq(df):
    if "Glu_eq" not in df.columns:
        df = df.copy()
        df["Glu_eq"] = df.get("Glucose", 0).fillna(0) + df.get("Ethanol", 0).fillna(0) * 1.2
    return df


def dynamic_sim_fit(df, params, model_fn, num_dense=200):
    """Integrate the 5-state ODE on a grid that includes the measurement times."""
    t0 = float(df["Time (h)"].iloc[0]); tf = float(df["Time (h)"].iloc[-2])
    t_dense = np.linspace(t0, tf, num_dense)
    t_samp = df["Time (h)"].dropna().values.astype(float)
    t = np.union1d(t_dense, t_samp)
    X0 = [float(df["DW (g/L)"].iloc[0]), float(df["Glu_eq"].iloc[0]),
          float(df["Xylose"].iloc[0]), float(df["Xylitol"].iloc[0]),
          float(df["Reactor volume (mL)"].iloc[0]) / 1000.0]
    result = odeint(model_fn, X0, t, args=(np.asarray(params), df),
                    rtol=1e-6, atol=1e-8, mxstep=5000)
    return result, t


def objective_sse(params, df, model_fn):
    try:
        result, t = dynamic_sim_fit(df, params, model_fn)
    except Exception:
        return 1e12
    if not np.all(np.isfinite(result)):
        return 1e12
    sse = 0.0
    for col, idx in _STATE_MAP:
        if col not in df.columns:
            continue
        sub = df[["Time (h)", col]].dropna()
        tt = sub["Time (h)"].values.astype(float); yy = sub[col].values.astype(float)
        yi = np.interp(tt, t, result[:, idx])
        sse += float(np.sum((yi - yy) ** 2))
    return sse


# ── full pipeline for one experiment ─────────────────────────────────────────

def _fit(p0, df, model_fn, bounds, eps=1e-3, maxiter=300):
    """Least-squares fit with L-BFGS-B in NORMALISED parameter space.

    The kinetic parameters span orders of magnitude (e.g. mumax ~0.5 vs KP ~80),
    so a finite-difference gradient in physical units is ill-conditioned and
    L-BFGS-B stalls (zero effective steps). We optimise the normalised variables
    u = (p - lb)/(ub - lb) in [0,1] (all O(1)), de-normalising to physical units
    inside the objective, which makes the gradient well-conditioned so the
    optimiser actually moves. `eps` is the finite-difference step in normalised
    space (larger than the default to survive ODE-integration noise).
    Returns a scipy OptimizeResult with `.x` in PHYSICAL units.
    """
    lbs = np.array([b[0] for b in bounds], float)
    ubs = np.array([b[1] for b in bounds], float)
    span = np.maximum(ubs - lbs, 1e-12)

    def _obj(u):
        phys = lbs + np.clip(u, 0.0, 1.0) * span
        return objective_sse(phys, df, model_fn)

    u0 = np.clip((np.asarray(p0, float) - lbs) / span, 0.0, 1.0)
    res = minimize(_obj, u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(lbs),
                   options={"maxiter": maxiter, "eps": eps})
    res.x = lbs + np.clip(res.x, 0.0, 1.0) * span      # back to physical units
    return res


def _multistart(fn, df, bounds, lbs_a, ubs_a, n, seed=0):
    rng = np.random.default_rng(seed); best, tot = None, 0
    for _ in range(n):
        rr = _fit(lbs_a + rng.random(len(lbs_a)) * (ubs_a - lbs_a), df, fn, bounds)
        tot += int(rr.nfev)
        if best is None or rr.fun < best.fun:
            best = rr
    return best, tot


def run_experiment(df, name, clf_model, reg_model, clf_scaler, reg_scaler,
                   label_names, reg_meta, dataset_type, seq_len, device="cpu",
                   method="gpr", do_fit=True, top_k=5,
                   do_baselines=False, n_multistart=10, prob_threshold=0.01,
                   estimate_conf=0.1, infer_all_sse=True):
    """Classify, then:
      (a) infer_all_sse: the LSTM-predicted parameters' SSE for EVERY mechanism
          (no refinement) -> `lstm_sse_all` (feeds the experiment x mechanism matrix);
      (b) full parameter estimation for every mechanism with confidence > estimate_conf
          (and the identified one): LSTM / refined / multistart parameters + SSE,
          stored under `mechanisms`.
    Top-level fields for the identified mechanism are kept for backward compatibility.
    """
    df = _ensure_glu_eq(df)

    # 1. classify (scale-invariant input)
    t_clf, _, _ = prepare_experimental_input(df, dataset_type=dataset_type,
                                             scaler=clf_scaler, seq_len=seq_len, method=method)
    clf_model.eval()
    with torch.no_grad():
        _, _, probs = clf_model.predict_with_confidence(t_clf.to(device))
    probs = probs.squeeze(0).cpu().numpy()
    prob_of = {int(label_names[i].replace("model", "")): float(probs[i]) for i in range(len(probs))}
    ranked = sorted(prob_of.items(), key=lambda kv: -kv[1])
    candidates = [{"model": m, "prob": round(p, 4)} for m, p in ranked if p > prob_threshold]
    reg_known = sorted(int(k) for k in reg_meta["bounds"])
    pred_mid = next((m for m, _ in ranked if m in reg_known), None)
    if pred_mid is None:
        return {"experiment": name, "candidates": candidates,
                "error": "no candidate mechanism known to the regressor"}
    confidence = prob_of[pred_mid]

    # 2. regressor input (computed once, shared across mechanisms)
    t_reg, _, _ = prepare_experimental_input(df, dataset_type=dataset_type,
                                             scaler=reg_scaler, seq_len=seq_len, method=method)
    t_reg = t_reg.to(device); reg_model.eval()

    def lstm_params(mid):
        b = reg_meta["bounds"][str(mid)]
        lbs = torch.tensor(b["lbs"], dtype=torch.float32, device=device)
        ubs = torch.tensor(b["ubs"], dtype=torch.float32, device=device)
        with torch.no_grad():
            return reg_model.predict_physical(t_reg, mid, lbs, ubs).squeeze(0).cpu().numpy()

    # 3. LSTM-only SSE for EVERY mechanism (no refinement) -> matrix row
    lstm_sse_all, p_cache = {}, {}
    if infer_all_sse:
        for mid in reg_known:
            p = lstm_params(mid); p_cache[mid] = p
            lstm_sse_all[str(mid)] = float(objective_sse(p, df, get_model(f"model{mid}")))

    # 4. full estimation for mechanisms above estimate_conf (+ the identified one)
    to_estimate = sorted({m for m in reg_known if prob_of.get(m, 0.0) > estimate_conf} | {pred_mid})
    mechanisms = {}
    for mid in to_estimate:
        fn = get_model(f"model{mid}")
        b = reg_meta["bounds"][str(mid)]; bounds = list(zip(b["lbs"], b["ubs"]))
        p_lstm = p_cache.get(mid) if mid in p_cache else lstm_params(mid)
        sse_lstm = lstm_sse_all.get(str(mid))
        if sse_lstm is None:
            sse_lstm = float(objective_sse(p_lstm, df, fn))
        entry = dict(prob=round(prob_of.get(mid, 0.0), 4),
                     param_names=reg_meta["param_names"][str(mid)],
                     lstm_params=p_lstm.tolist(), sse_lstm=sse_lstm)
        if do_fit:
            r = _fit(p_lstm, df, fn, bounds)
            entry.update(refined_params=r.x.tolist(), sse_refined=float(r.fun), refined_nfev=int(r.nfev))
        if do_baselines:
            best, tot = _multistart(fn, df, bounds, np.array(b["lbs"]), np.array(b["ubs"]), n_multistart)
            entry.update(multistart_params=best.x.tolist(), sse_multistart=float(best.fun),
                         multistart_nfev=int(tot))
        mechanisms[str(mid)] = entry

    # 5. top-level fields (backward compatible) from the identified mechanism
    top = mechanisms[str(pred_mid)]
    out = {"experiment": name, "predicted_model": pred_mid, "confidence": confidence,
           "candidates": candidates, "lstm_sse_all": lstm_sse_all, "mechanisms": mechanisms,
           "param_names": top["param_names"], "lstm_params": top["lstm_params"],
           "sse_init": top["sse_lstm"]}
    if "refined_params" in top:
        out.update(fitted_params=top["refined_params"], sse_fitted=top["sse_refined"], nfev=top["refined_nfev"])
    if "multistart_params" in top:
        out.update(multistart_params=top["multistart_params"], sse_multistart=top["sse_multistart"],
                   multistart_nfev=top["multistart_nfev"])
        b = reg_meta["bounds"][str(pred_mid)]
        r_naive = _fit(0.5 * (np.array(b["lbs"]) + np.array(b["ubs"])), df,
                       get_model(f"model{pred_mid}"), list(zip(b["lbs"], b["ubs"])))
        out.update(naive_sse=float(r_naive.fun), naive_nfev=int(r_naive.nfev), naive_params=r_naive.x.tolist())
    return out


# ── per-experiment overlay plot (figure 1) ───────────────────────────────────

_STATE_NAMES = ["Biomass X (g/L)", "Glucose-eq S1 (g/L)", "Xylose S2 (g/L)", "Xylitol P (g/L)"]
_EXP_COLS = ["DW (g/L)", "Glu_eq", "Xylose", "Xylitol"]


def _plot_mechanism_fit(df, exp_name, mid, prob, lstm_params, sse_lstm,
                        refined_params, sse_refined, multi_params, sse_multi,
                        out_path, prob_label="confidence"):
    """Overlay measured points against the LSTM-init, refined and (optional)
    multistart simulations for ONE mechanism of one experiment."""
    df = _ensure_glu_eq(df)
    fn = get_model(f"model{mid}")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)); axes = axes.ravel()
    try:
        sim_i, ti = dynamic_sim_fit(df, lstm_params, fn)
        sim_f, tf = (dynamic_sim_fit(df, refined_params, fn)
                     if refined_params is not None else (None, None))
        sim_m, tm = (dynamic_sim_fit(df, multi_params, fn)
                     if multi_params is not None else (None, None))
    except Exception as e:
        print(f"  plot {exp_name} M{mid}: simulation failed ({e})"); plt.close(fig); return
    for j, ax in enumerate(axes):
        il = f"LSTM init (SSE {sse_lstm:.1f})" if sse_lstm is not None else "LSTM init"
        ax.plot(ti, sim_i[:, j], "--", color="grey", lw=1.5, label=il)
        if sim_f is not None:
            fl = f"LSTM refined (SSE {sse_refined:.1f})" if sse_refined is not None else "LSTM refined"
            ax.plot(tf, sim_f[:, j], "-", color="steelblue", lw=2.2, label=fl)
        if sim_m is not None:
            ml = f"multistart (SSE {sse_multi:.1f})" if sse_multi is not None else "multistart"
            ax.plot(tm, sim_m[:, j], "-.", color="#2ca02c", lw=1.8, label=ml)
        col = _EXP_COLS[j]
        if col in df.columns:
            sub = df[["Time (h)", col]].dropna()
            ax.scatter(sub["Time (h)"], sub[col], color="black", s=35, zorder=5, label="measured")
        ax.set_title(_STATE_NAMES[j]); ax.set_xlabel("time (h)"); ax.set_ylabel("g/L")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ptxt = f"  ({prob_label} {prob:.2f})" if prob is not None else ""
    fig.suptitle(f"{exp_name}  |  model{mid}{ptxt}")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_experiment_fit(df, res, out_path):
    """Overlay measured points vs simulations for the IDENTIFIED (top) mechanism."""
    if "error" in res:
        return
    _plot_mechanism_fit(
        df, res["experiment"], res["predicted_model"], res.get("confidence"),
        res["lstm_params"], res.get("sse_init"),
        res.get("fitted_params"), res.get("sse_fitted"),
        res.get("multistart_params"), res.get("sse_multistart", res.get("multistart_sse")),
        out_path)


def plot_candidate_fits(df, res, out_dir):
    """Write one fit figure per candidate mechanism stored in res['mechanisms']
    (every mechanism kept above --estimate-conf), so secondary mechanisms are
    visualised too, not only the top pick. Returns the list of files written.

    Files: fit_<exp>_M<mid>.png. When there is a single candidate this is the
    same content as fit_<exp>.png (which is still written for the top pick).
    """
    if "error" in res:
        return []
    mechs = res.get("mechanisms", {})
    written = []
    for mid, e in sorted(mechs.items(), key=lambda kv: -kv[1].get("prob", 0)):
        out_path = os.path.join(out_dir, f"fit_{res['experiment']}_M{mid}.png")
        _plot_mechanism_fit(
            df, res["experiment"], int(mid), e.get("prob"),
            e["lstm_params"], e.get("sse_lstm"),
            e.get("refined_params"), e.get("sse_refined"),
            e.get("multistart_params"), e.get("sse_multistart"),
            out_path, prob_label="prob")
        written.append(out_path)
    return written
