"""
Generate the in-silico "simple benchmark" datasets for the single-substrate
mechanisms (kinetic_models/simple_models.py).

For each of the 4 mechanisms this produces three representations, mirroring the
dual-substrate generate_v2.py layout so the same LSTM code can consume them:

  concentrations : [X, S, V, F/V]        shape (N, T, 4)
  rates          : [mu, qS, F/V]         shape (N, T, 3)
  hybrid         : [X,S,V,F/V,mu,qS]      shape (N, T, 6)  (F/V once)

Gaussian noise (scaled to each feature's dynamic range) is added.

Alongside the arrays we persist, per mechanism:
  params_model{id}.csv      — true kinetic parameters (regression targets)
  conditions_model{id}.csv  — ICs, Sin, t_total, feed rates
                              (needed to RE-SIMULATE during the fitting/refinement
                               benchmark; without these the fit is impossible)

Saved under: data/generated_simple/{concentrations,rates,hybrid}/

Usage
-----
  python datagen/generate_simple.py                 # default N per model
  python datagen/generate_simple.py --n 400         # quick run
  python datagen/generate_simple.py --n 3000 --timepoints 100
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import odeint

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from kinetic_models.simple_models import (
    SIMPLE_MODELS, get_simple_model, get_param_names, compute_mu,
)
from datagen.param_ranges_simple import get_param_bounds
from datagen.feed_profiles_simple import (
    sample_feed_and_ic, make_feed_fn, N_FEED_SEG,
)
from datagen.noise import add_gaussian_noise
from scipy.stats.qmc import LatinHypercube


MODEL_IDS = list(SIMPLE_MODELS.keys())        # [1, 2, 3, 4]
N_SAMPLES = 2000
N_TIMEPOINTS = 100
NOISE_PCT = 0.03
SAVE_DIR = os.path.join(_ROOT, "data", "generated_simple")
BASE_SEED = 42

# Trajectory sanity thresholds
MAX_BIOMASS = 200.0
MIN_BIOMASS = -0.5
MAX_SUBSTRATE = 400.0
MIN_GROWTH = 0.5      # reject near-static trajectories (max X - min X must exceed this)


def sample_params_lhs(model_id: int, n: int, seed: int = None) -> np.ndarray:
    lbs, ubs = get_param_bounds(model_id)
    sampler = LatinHypercube(d=len(lbs), seed=seed)
    unit = sampler.random(n)
    return np.array(lbs) + unit * (np.array(ubs) - np.array(lbs))


def simulate_one(model_id, params, ic, feed_fn, Sin, t_total, n_points):
    """Run one ODE simulation. Returns (traj (T,3), t_span, FV (T,)) or (None,None,None)."""
    rhs = get_simple_model(model_id)
    t_span = np.linspace(0.0, t_total, n_points)
    y0 = [ic["X0"], ic["S0"], ic["V0"]]
    try:
        traj = odeint(rhs, y0, t_span, args=(params, feed_fn, Sin),
                      rtol=1e-6, atol=1e-8, mxstep=5000)
    except Exception:
        return None, None, None

    if np.any(~np.isfinite(traj)):
        return None, None, None
    X, S, V = traj[:, 0], traj[:, 1], traj[:, 2]
    if X.max() > MAX_BIOMASS or X.min() < MIN_BIOMASS or S.max() > MAX_SUBSTRATE:
        return None, None, None
    if (X.max() - X.min()) < MIN_GROWTH:          # reject static / dead cultures
        return None, None, None

    FV = np.array([feed_fn(t) / max(float(V[i]), 1e-9) for i, t in enumerate(t_span)])
    return traj, t_span, FV


def compute_rates(model_id, traj, params):
    """Exact specific rates from the mechanism: mu and specific substrate uptake qS."""
    X, S = traj[:, 0], traj[:, 1]
    p = {n: float(v) for n, v in zip(get_param_names(model_id), params)}
    mu = compute_mu(model_id, X, S, p)          # (T,)
    qS = (1.0 / p["Yxs"]) * mu                  # specific substrate consumption
    return np.stack([mu, qS], axis=1)           # (T, 2)


def generate_model_dataset(model_id, n_samples, n_timepoints, noise_pct, save_dir, seed):
    param_names = get_param_names(model_id)
    params_array = sample_params_lhs(model_id, n_samples, seed=seed)
    feed_ic = sample_feed_and_ic(n_samples, seed=seed + 1000)

    conc_list, rate_list, param_list, cond_list = [], [], [], []
    failed = 0

    for i in range(n_samples):
        params = params_array[i]
        fp = {k: (feed_ic[k][i] if k != "feed_rates" else feed_ic[k][i]) for k in feed_ic}
        feed_fn = make_feed_fn(fp)
        Sin = float(fp["Sin"])
        t_total = float(fp["t_total"])
        ic = {"X0": float(fp["X0"]), "S0": float(fp["S0"]), "V0": float(fp["V0"])}

        traj, t_span, FV = simulate_one(
            model_id, params, ic, feed_fn, Sin, t_total, n_timepoints)
        if traj is None:
            failed += 1
            continue

        FV = FV.reshape(-1, 1)
        rates = compute_rates(model_id, traj, params)          # (T, 2)

        conc = np.concatenate([traj, FV], axis=1)              # [X,S,V,F/V] (T,4)
        rates_fv = np.concatenate([rates, FV], axis=1)         # [mu,qS,F/V] (T,3)

        conc_n = add_gaussian_noise(conc, noise_pct, clip_zero=True, seed=seed + i)
        rate_n = add_gaussian_noise(rates_fv, noise_pct, clip_zero=False,
                                    seed=seed + i + n_samples)

        conc_list.append(conc_n)
        rate_list.append(rate_n)
        param_list.append(params)
        cond_list.append({
            "X0": ic["X0"], "S0": ic["S0"], "V0": ic["V0"],
            "Sin": Sin, "t_total": t_total,
            **{f"feed_rate_{j}": float(fp["feed_rates"][j]) for j in range(N_FEED_SEG)},
        })

    n_ok = len(conc_list)
    print(f"  model{model_id} ({SIMPLE_MODELS[model_id][0]}): {n_ok}/{n_samples} ok "
          f"({failed} rejected)")
    if n_ok == 0:
        return 0

    conc_arr = np.stack(conc_list, axis=0)                     # (N,T,4) [X,S,V,F/V]
    rate_arr = np.stack(rate_list, axis=0)                     # (N,T,3) [mu,qS,F/V]
    # hybrid = concentrations + rate channels WITHOUT re-duplicating F/V
    #        = [X, S, V, F/V, mu, qS]  (T,6)
    hybrid_arr = np.concatenate([conc_arr, rate_arr[:, :, :2]], axis=2)
    params_df = pd.DataFrame(param_list, columns=param_names)
    cond_df = pd.DataFrame(cond_list)

    for dtype, arr in [("concentrations", conc_arr), ("rates", rate_arr),
                       ("hybrid", hybrid_arr)]:
        out_dir = os.path.join(save_dir, dtype)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"model{model_id}.npy"), arr)
        params_df.to_csv(os.path.join(out_dir, f"params_model{model_id}.csv"), index=False)
        cond_df.to_csv(os.path.join(out_dir, f"conditions_model{model_id}.csv"), index=False)

    return n_ok


def generate_all(model_ids=MODEL_IDS, n_samples=N_SAMPLES, n_timepoints=N_TIMEPOINTS,
                 noise_pct=NOISE_PCT, save_dir=SAVE_DIR, base_seed=BASE_SEED):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 60)
    print("BioKin single-substrate benchmark generation")
    print(f"  models={model_ids}  n={n_samples}  T={n_timepoints}  noise={noise_pct*100:.0f}%")
    print("=" * 60)
    meta = {"model_ids": str(model_ids), "n_samples": n_samples,
            "n_timepoints": n_timepoints, "noise_pct": noise_pct,
            "features": "conc[X,S,V,F/V](4) rates[mu,qS,F/V](3) hybrid[X,S,V,F/V,mu,qS](6)"}
    pd.DataFrame([meta]).to_csv(os.path.join(save_dir, "metadata.csv"), index=False)
    total = 0
    for mid in model_ids:
        total += generate_model_dataset(mid, n_samples, n_timepoints, noise_pct,
                                        save_dir, seed=base_seed + mid)
    print(f"Done. {total} trajectories across {len(model_ids)} mechanisms.")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--timepoints", type=int, default=N_TIMEPOINTS)
    ap.add_argument("--noise", type=float, default=NOISE_PCT)
    ap.add_argument("--models", type=int, nargs="+", default=MODEL_IDS)
    ap.add_argument("--save-dir", type=str, default=SAVE_DIR)
    args = ap.parse_args()
    generate_all(model_ids=args.models, n_samples=args.n, n_timepoints=args.timepoints,
                 noise_pct=args.noise, save_dir=args.save_dir)
