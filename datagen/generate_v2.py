"""
Generate in silico datasets with varied feed profiles and LHS parameter sampling.

Three dataset representations are produced for each of 15 kinetic models:

  concentrations : [X, S1, S2, P, V, F/V]        shape (N, T, 6)   + 5% noise
  rates          : [mu_obs, q_S1, q_S2, q_P]      shape (N, T, 4)   + 5% noise
  hybrid         : concentrations + rates          shape (N, T, 10)  + 5% noise

Saved to:
  data/generated_v2/concentrations/model{id}.npy  +  params_model{id}.csv
  data/generated_v2/rates/model{id}.npy           +  params_model{id}.csv
  data/generated_v2/hybrid/model{id}.npy          +  params_model{id}.csv

Usage:
  python datagen/generate_v2.py                        # all 15 models, 1 worker
  python datagen/generate_v2.py --workers 14           # parallel on 14 cores
  python datagen/generate_v2.py --models 21 22         # specific models
  python datagen/generate_v2.py --n 100                # quick test
  python datagen/generate_v2.py --n 5000 --workers 14  # full run, parallel
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from scipy.integrate import odeint
import multiprocessing as mp

# Resolve project root so this script can be run from anywhere
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from kinetic_models.base_models import get_model
from kinetic_models.feed_fun import create_feed_rate_function
from datagen.param_ranges_v2 import PARAM_RANGES, get_param_bounds, get_param_names
from datagen.feed_profiles import sample_feed_and_ic, make_exter_df
from datagen.rates_calculator import compute_specific_rates
from datagen.noise import add_gaussian_noise

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_IDS   = list(range(21, 32))   # models 21–31
N_SAMPLES   = 10000
N_TIMEPOINTS = 100
NOISE_PCT   = 0.05
SAVE_DIR    = os.path.join(_ROOT, "data", "generated_v2")
BASE_SEED   = 42

# Sanity-check thresholds for filtering unrealistic simulations
MAX_BIOMASS = 150.0   # g/L
MIN_BIOMASS = -0.5    # g/L (small negative allowed for numerical noise)
MAX_PRODUCT = 500.0   # g/L


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_params_lhs(model_id: int, n: int, seed: int = None) -> np.ndarray:
    """
    Sample n parameter sets using Latin Hypercube Sampling.

    Returns
    -------
    params : (n, n_params) array
    """
    lbs, ubs = get_param_bounds(model_id)
    dim = len(lbs)
    sampler = LatinHypercube(d=dim, seed=seed)
    unit = sampler.random(n)
    return np.array(lbs) + unit * (np.array(ubs) - np.array(lbs))


# ── Simulation ────────────────────────────────────────────────────────────────

def compute_FV(trajectory: np.ndarray, t_span: np.ndarray, exter_df) -> np.ndarray:
    """
    Compute F(t)/V(t) dilution rate at each timepoint.

    Returns
    -------
    FV : (T,) array in h^-1
    """
    feed_func = create_feed_rate_function(exter_df)
    FV = np.array([
        feed_func(t) / 1000.0 / max(float(trajectory[i, 4]), 1e-8)
        for i, t in enumerate(t_span)
    ])
    return FV


def simulate_one(
    model_fn,
    params: np.ndarray,
    exter_df,
    t_total: float,
    ic: dict,
    n_points: int = N_TIMEPOINTS,
):
    """
    Run a single ODE simulation.

    Parameters
    ----------
    model_fn : ODE function
    params   : kinetic parameters
    exter_df : synthetic experiment DataFrame
    t_total  : experiment end time (h)
    ic       : dict with X0, S1_0, S2_0, P0, V0_mL
    n_points : number of output timepoints

    Returns
    -------
    (trajectory, t_span) or (None, None) if simulation failed / unrealistic
    """
    t_span = np.linspace(0.0, t_total, n_points)
    X0 = [
        ic["X0"],
        ic["S1_0"],
        ic["S2_0"],
        ic["P0"],
        ic["V0_mL"] / 1000.0,   # mL → L
    ]

    try:
        result = odeint(
            model_fn, X0, t_span,
            args=(params, exter_df),
            rtol=1e-6, atol=1e-8,
            mxstep=5000,
        )
    except Exception:
        return None, None

    # Filter unrealistic trajectories
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        return None, None
    if result[:, 0].max() > MAX_BIOMASS:
        return None, None
    if result[:, 0].min() < MIN_BIOMASS:
        return None, None
    if result[:, 3].max() > MAX_PRODUCT:
        return None, None

    return result, t_span


# ── Dataset generation for one model ─────────────────────────────────────────

def generate_model_dataset(
    model_id: int,
    n_samples: int = N_SAMPLES,
    n_timepoints: int = N_TIMEPOINTS,
    noise_pct: float = NOISE_PCT,
    save_dir: str = SAVE_DIR,
    seed: int = BASE_SEED,
) -> int:
    """
    Generate all three datasets for a single kinetic model.

    Returns
    -------
    n_success : number of successful simulations saved
    """
    model_fn = get_model(f"model{model_id}")
    param_names = get_param_names(model_id)

    # Sample kinetic parameters via LHS
    params_array = sample_params_lhs(model_id, n_samples, seed=seed)

    # Sample feed profiles and initial conditions via LHS
    feed_ic = sample_feed_and_ic(n_samples, seed=seed + 1000)

    conc_list   = []
    rate_list   = []
    param_list  = []
    failed      = 0

    for i in tqdm(range(n_samples), desc=f"  model{model_id}", ncols=80, leave=False):
        params  = params_array[i]
        fp      = {k: feed_ic[k][i] for k in feed_ic}

        # Build synthetic experiment DataFrame
        exter_df = make_exter_df(fp, idx=i)

        t_total = float(fp["t_total"])
        ic = {
            "X0":    float(fp["X0"]),
            "S1_0":  float(fp["S1_0"]),
            "S2_0":  float(fp["S2_0"]),
            "P0":    float(fp["P0"]),
            "V0_mL": float(fp["V0_mL"]),
        }

        # Simulate
        traj, t_span = simulate_one(model_fn, params, exter_df, t_total, ic, n_timepoints)
        if traj is None:
            failed += 1
            continue

        # F/V vector (T, 1) — included in concentrations and hybrid datasets
        FV = compute_FV(traj, t_span, exter_df).reshape(-1, 1)

        # Specific rates from exact ODE RHS (T, 4)
        rates = compute_specific_rates(traj, t_span, params, model_fn, exter_df)

        # Concentrations array: [X, S1, S2, P, V, F/V]  (T, 6)
        conc = np.concatenate([traj, FV], axis=1)

        # Rates with F/V appended: [mu_obs, q_S1, q_S2, q_P, F/V]  (T, 5)
        rates_with_FV = np.concatenate([rates, FV], axis=1)

        # Add noise
        # concentrations: clip to 0 (can't be negative)
        # rates: no clipping (negative rates are physically meaningful)
        conc_noisy  = add_gaussian_noise(conc,          noise_pct, clip_zero=True,  seed=seed + i)
        rates_noisy = add_gaussian_noise(rates_with_FV, noise_pct, clip_zero=False, seed=seed + i + n_samples)

        conc_list.append(conc_noisy)
        rate_list.append(rates_noisy)
        param_list.append(params)

    n_success = len(conc_list)
    print(f"  model{model_id}: {n_success}/{n_samples} ok  ({failed} failed)")

    if n_success == 0:
        print(f"  WARNING: No successful simulations for model{model_id} — skipping.")
        return 0

    # Stack into arrays
    conc_arr   = np.stack(conc_list,  axis=0)                              # (N, T, 6)
    rate_arr   = np.stack(rate_list,  axis=0)                              # (N, T, 5)
    hybrid_arr = np.concatenate([conc_arr, rate_arr], axis=2)              # (N, T, 11)
    params_df  = pd.DataFrame(param_list, columns=param_names)

    # Save each dataset type
    for dataset_type, arr in [
        ("concentrations", conc_arr),
        ("rates",          rate_arr),
        ("hybrid",         hybrid_arr),
    ]:
        out_dir = os.path.join(save_dir, dataset_type)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, f"model{model_id}.npy"), arr)
        params_df.to_csv(
            os.path.join(out_dir, f"params_model{model_id}.csv"), index=False
        )

    return n_success


# ── Worker function (must be top-level for multiprocessing on macOS) ──────────

def _worker(kwargs):
    """Thin wrapper so pool.map can call generate_model_dataset."""
    model_id = kwargs.pop("model_id")
    t0 = time.time()
    n_ok = generate_model_dataset(model_id=model_id, **kwargs)
    elapsed = time.time() - t0
    print(f"  [model{model_id}] done — {n_ok} samples in {elapsed/60:.1f} min",
          flush=True)
    return model_id, n_ok


# ── Main orchestrator ─────────────────────────────────────────────────────────

def generate_all_datasets(
    model_ids=MODEL_IDS,
    n_samples: int = N_SAMPLES,
    n_timepoints: int = N_TIMEPOINTS,
    noise_pct: float = NOISE_PCT,
    save_dir: str = SAVE_DIR,
    base_seed: int = BASE_SEED,
    n_workers: int = 1,
):
    """
    Generate datasets for all selected models.

    Parameters
    ----------
    n_workers : number of parallel processes.
                Set to os.cpu_count() to use all cores.
                Each worker handles one model independently.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_workers = min(n_workers, len(model_ids))

    print("=" * 60)
    print("BioKin in silico dataset generation (v2)")
    print("=" * 60)
    print(f"  Models:      {model_ids}")
    print(f"  Samples:     {n_samples} per model")
    print(f"  Timepoints:  {n_timepoints}")
    print(f"  Noise:       {noise_pct * 100:.0f}%")
    print(f"  Workers:     {n_workers}")
    print(f"  Save dir:    {save_dir}")
    print("=" * 60)

    # Save metadata
    meta = {
        "model_ids":    str(model_ids),
        "n_samples":    n_samples,
        "n_timepoints": n_timepoints,
        "noise_pct":    noise_pct,
        "base_seed":    base_seed,
        "n_workers":    n_workers,
        "datasets":     "concentrations [X,S1,S2,P,V,F/V], rates [mu,qS1,qS2,qP], hybrid",
    }
    pd.DataFrame([meta]).to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

    # Build kwargs list — each model gets a unique seed
    job_kwargs = [
        {
            "model_id":    mid,
            "n_samples":   n_samples,
            "n_timepoints": n_timepoints,
            "noise_pct":   noise_pct,
            "save_dir":    save_dir,
            "seed":        base_seed + mid * 100,
        }
        for mid in model_ids
    ]

    t_start = time.time()

    if n_workers == 1:
        # Sequential — simple and tqdm-friendly
        results = [_worker(kw) for kw in job_kwargs]
    else:
        # Parallel — one model per core
        # macOS uses 'spawn' by default (safe); set explicitly to be sure
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, job_kwargs)

    total_time = time.time() - t_start

    summary = dict(results)
    print("\n" + "=" * 60)
    print("Summary:")
    total = 0
    for mid in model_ids:
        n = summary.get(mid, 0)
        print(f"  model{mid:>2d}: {n:>5d} samples")
        total += n
    print(f"  Total:   {total} simulations across {len(summary)} models")
    print(f"  Time:    {total_time/60:.1f} min")
    print("=" * 60)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BioKin v2 in silico datasets")
    parser.add_argument("--models",  nargs="+", type=int, default=MODEL_IDS,
                        help="Model IDs to generate (default: all 15)")
    parser.add_argument("--n",       type=int, default=N_SAMPLES,
                        help=f"Samples per model (default: {N_SAMPLES})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1). Use 14 for Mac Studio.")
    parser.add_argument("--seed",    type=int, default=BASE_SEED,
                        help=f"Base random seed (default: {BASE_SEED})")
    parser.add_argument("--outdir",  type=str, default=SAVE_DIR,
                        help="Output directory")
    args = parser.parse_args()

    generate_all_datasets(
        model_ids=args.models,
        n_samples=args.n,
        save_dir=args.outdir,
        base_seed=args.seed,
        n_workers=args.workers,
    )
