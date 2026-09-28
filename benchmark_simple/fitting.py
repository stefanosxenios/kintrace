"""
Parameter-fitting / refinement benchmark for the single-substrate system.

This is the EXP2/EXP3 decision gate: does an LSTM-proposed starting point give
a *materially better* least-squares fit than naive / random / multistart
initialisation?

For a given noisy trajectory and its known operating conditions, we fit the
mechanism's kinetic parameters with scipy.optimize.least_squares (TRF, box
bounds = LHS ranges) from several initialisations and record, for each:

    success       : did it converge to a good fit (final SSE below tol)?
    nfev          : number of residual (=> model simulation) evaluations
    wall_s        : wall-clock seconds
    sse           : final sum of squared residuals (scaled X,S space)
    params        : fitted physical parameters

Initialisation strategies
-------------------------
    lstm           : the Stage-2 predicted parameter vector
    naive          : geometric-ish midpoint of the bounds (a "sensible default")
    random         : a single uniform draw within bounds
    lhs_multistart : best-of-k uniform draws (k local fits, best final SSE);
                     the standard baseline a practitioner would actually use
"""

import time
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares

from kinetic_models.simple_models import get_simple_model, get_param_names
from datagen.param_ranges_simple import get_param_bounds
from datagen.feed_profiles_simple import make_feed_fn, N_FEED_SEG


def _feed_from_condition(cond_row):
    fp = {"t_total": float(cond_row["t_total"]),
          "feed_rates": [float(cond_row[f"feed_rate_{j}"]) for j in range(N_FEED_SEG)]}
    return make_feed_fn(fp)


def simulate_traj(model_id, params, cond_row, n_points):
    """Simulate [X,S,V] for a mechanism given a conditions row. Returns (T,3) or None."""
    rhs = get_simple_model(model_id)
    t_total = float(cond_row["t_total"])
    Sin = float(cond_row["Sin"])
    feed_fn = _feed_from_condition(cond_row)
    t_span = np.linspace(0.0, t_total, n_points)
    y0 = [float(cond_row["X0"]), float(cond_row["S0"]), float(cond_row["V0"])]
    try:
        traj = odeint(rhs, y0, t_span, args=(np.asarray(params), feed_fn, Sin),
                      rtol=1e-6, atol=1e-8, mxstep=5000)
    except Exception:
        return None
    if np.any(~np.isfinite(traj)):
        return None
    return traj


def _residuals(theta, model_id, cond_row, X_obs, S_obs, scaleX, scaleS):
    traj = simulate_traj(model_id, theta, cond_row, len(X_obs))
    if traj is None:
        return np.full(2 * len(X_obs), 1e3)
    rX = (traj[:, 0] - X_obs) / scaleX
    rS = (traj[:, 1] - S_obs) / scaleS
    return np.concatenate([rX, rS])


def generous_bounds(model_id, lb_factor=0.1, ub_factor=5.0):
    """Wide box bounds around the training ranges — lets a fit reach OOD values."""
    lbs, ubs = get_param_bounds(model_id)
    lbs, ubs = np.array(lbs, float), np.array(ubs, float)
    return lbs * lb_factor, ubs * ub_factor


def fit_from_init(model_id, theta0, cond_row, X_obs, S_obs, bounds=None):
    """Run one local least-squares fit from theta0. Returns a result dict.

    bounds : optional (lbs, ubs) arrays. Defaults to the training LHS ranges;
             pass generous_bounds(...) to allow out-of-distribution fits.
    """
    if bounds is None:
        lbs, ubs = get_param_bounds(model_id)
    else:
        lbs, ubs = bounds
    lbs, ubs = np.array(lbs, float), np.array(ubs, float)
    theta0 = np.clip(np.asarray(theta0, float), lbs, ubs)
    scaleX = max(np.std(X_obs), 1e-3)
    scaleS = max(np.std(S_obs), 1e-3)

    # Variable (parameter) scaling: the kinetic parameters span very different
    # magnitudes (e.g. mumax ~0.5 vs Xmax ~40), so without it TRF's trust region
    # is dominated by the large-scale parameters and the step on the small ones
    # underflows -- the fit stalls near its initialisation (the same pathology
    # fixed in the CS2 L-BFGS-B refinement). Setting x_scale to each parameter's
    # box width makes every variable O(1), so the optimiser actually moves.
    x_scale = np.maximum(ubs - lbs, 1e-12)

    t0 = time.perf_counter()
    res = least_squares(
        _residuals, theta0, bounds=(lbs, ubs), method="trf", x_scale=x_scale,
        args=(model_id, cond_row, X_obs, S_obs, scaleX, scaleS),
        max_nfev=200, xtol=1e-8, ftol=1e-8,
    )
    wall = time.perf_counter() - t0
    sse = float(np.sum(res.fun ** 2))
    return dict(params=res.x, sse=sse, nfev=int(res.nfev), wall_s=wall,
                theta0=theta0)


def naive_init(model_id):
    lbs, ubs = get_param_bounds(model_id)
    return 0.5 * (np.array(lbs) + np.array(ubs))


def random_init(model_id, rng):
    lbs, ubs = get_param_bounds(model_id)
    return np.array(lbs) + rng.random(len(lbs)) * (np.array(ubs) - np.array(lbs))


def lhs_multistart(model_id, cond_row, X_obs, S_obs, k=5, rng=None):
    """Best-of-k random restarts. Aggregates nfev and wall time across restarts."""
    rng = rng or np.random.default_rng(0)
    best, tot_nfev, tot_wall = None, 0, 0.0
    for _ in range(k):
        r = fit_from_init(model_id, random_init(model_id, rng), cond_row, X_obs, S_obs)
        tot_nfev += r["nfev"]
        tot_wall += r["wall_s"]
        if best is None or r["sse"] < best["sse"]:
            best = r
    best = dict(best)
    best["nfev"] = tot_nfev
    best["wall_s"] = tot_wall
    return best


def benchmark_one(model_id, cond_row, X_obs, S_obs, lstm_theta,
                  sse_success_tol=None, multistart_k=5, seed=0):
    """
    Fit one trajectory from all initialisation strategies.

    Returns dict strategy -> result dict (with an added 'success' bool).
    """
    rng = np.random.default_rng(seed)
    strategies = {
        "lstm":  fit_from_init(model_id, lstm_theta, cond_row, X_obs, S_obs),
        "naive": fit_from_init(model_id, naive_init(model_id), cond_row, X_obs, S_obs),
        "random": fit_from_init(model_id, random_init(model_id, rng),
                                cond_row, X_obs, S_obs),
        "lhs_multistart": lhs_multistart(model_id, cond_row, X_obs, S_obs,
                                         k=multistart_k, rng=rng),
    }
    # define success relative to the best SSE achieved on this trajectory
    best_sse = min(r["sse"] for r in strategies.values())
    tol = sse_success_tol if sse_success_tol is not None else best_sse * 1.05 + 1e-6
    for r in strategies.values():
        r["success"] = bool(r["sse"] <= tol)
    return strategies
