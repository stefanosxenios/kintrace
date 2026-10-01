"""Shared building blocks: hybrid Neural ODE training, bounded refinement and BIC ranking.

Every case study uses the same functions:
  * train_node     -- Adam on the range-scaled misfit with a growing horizon (curriculum)
  * refine         -- bounded least squares in normalised coordinates z in [1, 2]
  * identify       -- refine every candidate from several starts, keep the best, rank by BIC
  * initial_state  -- data-based estimate of the initial state
Noise model: either known (sigma given, chi2-BIC) or unknown per channel (sigma=None, profile-likelihood BIC).
"""
import os
os.environ.setdefault("GFORTRAN_UNBUFFERED_ALL", "1")   # so the LSODA messages are silenced, not buffered
import sys
import copy
import contextlib
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import ODEintWarning
from scipy.optimize import least_squares

KINTRACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, KINTRACE)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=ODEintWarning)


@contextlib.contextmanager
def fortran_quiet():
    """LSODA prints its warnings from Fortran straight to file descriptor 1; silence it while it runs."""
    saved, null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(null)
        os.close(saved)


def mlp(n_in, n_out, hidden=16):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def train_node(model, t_obs, Y_obs, n_iter=300, lr=1e-2):
    """Adam on the range-scaled misfit. Y_obs: (time, runs, channels), NaN = not measured.
    The fitting horizon grows from 25 % of the samples to all of them over the first half of training.
    Keeps the best full-horizon state; returns its loss."""
    t, Y = torch.tensor(t_obs), torch.tensor(Y_obs)
    mask, Y = torch.isfinite(Y), torch.nan_to_num(Y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state = np.inf, None
    for it in range(n_iter):
        k = max(4, int(min(1.0, 0.25 + it / (0.5 * n_iter)) * len(t)))
        opt.zero_grad()
        r = (model.simulate(t[:k])[..., :Y.shape[-1]] - Y[:k]) / model.loss_scale
        loss = (r[mask[:k]] ** 2).mean()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        opt.step()
        if k == len(t) and loss.item() < best_loss:
            best_loss, best_state = loss.item(), copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


def initial_state(t, Y, window=1.0, floor=0.02):
    """Initial state per run and channel. Y: (time, ..., channels), NaN allowed.
    If >= 2 samples lie within `window` h of the first sample: intercept of a straight line through them;
    otherwise the first measured value. Floored at `floor` x the channel maximum (a culture started at
    zero never grows)."""
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)
    flat = Y.reshape(len(t), -1)
    out = np.empty(flat.shape[1])
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        tj, yj = t[ok], flat[ok, j]
        near = tj <= tj[0] + window + 1e-9
        if near.sum() >= 2:
            A = np.c_[np.ones(near.sum()), tj[near] - t[0]]
            out[j] = np.linalg.lstsq(A, yj[near], rcond=None)[0][0]
        else:
            out[j] = yj[0]
    out = out.reshape(Y.shape[1:])
    return np.maximum(out, floor * np.nanmax(Y, 0))


def refine(simulate, theta0, lb, ub, t_obs, Y_obs, sigma, max_nfev=300):
    """Bounded least squares in z = 1 + (theta - lb)/(ub - lb) in [1, 2], finite-difference step 1e-3.
    Returns (theta, chi2, nfev); chi2 is the sum of squared sigma-scaled residuals."""
    mask = np.isfinite(Y_obs)
    to_theta = lambda z: lb + (z - 1) * (ub - lb)

    def residuals(z):
        Y = simulate(to_theta(z), t_obs)
        if Y is None or not np.all(np.isfinite(Y[mask])):
            return np.full(mask.sum(), 1e3)
        return ((Y - Y_obs) / sigma)[mask]

    z0 = 1 + (np.clip(theta0, lb, ub) - lb) / (ub - lb)
    r = least_squares(residuals, z0, bounds=(1, 2), diff_step=1e-3, max_nfev=max_nfev)
    return to_theta(r.x), float(np.sum(r.fun ** 2)), r.nfev


def channel_rss(simulate, theta, t_obs, Y_obs):
    """Residual sum of squares per channel (last axis), and number of measured values per channel."""
    Y = simulate(theta, t_obs)
    mask = np.isfinite(Y_obs)
    if Y is None:
        return np.full(Y_obs.shape[-1], np.inf), mask.sum(0)
    r = np.where(mask, Y - np.nan_to_num(Y_obs), 0.0)
    return (r ** 2).sum(0), mask.sum(0)


_JOB = {}   # identify() context for the forked workers (closures cannot be pickled)


def _fit_candidate(m):
    """Refine candidate m from each of its starting points and keep the best fit."""
    J = _JOB
    known, x0, t_obs, Y_obs = J["known"], J["x0"], J["t_obs"], J["Y_obs"]
    lb, ub = J["bounds"](m)
    row, best = dict(mechanism=m, nfev=0), None
    for name, start in J["starts"](m).items():
        theta0, x0s = start if isinstance(start, tuple) else (start, x0)   # a start may carry its own x0
        n = len(theta0)
        sim = lambda th, t: J["simulate"](m, th[:n], t, th[n:])
        s = J["s0"].copy()
        # the initial states stay within +-3 sigma of the data-based estimate; under unknown noise this uses the
        # initial noise scale, so a poor fit cannot widen its own bounds
        x0_lb, x0_ub = np.maximum(x0 - 3 * J["s0"], 0), x0 + 3 * J["s0"]
        p0 = np.clip(np.r_[theta0, x0s], np.r_[lb, x0_lb], np.r_[ub, x0_ub])
        for _ in range(1 if known else J["profile_passes"]):
            theta, chi2, nfev = refine(sim, p0, np.r_[lb, x0_lb], np.r_[ub, x0_ub], t_obs, Y_obs, s)
            row["nfev"] += nfev
            p0 = theta
            if not known:                                  # re-estimate channel SDs from the residuals
                ch = J["channels"]
                rss, cnt = channel_rss(sim, theta, t_obs, Y_obs)
                var = np.array([rss[ch == c].sum() / max(cnt[ch == c].sum(), 1) for c in ch])
                s = np.sqrt(np.maximum(var, 1e-12))
        if known:
            crit = chi2
        else:
            ch = J["channels"]
            rss, cnt = channel_rss(sim, theta, t_obs, Y_obs)
            crit = sum(cnt[ch == c].sum() * np.log(max(rss[ch == c].sum(), 1e-300) / max(cnt[ch == c].sum(), 1))
                       for c in np.unique(ch))
        row[f"crit from {name}"] = crit
        if best is None or crit < best[1]:
            best = (theta, crit, chi2)
    return dict(row, k=n, crit=best[1], chi2=best[2], theta=best[0][:n], x0=best[0][n:])


def identify(candidates, starts, simulate, bounds, x0, t_obs, Y_obs, sigma, channels=None, profile_passes=3,
             workers=1):
    """Refine every candidate from each starting point in starts(m) (dict name -> theta0), keep the best fit,
    rank by BIC.

    Y_obs: (time, columns). x0: initial-state estimate per column; it is refined within +-3 sigma.
    sigma: known noise SD per column (chi2-BIC), or None for unknown noise: then `channels` maps every column
    to a channel, the channel SDs are profiled out by iterative reweighting (`profile_passes` passes), and
    BIC = sum_j n_j ln(RSS_j / n_j) + k ln n. The initial-state bounds then use the starting noise scale
    (5 % of the channel range). workers > 1 refines the candidates in parallel (fork)."""
    import multiprocessing as mp
    x0 = np.asarray(x0, float)
    n_data = int(np.isfinite(Y_obs).sum())
    known = sigma is not None
    if known:
        s0 = np.asarray(sigma, float)
    else:
        channels = np.asarray(channels)
        scale = np.nanmax(Y_obs, 0) - np.nanmin(Y_obs, 0)
        s0 = np.array([0.05 * np.nanmax(scale[channels == c]) for c in channels])
    _JOB.clear()
    _JOB.update(known=known, x0=x0, t_obs=t_obs, Y_obs=Y_obs, s0=s0, channels=channels, starts=starts,
                simulate=simulate, bounds=bounds, profile_passes=profile_passes)
    candidates = list(candidates)
    if workers > 1 and len(candidates) > 1:
        with mp.get_context("fork").Pool(min(workers, len(candidates))) as pool:
            rows = pool.map(_fit_candidate, candidates, chunksize=1)
    else:
        rows = [_fit_candidate(m) for m in candidates]
    return add_bic(pd.DataFrame(rows).set_index("mechanism"), n_data)


def add_bic(df, n_data):
    """BIC = crit + k ln n (crit = chi2 or the profiled -2 log-likelihood), and BIC weights."""
    df = df.copy()
    df["BIC"] = df.crit + df.k * np.log(n_data)
    w = np.exp(-(df.BIC - df.BIC.min()) / 2)
    df["weight"] = w / w.sum()
    return df


def shortlist(df, thr=0.1):
    return [(m, float(w)) for m, w in df.weight.sort_values(ascending=False).items() if w > thr]
