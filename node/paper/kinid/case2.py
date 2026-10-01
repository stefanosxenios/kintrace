"""Case study 2: dual-substrate xylitol bioconversion, state [X, S1, S2, P] with known dilution D(t) = F(t)/V(t).

dX/dt  = (mu1 - kd - D) X
dS1/dt = -mu1 X / YXS1 + D (S1f - S1)        S1: glucose equivalent (glucose + 1.2 ethanol)
dS2/dt = -mu2 X / YPS2 + D (S2f - S2)        S2: xylose
dP/dt  =  mu2 X - D P                        P : xylitol
Library: models 21-31 (Monod baseline + product/substrate inhibition on growth or conversion + cell death).
"""
import time
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import odeint as scipy_odeint
from scipy.optimize import least_squares
from torchdiffeq import odeint

from .core import fortran_quiet, mlp, train_node, initial_state, identify, add_bic
from .sr import exp_, sr_mu, fit_law, sr_candidates, sr_formula

from datagen.param_ranges_v2 import get_param_names, get_param_bounds
from kinetic_models.feed_fun import create_feed_rate_function

CH = ["X", "S1", "S2", "P"]
MECHS = list(range(21, 32))


# ── library ──────────────────────────────────────────────────────────────────
def library_rates(p, S1, S2, P):
    mu1 = p["mumax1"] * S1 / (p["KS1"] + S1 + (S2 ** 2 / p["KSI1"] if "KSI1" in p else 0))
    mu1 = mu1 / (1 + P / p["KPI1"]) if "KPI1" in p else mu1
    mu2 = p["mumax2"] * S2 / (p["KP"] + S2 + (S2 ** 2 / p["KSI2"] if "KSI2" in p else 0))
    mu2 = mu2 / (1 + P / p["KPI2"]) if "KPI2" in p else mu2
    return mu1, mu2


def lib_entry(m):
    names = list(get_param_names(m))
    lo, hi = [np.array(b, float) for b in get_param_bounds(m)]
    return dict(names=names, rates=lambda S1, S2, P, p: library_rates(p, S1, S2, P), death="kd" in names,
                centre=0.5 * (lo + hi), bounds=(0.1 * lo, 5 * hi), mech=m, ranges=(lo, hi))


LIBRARY = {f"M{m}": lib_entry(m) for m in MECHS}
TERMS_OF = {21: [], 22: ["KPI1"], 23: ["KSI1"], 24: ["KSI1", "KPI1"], 25: ["kd", "KPI1"], 26: ["kd", "KSI1"],
            27: ["kd", "KSI1", "KPI1"], 28: ["KPI2"], 29: ["KSI2"], 30: ["KSI2", "KPI2"], 31: ["kd", "KSI2", "KPI2"]}


# ── simulation ───────────────────────────────────────────────────────────────
def rhs(y, t, rates, p, run):
    X, S1, S2, P = np.maximum(y, 0.0)
    D = run["D"](t)
    mu1, mu2 = rates(S1, S2, P, p)
    return [(mu1 - p.get("kd", 0.0) - D) * X, -mu1 * X / p["YXS1"] + D * (run["S1f"] - S1),
            -mu2 * X / p["YPS2"] + D * (run["S2f"] - S2), mu2 * X - D * P]


def simulate_runs(rates, p, t, x0, runs):
    """States of every run side by side on grid t, (T, 4 runs); NaN after a run's last sample.
    None if an integration fails or blows up."""
    t = np.asarray(t, float)
    out = []
    for i, run in enumerate(runs):
        keep = t <= run["t_end"] + 1e-9
        full = np.full((len(t), 4), np.nan)
        try:
            with fortran_quiet():
                y = scipy_odeint(rhs, x0[4 * i:4 * i + 4], t[keep], args=(rates, p, run), rtol=1e-6, atol=1e-8,
                                 mxstep=5000)
        except Exception:
            return None
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e6:
            return None
        full[keep] = y
        out.append(full)
    return np.hstack(out)


def simulate(entry, theta, t, x0, runs):
    return simulate_runs(entry["rates"], dict(zip(entry["names"], theta)), t, x0, runs)


def union_grid(runs):
    """Common time grid (union of sampling times) and data on it, (T, runs, 4) with NaN where not sampled."""
    t = np.unique(np.concatenate([r["t"] for r in runs]))
    Y = np.full((len(t), len(runs), 4), np.nan)
    for i, r in enumerate(runs):
        Y[np.searchsorted(t, r["t"]), i] = r["Y"]
    return t, Y


# ── data ─────────────────────────────────────────────────────────────────────
def _run_from(name, t, Y, F_mLh, V_L, S1f, S2f, extra=None):
    run = dict(name=name, t=np.asarray(t, float), Y=np.asarray(Y, float), S1f=float(S1f), S2f=float(S2f),
               t_end=float(np.max(t)))
    tv, vv = np.asarray(V_L[0], float), np.asarray(V_L[1], float)
    run["F"] = lambda tt: F_mLh(tt) / 1000.0
    run["V"] = lambda tt: float(np.interp(tt, tv, vv))
    run["D"] = lambda tt: run["F"](tt) / run["V"](tt)
    run["V_table"] = (tv, vv)
    run.update(extra or {})
    return run


def load_real(path, sheet="S. cerevisiae (plasmid)", labels=("BC12", "BC15", "BC18", "BC19"), ethanol_factor=1.2):
    """Fed-batch runs from the workbook: biomass (DW), glucose equivalent, xylose, xylitol; feed and volume."""
    df = pd.read_excel(path, sheet_name=sheet, header=1)
    df["label"] = df["Sample name"].astype(str).str.split("_").str[0]
    runs = {}
    for lab in labels:
        g = df[df["label"] == lab].copy()
        feed = g[g["Sample name"] == f"{lab}_Feed"].iloc[0]
        eth_f = feed["Ethanol"] if np.isfinite(feed["Ethanol"]) else 0.0
        S1f, S2f = float(feed["Glucose"]) + ethanol_factor * float(eth_f), float(feed["Xylose"])
        g = g[g["Sample name"] != f"{lab}_Feed"].copy()
        g["Time (h)"] = pd.to_numeric(g["Time (h)"], errors="coerce")
        g = g[g["Time (h)"] != 0].copy()                    # medium sample taken before inoculation
        g["Time (h)"] = g["Time (h)"] - g["Time (h)"].min()
        g = g.reset_index(drop=True)
        glu_eq = g["Glucose"] + ethanol_factor * g["Ethanol"]
        Y = np.c_[g["DW (g/L)"], glu_eq, g["Xylose"], g["Xylitol"]].astype(float)
        F = create_feed_rate_function(g[["Time (h)", "Feed rate (mL/h)", "Feed added (mL)"]])
        V = (g["Time (h)"].to_numpy(float), g["Reactor volume (mL)"].to_numpy(float) / 1000.0)
        runs[lab] = _run_from(lab, g["Time (h)"], Y, F, V, S1f, S2f,
                              extra=dict(sem_X=g["sem"].to_numpy(float), raw=g))
    return runs


def make_virtual(base, entry, theta, noise=0.05, seed=0, name=None):
    """Virtual run with the feed, volume, initial state and sampling schedule (incl. missing values) of a real
    run `base`, kinetics `entry` with parameters `theta`, and Gaussian noise of `noise` x each channel's range."""
    x0 = initial_state(base["t"], base["Y"][:, None, :]).ravel()
    truth = simulate(entry, theta, base["t"], x0, [base])
    rng = np.random.default_rng(seed)
    sd = noise * (np.nanmax(truth, 0) - np.nanmin(truth, 0))
    Y = np.clip(truth + rng.normal(0, sd, truth.shape), 0, None)
    Y[~np.isfinite(base["Y"])] = np.nan
    run = dict(base)
    run.update(name=name or f"{base['name']}-virtual", Y=Y, truth=truth, sigma=sd, x0_true=x0)
    return run


# ── hybrid Neural ODE ────────────────────────────────────────────────────────
class HybridXyl(nn.Module):
    """[mu1, mu2] = softplus(NN(S1, S2, P)) * [S1/(S1+K1), S2/(S2+K2)], shared by all runs; YXS1, YPS2, kd trainable."""
    step = 1.0

    def __init__(self, runs, t_grid, Y):
        super().__init__()
        self.Dfns = [r["D"] for r in runs]
        self.scale = torch.tensor(np.nanmax(Y, (0, 1)))
        rng = np.nanmax(Y, 0) - np.nanmin(Y, 0)
        self.loss_scale = torch.tensor(np.maximum(rng, 1e-3 * np.nanmax(Y, (0, 1))))
        self.y_first = torch.tensor(initial_state(t_grid, Y))
        self.dy0 = nn.Parameter(torch.zeros_like(self.y_first))
        self.K = torch.tensor([0.25 * np.median(np.nanmax(Y[..., 1], 0)), 0.1 * np.median(np.nanmax(Y[..., 2], 0))])
        self.Sf = torch.tensor([[r["S1f"], r["S2f"]] for r in runs])
        self.net = mlp(3, 2)
        self.log_Y = nn.Parameter(torch.log(torch.tensor([0.5, 1.0])))
        self.log_kd = nn.Parameter(torch.tensor(np.log(0.01)))

    def rates(self, S1, S2, P):
        S = torch.stack([S1, S2], -1).clamp(min=0)
        g = nn.functional.softplus(self.net(torch.stack([S1, S2, P], -1).clamp(min=0) / self.scale[1:]))
        mu = g * S / (S + self.K)
        return mu[..., 0], mu[..., 1]

    def forward(self, t, y):
        X, S1, S2, P = y.unbind(-1)
        D = torch.tensor([d(float(t)) for d in self.Dfns])
        mu1, mu2 = self.rates(S1, S2, P)
        Y1, Y2 = torch.exp(self.log_Y)
        return torch.stack([(mu1 - torch.exp(self.log_kd) - D) * X, -mu1 * X / Y1 + D * (self.Sf[:, 0] - S1),
                            -mu2 * X / Y2 + D * (self.Sf[:, 1] - S2), mu2 * X - D * P], -1)

    def simulate(self, t):
        x0 = (self.y_first + 0.05 * self.scale * self.dy0).clamp(min=0)
        return odeint(self, x0, torch.as_tensor(t), method="rk4", options=dict(step_size=self.step))


def fit_node(runs, n_iter=200, lr=1e-2, torch_seed=0, n_dense=141):
    t, Y = union_grid(runs)
    torch.manual_seed(torch_seed)
    model = HybridXyl(runs, t, Y)
    t0 = time.time()
    loss = train_node(model, t, Y, n_iter=n_iter, lr=lr)
    td = np.linspace(0, t.max(), n_dense)
    with torch.no_grad():
        yd = model.simulate(td)
        mu1, mu2 = model.rates(yd[..., 1], yd[..., 2], yd[..., 3])
    yd, mu1, mu2 = yd.numpy(), mu1.numpy(), mu2.numpy()
    keep = [td <= r["t_end"] + 1e-9 for r in runs]
    cat = lambda a: np.concatenate([a[k, i] for i, k in enumerate(keep)])
    out = dict(loss=loss, seconds=time.time() - t0, t_dense=td, y_dense=yd, mu1_dense=mu1, mu2_dense=mu2, keep=keep,
               S1=np.clip(cat(yd[..., 1]), 0, None), S2=np.clip(cat(yd[..., 2]), 0, None),
               P=np.clip(cat(yd[..., 3]), 0, None), X=cat(yd[..., 0]), mu1=cat(mu1), mu2=cat(mu2),
               yields=torch.exp(model.log_Y).tolist(), kd=torch.exp(model.log_kd).item())
    with torch.no_grad():
        out["y_fit"] = model.simulate(t).numpy()
    out["t_grid"] = t
    return out


# ── library selection ────────────────────────────────────────────────────────
def match(entry, dense):
    """Starting point: the candidate's mu1, mu1 - kd and mu2 fitted to the learned rates; yields from the NODE."""
    names = entry["names"]
    kin = [i for i, n in enumerate(names) if n not in ("YXS1", "YPS2")]
    lb, ub = entry["bounds"]
    s1, s2 = dense["mu1"].max() + 1e-9, dense["mu2"].max() + 1e-9

    def res(th):
        p = dict(zip([names[i] for i in kin], th))
        mu1, mu2 = library_rates(p, dense["S1"], dense["S2"], dense["P"])
        kd = p.get("kd", 0.0)
        return np.concatenate([(mu1 - dense["mu1"]) / s1, (mu1 - kd - dense["mu1"] + dense["kd"]) / s1,
                               (mu2 - dense["mu2"]) / s2])

    fit = least_squares(res, entry["centre"][kin], bounds=(lb[kin], ub[kin]), x_scale=(ub - lb)[kin])
    theta = np.empty(len(names))
    theta[kin] = fit.x
    theta[names.index("YXS1")], theta[names.index("YPS2")] = dense["yields"]
    return np.clip(theta, lb, ub)


def data_matrix(runs):
    t, Y = union_grid(runs)
    return t, Y.reshape(len(t), -1), Y


INACTIVE_AT = {"kd": "lb", "KSI1": "ub", "KPI1": "ub", "KSI2": "ub", "KPI2": "ub"}


def nested_in(library):
    """For each library mechanism, the library mechanisms whose parameters are a proper subset of its own."""
    names = {m: set(e["names"]) for m, e in library.items() if "mech" in e}
    return {m: [s for s in names if names[s] < names[m]] for m in names}


def nested_start(entry, sub, theta_sub):
    """Start for `entry` from the fit of a mechanism nested in it: shared parameters copied, the additional
    inhibition constants at their upper bound and the death rate at its lower bound (terms inactive)."""
    lb, ub = entry["bounds"]
    p = dict(zip(sub["names"], theta_sub))
    return np.array([p[n] if n in p else (lb[i] if INACTIVE_AT[n] == "lb" else ub[i])
                     for i, n in enumerate(entry["names"])])


def select(library, runs, dense, sigma="known", starts=None, workers=1, nested_rounds=3):
    """sigma='known': chi2-BIC with each run's sigma; sigma=None: per-channel SDs profiled out.
    Library mechanisms that fit worse than a mechanism nested in them are refined again from that fit
    (up to `nested_rounds` rounds), so that the fits respect the nesting."""
    t, Yf, Y3 = data_matrix(runs)
    x0 = initial_state(t, Y3).ravel()
    starts = starts or (lambda m: {"NODE": match(library[m], dense), "centre": library[m]["centre"]})
    sig = np.concatenate([r["sigma"] for r in runs]) if sigma == "known" else None
    run = lambda cands, st: identify(cands, starts=st, simulate=lambda m, th, tt, x: simulate(library[m], th, tt, x, runs),
                                     bounds=lambda m: library[m]["bounds"], x0=x0, t_obs=t, Y_obs=Yf, sigma=sig,
                                     channels=np.tile(np.arange(4), len(runs)), workers=workers)
    df = run(list(library), starts)
    n_data = int(np.isfinite(Yf).sum())
    nest = nested_in(library)
    for _ in range(nested_rounds):
        redo = {}
        for m, subs in nest.items():
            if subs:
                s = min(subs, key=lambda x: df.loc[x, "crit"])
                if df.loc[s, "crit"] < df.loc[m, "crit"] - 1e-6:
                    redo[m] = (nested_start(library[m], library[s], df.loc[s, "theta"]), df.loc[s, "x0"])
        if not redo:
            break
        df2 = run(list(redo), lambda m: {"nested": redo[m]})
        df = df.astype({c: object for c in ("theta", "x0")})
        for m in redo:
            df.loc[m, "crit from nested"] = df2.loc[m, "crit"]
            df.loc[m, "nfev"] += df2.loc[m, "nfev"]
            if df2.loc[m, "crit"] < df.loc[m, "crit"]:
                for c in ("crit", "chi2", "theta", "x0"):
                    df.at[m, c] = df2.at[m, c]
        df = add_bic(df, n_data)
    return df


# ── symbolic regression on mu1 and mu2 ───────────────────────────────────────
SR_MU1 = dict(lhs="S1", vars=["S1", "S2", "P"], terms={
    "S1": lambda v, q: v["S1"],
    "mu1": lambda v, q: v["S1"] ** 0,
    "mu1*S2^2": lambda v, q: v["S2"] ** 2,
    "mu1*P": lambda v, q: v["P"],
    "mu1*S1*P": lambda v, q: v["S1"] * v["P"],
})
SR_MU2 = dict(lhs="S2", vars=["S2", "P"], Kvar="P", terms={
    "S2": lambda v, q: v["S2"],
    "S2*exp(-P/K)": lambda v, q: v["S2"] * exp_(-v["P"] / q["K"]),
    "mu2": lambda v, q: v["S2"] ** 0,
    "mu2*S2^2": lambda v, q: v["S2"] ** 2,
    "mu2*P": lambda v, q: v["P"],
    "mu2*S2*P": lambda v, q: v["S2"] * v["P"],
    "mu2*S2^2*P": lambda v, q: v["S2"] ** 2 * v["P"],
})


def _block(law, tag):
    k = len(law["terms"])
    names = [f"{tag}{i + 1}" for i in range(k)] + ([f"K_{tag}"] if law["q"] else [])
    vals = list(law["coef"]) + ([law["q"]["K"]] if law["q"] else [])
    return names, vals


def sr_entry(law1, law2, death, yields, kd):
    n1, v1 = _block(law1, "a")
    n2, v2 = _block(law2, "b")
    names = n1 + n2 + (["kd"] if death else []) + ["YXS1", "YPS2"]
    start = np.r_[v1, v2, [kd] if death else [], yields]
    lo, hi = np.minimum(0.1 * start, 10 * start), np.maximum(0.1 * start, 10 * start)
    lo[-2:], hi[-2:] = 0.03, 5.0
    if death:
        lo[-3], hi[-3] = 5e-4, 0.5

    def unpack(p, law, tag):
        k = len(law["terms"])
        return dict(terms=law["terms"], coef=[p[f"{tag}{i + 1}"] for i in range(k)],
                    q={"K": p[f"K_{tag}"]} if law["q"] else {})

    def rates(S1, S2, P, p):
        v = {"S1": S1, "S2": S2, "P": P}
        return sr_mu(v, SR_MU1, unpack(p, law1, "a")), sr_mu(v, SR_MU2, unpack(p, law2, "b"))

    return dict(names=names, rates=rates, death=death, centre=start, bounds=(lo, hi), laws=(law1, law2),
                unpack=unpack)


def laws_from_theta(entry, theta):
    p = dict(zip(entry["names"], theta))
    return entry["unpack"](p, entry["laws"][0], "a"), entry["unpack"](p, entry["laws"][1], "b")


def sr_library(dense, per_size=1, max1=3, max2=6):
    v = {"S1": dense["S1"], "S2": dense["S2"], "P": dense["P"]}
    c1 = sr_candidates(v, dense["mu1"], SR_MU1, max_terms=max1, per_size=per_size)
    c2 = sr_candidates(v, dense["mu2"], SR_MU2, max_terms=max2, per_size=per_size)
    lib = {}
    for (l1, _), (l2, _) in itertools.product(c1, c2):
        for death in (False, True):
            lib[f"SR: mu1[{' + '.join(l1['terms'])}] mu2[{' + '.join(l2['terms'])}]{' +kd' if death else ''}"] = \
                sr_entry(l1, l2, death, dense["yields"], dense["kd"])
    cands = dict(mu1=[(l, float(e), str(sr_formula(SR_MU1, l))) for l, e in c1],
                 mu2=[(l, float(e), str(sr_formula(SR_MU2, l))) for l, e in c2])
    return lib, cands


def summarise(df, entries):
    out = {}
    for m, row in df.iterrows():
        e = entries[m]
        rec = dict(k=int(row.k), crit=float(row.crit), chi2=float(row.chi2), BIC=float(row.BIC),
                   weight=float(row.weight), theta=np.asarray(row.theta, float), x0=np.asarray(row.x0, float),
                   library=m in LIBRARY)
        if "laws" in e:
            l1, l2 = laws_from_theta(e, row.theta)
            rec.update(laws=(l1, l2), formula=(str(sr_formula(SR_MU1, l1)), str(sr_formula(SR_MU2, l2))))
        out[m] = rec
    return out


def analyse(runs, sigma="known", with_sr=True, per_size=1, workers=1):
    out, t0 = {}, time.time()
    dense = fit_node(runs)
    out["node"] = dense
    sel = select(LIBRARY, runs, dense, sigma=sigma, workers=workers)
    out["library"] = summarise(sel, LIBRARY)
    t, Yf, _ = data_matrix(runs)
    out["n"] = int(np.isfinite(Yf).sum())
    if with_sr:
        lib, cands = sr_library(dense, per_size=per_size)
        out["sr_candidates"] = cands
        sr_sel = select(lib, runs, dense, sigma=sigma, starts=lambda m: {"SR": lib[m]["centre"]},
                        workers=workers)
        out["all"] = summarise(add_bic(pd.concat([sel, sr_sel]), out["n"]), {**LIBRARY, **lib})
    out["seconds"] = time.time() - t0
    return out
