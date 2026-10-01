"""Case study 1: single-substrate fed-batch, state [X, S, V].

dX/dt = (mu - kd - D) X,  dS/dt = -mu X / Yxs + D (Sin - S),  dV/dt = F,  D = F/V.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import odeint as scipy_odeint
from scipy.optimize import least_squares
from torchdiffeq import odeint

from .core import (fortran_quiet, mlp, train_node, initial_state, refine, identify, add_bic)
from .sr import exp_, sr_mu, fit_law, sr_candidates, sr_formula

from datagen.param_ranges_simple import PARAM_RANGES_SIMPLE
from datagen.feed_profiles_simple import make_feed_fn
from datagen.noise import add_gaussian_noise

# ── library ──────────────────────────────────────────────────────────────────
RATE_LAWS = {   # name: (mu(X, S, p), kinetic parameter names)
    "monod":       (lambda X, S, p: p["mumax"] * S / (p["Ks"] + S), ["mumax", "Ks"]),
    "contois":     (lambda X, S, p: p["mumax"] * S / (p["Kc"] * X + S), ["mumax", "Kc"]),
    "haldane":     (lambda X, S, p: p["mumax"] * S / (p["Ks"] + S + S ** 2 / p["KI"]), ["mumax", "Ks", "KI"]),
    "biomass_inh": (lambda X, S, p: p["mumax"] * S / (p["Ks"] + S) * np.maximum(1 - X / p["Xmax"], 0),
                    ["mumax", "Ks", "Xmax"]),
    "tessier":     (lambda X, S, p: p["mumax"] * (1 - np.exp(-S / p["Ks"])), ["mumax", "Ks"]),
    "moser":       (lambda X, S, p: p["mumax"] * S ** p["n"] / (p["Ks"] + S ** p["n"]), ["mumax", "Ks", "n"]),
    "aiba":        (lambda X, S, p: p["mumax"] * S / (p["Ks"] + S) * np.exp(-S / p["KI"]), ["mumax", "Ks", "KI"]),
}
RANGES = {**PARAM_RANGES_SIMPLE[3], **PARAM_RANGES_SIMPLE[2], **PARAM_RANGES_SIMPLE[4],
          "kd": (0.005, 0.1), "n": (1.0, 3.0)}
RANGES_BY_LAW = {"moser": {"Ks": (0.1, 25.0)}, "aiba": {"KI": (5.0, 50.0)}}


def library_entry(law, death):
    mu_fn, kin = RATE_LAWS[law]
    names = kin + (["kd"] if death else []) + ["Yxs"]
    r = {**RANGES, **RANGES_BY_LAW.get(law, {})}
    lo, hi = np.array([r[n][0] for n in names]), np.array([r[n][1] for n in names])
    return dict(mu=mu_fn, names=names, death=death, centre=0.5 * (lo + hi), bounds=(0.1 * lo, 5 * hi))


LIBRARY = {}
for _law in ["monod", "contois", "haldane", "biomass_inh"]:
    LIBRARY[_law], LIBRARY[_law + "+kd"] = library_entry(_law, False), library_entry(_law, True)
for _law in ["tessier", "moser", "aiba"]:
    LIBRARY[_law] = library_entry(_law, False)
LIB4 = ["monod", "contois", "haldane", "biomass_inh"]

# ── experiments ──────────────────────────────────────────────────────────────
DESIGN = {
    "a": dict(X0=1.0, S0=30.0, V0=1.5, Sin=120.0, t_total=24.0, feed_rates=[0.0, 0.02, 0.05, 0.08]),
    "b": dict(X0=4.0, S0=5.0, V0=1.5, Sin=120.0, t_total=24.0, feed_rates=[0.03, 0.03, 0.06, 0.06]),
    "c": dict(X0=0.5, S0=60.0, V0=1.5, Sin=120.0, t_total=24.0, feed_rates=[0.0, 0.0, 0.03, 0.06]),
    "d": dict(X0=2.0, S0=20.0, V0=1.5, Sin=150.0, t_total=24.0, feed_rates=[0.0, 0.03, 0.03, 0.06]),  # later run
}
for _d in DESIGN.values():
    _d["feed"] = make_feed_fn(_d)
N_OBS, NOISE = 49, 0.03
T_OBS = np.linspace(0, 24.0, N_OBS)
T_DENSE = np.linspace(0, 24.0, 97)
TRUE = {
    "monod": [0.5, 2.0, 0.5],             "monod+kd": [0.5, 2.0, 0.04, 0.5],
    "contois": [0.5, 1.0, 0.5],           "contois+kd": [0.5, 1.0, 0.04, 0.5],
    "haldane": [0.6, 1.0, 8.0, 0.5],      "haldane+kd": [0.6, 1.0, 8.0, 0.04, 0.5],
    "biomass_inh": [0.5, 1.0, 20.0, 0.5], "biomass_inh+kd": [0.5, 1.0, 20.0, 0.04, 0.5],
    "tessier": [0.5, 3.0, 0.5],           "moser": [0.5, 16.0, 2.0, 0.5],        "aiba": [0.6, 1.0, 30.0, 0.5],
}
HIDDEN_P = dict(mumax=0.6, Ks=1.0, KI=20.0, Xmax=30.0, Yxs=0.5)
EXPERIMENTS = list(TRUE) + ["hidden"]


def hidden_mu(X, S, p):
    return RATE_LAWS["haldane"][0](X, S, p) * np.maximum(1 - X / p["Xmax"], 0)


def truth_of(name):
    if name == "hidden":
        return hidden_mu, HIDDEN_P
    return LIBRARY[name]["mu"], dict(zip(LIBRARY[name]["names"], TRUE[name]))


def rhs(y, t, mu_fn, p, feed, Sin):
    X, S, V = max(y[0], 0.0), max(y[1], 0.0), y[2]
    F = feed(t)
    D = F / V
    mu = mu_fn(X, S, p)
    return [(mu - p.get("kd", 0.0) - D) * X, -mu * X / p["Yxs"] + D * (Sin - S), F]


def simulate_runs(mu_fn, p, t, x0, runs):
    """[X, S] of every run side by side, (T, 2 runs); None if an integration fails or blows up."""
    out = []
    for i, run in enumerate(runs):
        try:
            with fortran_quiet():
                y = scipy_odeint(rhs, [*x0[2 * i:2 * i + 2], run["V0"]], t, args=(mu_fn, p, run["feed"], run["Sin"]),
                                 rtol=1e-6, atol=1e-8, mxstep=5000)
        except Exception:
            return None
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e6:
            return None
        out.append(y[:, :2])
    return np.hstack(out)


def simulate(entry, theta, t, x0, runs):
    return simulate_runs(entry["mu"], dict(zip(entry["names"], theta)), t, x0, runs)


def make_runs(name, seed, keys=("a", "b", "c")):
    """Noisy runs of experiment `name` for noise realisation `seed`."""
    mu_fn, p = truth_of(name)
    idx = EXPERIMENTS.index(name)
    runs = []
    for key in keys:
        d = DESIGN[key]
        truth = simulate_runs(mu_fn, p, T_OBS, [d["X0"], d["S0"]], [d])
        Y = add_gaussian_noise(truth, NOISE, clip_zero=True, seed=100000 * seed + 100 * idx + "abcd".index(key))
        runs.append(dict(d, key=key, truth=truth, Y=Y, sigma=NOISE * (Y.max(0) - Y.min(0))))
    return runs


# ── hybrid Neural ODE ────────────────────────────────────────────────────────
class HybridMulti(nn.Module):
    """Mass balances above for several runs at once; mu(X, S) = softplus(NN(X, S)) * S/(S + K), shared."""
    step = 0.25

    def __init__(self, runs, factor=True):
        super().__init__()
        Y = np.stack([r["Y"] for r in runs], 1)                  # (T, runs, 2)
        self.runs, self.factor = runs, factor
        self.scale = torch.tensor(Y.max((0, 1)))
        self.loss_scale = torch.tensor(Y.max(0) - Y.min(0))
        self.y_first = torch.tensor(initial_state(T_OBS, Y))
        self.K = 0.1 * float(np.median(Y[..., 1].max(0)))
        self.dy0 = nn.Parameter(torch.zeros_like(self.y_first))
        self.V0 = torch.tensor([r["V0"] for r in runs])
        self.Sin = torch.tensor([r["Sin"] for r in runs])
        self.net = mlp(2, 1)
        self.log_Yxs = nn.Parameter(torch.tensor(np.log(0.5)))
        self.log_kd = nn.Parameter(torch.tensor(np.log(0.01)))

    def mu(self, X, S):
        S = S.clamp(min=0)
        g = nn.functional.softplus(self.net(torch.stack([X, S], -1) / self.scale)).squeeze(-1)
        return g * S / (S + self.K) if self.factor else g

    def forward(self, t, y):
        X, S, V = y.unbind(-1)
        F = torch.tensor([r["feed"](float(t)) for r in self.runs])
        D = F / V
        mu = self.mu(X, S)
        return torch.stack([(mu - torch.exp(self.log_kd) - D) * X,
                            -mu * X / torch.exp(self.log_Yxs) + D * (self.Sin - S), F], -1)

    def simulate(self, t):
        x0 = (self.y_first + 0.05 * self.scale * self.dy0).clamp(min=0)
        y0 = torch.cat([x0, self.V0[:, None]], -1)
        return odeint(self, y0, torch.as_tensor(t), method="rk4", options=dict(step_size=self.step))


def fit_node(runs, factor=True, torch_seed=0):
    torch.manual_seed(torch_seed)
    model = HybridMulti(runs, factor=factor)
    t0 = time.time()
    loss = train_node(model, T_OBS, np.stack([r["Y"] for r in runs], 1))
    with torch.no_grad():
        yd = model.simulate(T_DENSE)
        mu = model.mu(yd[..., 0], yd[..., 1])
    yd, mu = yd.numpy(), mu.numpy()
    return dict(loss=loss, X=yd[..., 0].T.ravel(), S=np.clip(yd[..., 1], 0, None).T.ravel(), mu=mu.T.ravel(),
                Xr=yd[..., 0].T, Sr=yd[..., 1].T, mur=mu.T,
                Yxs=torch.exp(model.log_Yxs).item(), kd=torch.exp(model.log_kd).item(), seconds=time.time() - t0)


# ── library selection ────────────────────────────────────────────────────────
def match(entry, dense):
    """Starting point: the candidate's rate law fitted to the learned rate; Yxs and kd from the Neural ODE."""
    lb, ub = entry["bounds"]
    kin = [n for n in entry["names"] if n not in ("kd", "Yxs")]
    k = len(kin)
    res = lambda th: entry["mu"](dense["X"], dense["S"], dict(zip(kin, th))) - dense["mu"]
    r = least_squares(res, entry["centre"][:k], bounds=(lb[:k], ub[:k]), x_scale=(ub - lb)[:k])
    return np.clip(np.r_[r.x, [dense["kd"]] if entry["death"] else [], dense["Yxs"]], lb, ub)


def stack(runs):
    return np.hstack([r["Y"] for r in runs]), np.hstack([r["sigma"] for r in runs])


def select(library, runs, dense, starts=None):
    Y, sigma = stack(runs)
    starts = starts or (lambda m: {"NODE": match(library[m], dense), "centre": library[m]["centre"]})
    return identify(list(library), starts=starts, simulate=lambda m, th, t, x0: simulate(library[m], th, t, x0, runs),
                    bounds=lambda m: library[m]["bounds"], x0=initial_state(T_OBS, Y), t_obs=T_OBS, Y_obs=Y, sigma=sigma)


def chi2_ok(n):
    """Upper edge of the noise band of chi2 for n measured values."""
    return n + 3 * np.sqrt(2 * n)


# ── symbolic regression ──────────────────────────────────────────────────────
SR_C1 = dict(lhs="S", vars=["X", "S"], Kvar="S", terms={
    "S": lambda v, q: v["S"],
    "S*X": lambda v, q: v["S"] * v["X"],
    "S^2": lambda v, q: v["S"] ** 2,
    "S*exp(-S/K)": lambda v, q: v["S"] * exp_(-v["S"] / q["K"]),
    "mu": lambda v, q: v["S"] ** 0,
    "mu*X": lambda v, q: v["X"],
    "mu*S^2": lambda v, q: v["S"] ** 2,
    "mu*S*X": lambda v, q: v["S"] * v["X"],
})


def sr_entry(law, death, Yxs, kd):
    terms, k = law["terms"], len(law["terms"])
    has_K = bool(law["q"])
    names = [f"c{i + 1}" for i in range(k)] + (["K"] if has_K else []) + (["kd"] if death else []) + ["Yxs"]
    start = np.r_[law["coef"], [law["q"]["K"]] if has_K else [], [kd] if death else [], Yxs]
    lo, hi = np.minimum(0.1 * start, 10 * start), np.maximum(0.1 * start, 10 * start)
    lo[-1], hi[-1] = 0.03, 3.0
    if death:
        lo[-2], hi[-2] = 5e-4, 0.5

    def mu(X, S, p):
        q = {"K": p["K"]} if has_K else {}
        return sr_mu({"X": X, "S": S}, SR_C1, dict(terms=terms, coef=[p[f"c{i + 1}"] for i in range(k)], q=q))

    return dict(mu=mu, names=names, death=death, centre=start, bounds=(lo, hi), law=law)


def law_from_theta(entry, theta):
    p = dict(zip(entry["names"], theta))
    k = len(entry["law"]["terms"])
    return dict(terms=list(entry["law"]["terms"]), coef=[float(p[f"c{i + 1}"]) for i in range(k)],
                q={"K": float(p["K"])} if "K" in p else {})


def sr_library(laws, dense, tag="SR"):
    return {f"{tag}: {' + '.join(law['terms'])}{' +kd' if death else ''}": sr_entry(law, death, dense["Yxs"], dense["kd"])
            for law in laws for death in (False, True)}


def screen_all(runs, dense, tried, n_full=5, max_nfev=50):
    """Escalation: short refinement of every valid law against the data, full refinement of the n_full best."""
    v = {"X": dense["X"], "S": dense["S"]}
    laws = [law for law, _ in sr_candidates(v, dense["mu"], SR_C1, per_size=None)]
    Y, sigma = stack(runs)
    x0 = initial_state(T_OBS, Y)
    x0_lb, x0_ub = np.maximum(x0 - 3 * sigma, 0), x0 + 3 * sigma
    n_data = np.isfinite(Y).sum()
    scores = []
    for law in laws:
        if law["terms"] in tried:
            continue
        entry = sr_entry(law, False, dense["Yxs"], dense["kd"])
        n = len(entry["centre"])
        theta, chi2, _ = refine(lambda th, t: simulate(entry, th[:n], t, th[n:], runs), np.r_[entry["centre"], x0],
                                np.r_[entry["bounds"][0], x0_lb], np.r_[entry["bounds"][1], x0_ub],
                                T_OBS, Y, sigma, max_nfev=max_nfev)
        scores.append((chi2 + n * np.log(n_data), law_from_theta(entry, theta[:n])))
    best = [law for _, law in sorted(scores, key=lambda s: s[0])[:n_full]]
    lib = sr_library(best, dense, tag="SR (screened)")
    return lib, select(lib, runs, dense, starts=lambda m: {"SR": lib[m]["centre"]}), len(laws)


def summarise(df, library_names, entries):
    """Serializable ranking: name -> (k, chi2, BIC, weight, theta, x0, formula or None)."""
    out = {}
    for m, row in df.iterrows():
        law = law_from_theta(entries[m], row.theta) if m in entries and "law" in entries[m] else None
        out[m] = dict(k=int(row.k), chi2=float(row.chi2), BIC=float(row.BIC), weight=float(row.weight),
                      theta=np.asarray(row.theta, float), x0=np.asarray(row.x0, float),
                      library=m in library_names, law=law, formula=str(sr_formula(SR_C1, law)) if law else None)
    return out


def analyse(runs, with_sr=True, escalate=True):
    """Full pipeline on a set of runs: NODE, library selection, SR, escalation if the library fails."""
    out, t0 = {}, time.time()
    dense = fit_node(runs)
    out["node"] = {k: v for k, v in dense.items()}
    sel = select(LIBRARY, runs, dense)
    out["library"] = summarise(sel, list(LIBRARY), LIBRARY)
    Y, _ = stack(runs)
    n = int(np.isfinite(Y).sum())
    out["n"] = n
    out["library_adequate"] = bool(sel.chi2.min() <= chi2_ok(n))
    if with_sr:
        v = {"X": dense["X"], "S": dense["S"]}
        cands = sr_candidates(v, dense["mu"], SR_C1, per_size=2)
        out["sr_candidates"] = [(law, float(err), str(sr_formula(SR_C1, law))) for law, err in cands]
        lib = sr_library([law for law, _ in cands], dense)
        sr_sel = select(lib, runs, dense, starts=lambda m: {"SR": lib[m]["centre"]})
        entries = {**LIBRARY, **lib}
        allr = add_bic(pd.concat([sel, sr_sel]), n)
        out["escalated"] = False
        if escalate and not out["library_adequate"]:
            slib, ssel, n_laws = screen_all(runs, dense, tried=[law["terms"] for law, _ in cands])
            entries.update(slib)
            allr = add_bic(pd.concat([allr, ssel]), n)
            out["escalated"], out["n_screened"] = True, n_laws
        out["all"] = summarise(allr, list(LIBRARY), entries)
    out["seconds"] = time.time() - t0
    return out


def run_job(name, seed):
    """One experiment and noise realisation: single run (a) and three runs (a, b, c)."""
    runs = make_runs(name, seed)
    res = dict(name=name, seed=seed, runs=[{k: r[k] for k in ("key", "Y", "truth", "sigma")} for r in runs])
    res["single"] = analyse(runs[:1])
    res["joint"] = analyse(runs)
    return res


# ── closing the loop ─────────────────────────────────────────────────────────
ACCEPT_DBIC = -10.0


def accepted_law(result):
    """The law accepted from a joint analysis, or None. Accept: beats the library by dBIC < -10 and
    fits within the noise."""
    allr = result["all"]
    lib = {m: r for m, r in allr.items() if r["library"]}
    srs = {m: r for m, r in allr.items() if not r["library"]}
    if not srs:
        return None, None
    best_sr = min(srs, key=lambda m: srs[m]["BIC"])
    dbic = srs[best_sr]["BIC"] - min(r["BIC"] for r in lib.values())
    ok = dbic < ACCEPT_DBIC and srs[best_sr]["chi2"] <= chi2_ok(result["n"])
    return (best_sr if ok else None), dict(dbic=float(dbic), chi2=float(srs[best_sr]["chi2"]), name=best_sr,
                                           formula=srs[best_sr]["formula"], accepted=bool(ok))


def accepted_entry(name, rec):
    """Library entry for an accepted law, centred on its refined values (theta = [c..., (K), (kd), Yxs])."""
    law = rec["law"]
    death = name.endswith("+kd")
    k, has_K = len(law["terms"]), bool(law["q"])
    kd = rec["theta"][k + int(has_K)] if death else 0.0
    return sr_entry(law, death, rec["theta"][-1], kd)


def accepted_starts(entry, dense):
    law, _ = fit_law({"X": dense["X"], "S": dense["S"]}, dense["mu"], SR_C1, entry["law"]["terms"])
    starts = {"centre": entry["centre"]}
    if law is not None:
        node = np.r_[law["coef"], [law["q"]["K"]] if law["q"] else [], [dense["kd"]] if entry["death"] else [],
                     dense["Yxs"]]
        starts["NODE"] = np.clip(node, *entry["bounds"])
    return starts


def loop_job(seed, hidden_joint, other_nodes):
    """Close the loop for one seed: accept from the hidden experiment, test on a later run (design d) and
    check that the new entry does not steal the library experiments."""
    name, info = accepted_law(hidden_joint)
    res = dict(seed=seed, acceptance=info, later=None, steal={})
    if name is None:
        return res
    entry = accepted_entry(name, hidden_joint["all"][name])
    new = {"SR-1": entry}
    later_runs = make_runs("hidden", seed, keys=("d",))
    dense = fit_node(later_runs)
    base = select(LIBRARY, later_runs, dense)
    plus = add_bic(pd.concat([base, select(new, later_runs, dense, starts=lambda m: accepted_starts(entry, dense))]),
                   int(np.isfinite(later_runs[0]["Y"]).sum()))
    res["later"] = dict(base=summarise(base, list(LIBRARY), LIBRARY), plus=summarise(plus, list(LIBRARY), {**LIBRARY, **new}),
                        runs=[{k: r[k] for k in ("key", "Y", "truth", "sigma")} for r in later_runs])
    for ename, (runs, dense_e, joint_lib) in other_nodes.items():
        sel = select(new, runs, dense_e, starts=lambda m: accepted_starts(entry, dense_e))
        df = pd.DataFrame({m: dict(k=r["k"], chi2=r["chi2"], crit=r["chi2"], theta=r["theta"], x0=r["x0"])
                           for m, r in joint_lib.items()}).T
        df = pd.concat([df[["k", "crit", "chi2", "theta", "x0"]], sel[["k", "crit", "chi2", "theta", "x0"]]])
        df["k"] = df["k"].astype(int)
        df["crit"] = df["crit"].astype(float)
        plus_e = add_bic(df, int(np.isfinite(stack(runs)[0]).sum()))
        res["steal"][ename] = dict(weight_new=float(plus_e.weight["SR-1"]),
                                   selected=str(plus_e.weight.idxmax()), wmax=float(plus_e.weight.max()))
    return res
