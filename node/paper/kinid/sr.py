"""Symbolic regression of rate laws from learned rates.

A dictionary describes candidate laws for one specific rate mu:
    mu * lhs = sum_k c_k theta_k(v, q)
Terms whose name starts with "mu" are multiplied by mu (they build the denominator); the others build the
numerator. Terms containing "K" use a fitted constant q["K"] (exponential terms). For a law with term set T:
    mu = N / D,  N = sum_{plain} a_k theta_k,  D = lhs + sum_{mu-terms} b_k theta_k,  b_k >= 0.
"""
import itertools

import numpy as np
import sympy as sp
from scipy.optimize import least_squares


def exp_(x):
    return sp.exp(x) if isinstance(x, sp.Basic) else np.exp(x)


def is_den(term):
    return term.startswith("mu")


def sr_parts(v, d, law):
    """Numerator and denominator of the law {terms, coef, q} (coef in the mu*lhs convention)."""
    q = law["q"]
    num = sum(c * d["terms"][k](v, q) for k, c in zip(law["terms"], law["coef"]) if not is_den(k))
    den = v[d["lhs"]] - sum(c * d["terms"][k](v, q) for k, c in zip(law["terms"], law["coef"]) if is_den(k))
    return num + 0 * v[d["lhs"]], den


def sr_mu(v, d, law):
    num, den = sr_parts(v, d, law)
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def fit_law(v, mu, d, terms):
    """Fit the law with term set `terms` to the learned rate mu (least squares in rate space).
    Free numerator coefficients, non-negative denominator coefficients, fitted K for exponential terms.
    Returns (law, relative error ||mu_law - mu|| / ||mu||)."""
    terms = list(terms)
    plain = [k for k in terms if not is_den(k)]
    dens = [k for k in terms if is_den(k)]
    has_K = any("K" in k for k in terms)
    n_a, n_b = len(plain), len(dens)
    lhs = v[d["lhs"]]

    def law_of(p):
        q = {"K": np.exp(p[-1])} if has_K else {}
        return dict(terms=terms, q=q, coef=[p[plain.index(k)] if k in plain else -p[n_a + dens.index(k)]
                                            for k in terms])

    best = None
    kscale = max(np.max(v[d.get("Kvar", d["lhs"])]), 1e-6)
    for K0 in (np.geomspace(0.03, 3, 5) * kscale if has_K else [None]):
        q = {"K": K0} if has_K else {}
        Q = np.column_stack([d["terms"][k](v, q) * np.ones_like(mu) for k in dens] + [np.zeros_like(mu)])[:, :n_b]
        b0 = 0.1 * np.mean(lhs) / np.maximum(Q.mean(0), 1e-12)          # each denominator term ~10 % of lhs
        N = np.column_stack([d["terms"][k](v, q) * np.ones_like(mu) for k in plain])
        a0 = np.linalg.lstsq(N, mu * (lhs + Q @ b0), rcond=None)[0]
        p0 = np.r_[a0, b0, [np.log(K0)] if has_K else []]
        lb = np.r_[np.full(n_a, -np.inf), np.zeros(n_b), [np.log(1e-3 * kscale)] if has_K else []]
        ub = np.r_[np.full(n_a + n_b, np.inf), [np.log(1e3 * kscale)] if has_K else []]
        p0 = np.clip(p0, lb + 1e-12, ub - 1e-12)
        try:
            fit = least_squares(lambda p: sr_mu(v, d, law_of(p)) - mu, p0, bounds=(lb, ub), x_scale="jac")
        except Exception:
            continue
        if best is None or fit.cost < best.cost:
            best = fit
    if best is None:
        return None, np.inf
    law = law_of(best.x)
    return law, float(np.linalg.norm(sr_mu(v, d, law) - mu) / np.linalg.norm(mu))


def is_valid(v, mu, d, law, vanish=0.05):
    """Physical validity: a numerator, every denominator term switched on, mu >= 0 along the states, and
    mu -> 0 when the substrate (lhs) runs out (<= `vanish` x the largest learned rate)."""
    if law is None or all(is_den(k) for k in law["terms"]):
        return False
    lhs_mean = np.mean(v[d["lhs"]])
    for c, k in zip(law["coef"], law["terms"]):
        if is_den(k) and -c * np.mean(d["terms"][k](v, law["q"]) * np.ones_like(mu)) < 1e-6 * lhs_mean:
            return False
    if np.any(sr_mu(v, d, law) < 0):
        return False
    v0 = {**v, d["lhs"]: 1e-6 * np.max(v[d["lhs"]]) + 0 * v[d["lhs"]]}
    return bool(np.max(sr_mu(v0, d, law)) <= vanish * np.max(mu))


def sr_candidates(v, mu, d, max_terms=4, per_size=2):
    """Best-subset search: the `per_size` best valid laws of each size (all valid laws if per_size=None),
    ranked by their error against the learned rate. Returns a list of (law, rel_error)."""
    found = []
    for size in range(1, max_terms + 1):
        valid = []
        for terms in itertools.combinations(d["terms"], size):
            if all(is_den(k) for k in terms):
                continue
            law, err = fit_law(v, mu, d, terms)
            if is_valid(v, mu, d, law):
                valid.append((law, err))
        found += sorted(valid, key=lambda c: c[1])[:per_size]
    return found


def sr_formula(d, law, digits=3):
    """Readable law; normalised so that a constant term in the denominator (if any) equals 1."""
    sym = {name: sp.Symbol(name, positive=True) for name in d["vars"]}
    q = {k: sp.Float(val, digits) for k, val in law["q"].items()}
    num, den = sr_parts(sym, d, dict(law, coef=[sp.Float(c, digits) for c in law["coef"]], q=q))
    const = den.subs({s: 0 for s in sym.values()})
    if const != 0:
        num, den = sp.expand(num / const), sp.expand(den / const)
    return sp.N(num, digits) / sp.N(den, digits)


def law_latex(d, law, digits=3):
    return sp.latex(sr_formula(d, law, digits))
