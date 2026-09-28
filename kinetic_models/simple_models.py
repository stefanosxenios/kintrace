"""
Single-substrate fed-batch kinetic mechanisms — the "simple benchmark" system.

This is a deliberately clean, controllable testbed that sits *alongside* (not
inside) the dual-substrate xylitol library (models 1-31 in base_models.py).
State is 3-dimensional:

    y = [X, S, V]      X = biomass (g/L), S = substrate (g/L), V = volume (L)

All four mechanisms share the *same* fed-batch mass balance and differ ONLY in
the specific growth rate mu(.).  This is exactly what makes them a good
identification benchmark: the classifier must pick the mechanism from the shape
of the growth curve, and the regressor must recover its parameters.

Mass balance (identical for every mechanism)
--------------------------------------------
    dV/dt = F(t)                                  [L/h]
    dX/dt = mu * X          - (F/V) * X
    dS/dt = -(1/Yxs) * mu * X + (F/V) * (Sin - S)

Mechanisms (mu forms) and their kinetic parameters
--------------------------------------------------
    1  monod                 mu = mumax * S / (Ks + S)                     [mumax, Ks, Yxs]
    2  contois               mu = mumax * S / (Kc*X + S)                   [mumax, Kc, Yxs]
    3  substrate_inhibition  mu = mumax * S / (Ks + S + S^2/KI)            [mumax, Ks, KI, Yxs]
    4  biomass_inhibition    mu = mumax * S / (Ks + S) * (1 - X/Xmax)      [mumax, Ks, Xmax, Yxs]

`Sin` (feed substrate concentration) is a *known* operating condition, not an
estimated kinetic parameter — consistent with how it is treated in the reference
neural-ODE notebooks (scripts/sindy_paper*.ipynb).

The ODE right-hand sides use the signature expected by scipy.integrate.odeint:

    rhs(y, t, k, feed_fn, Sin)

where `k` is the parameter vector (order matches PARAM_NAMES below) and
`feed_fn` is a callable feed_fn(t) -> F [L/h].
"""

import numpy as np


# ── Mechanism metadata ───────────────────────────────────────────────────────

# id -> (name, ordered parameter names)
SIMPLE_MODELS = {
    1: ("monod",                ["mumax", "Ks", "Yxs"]),
    2: ("contois",              ["mumax", "Kc", "Yxs"]),
    3: ("substrate_inhibition", ["mumax", "Ks", "KI", "Yxs"]),
    4: ("biomass_inhibition",   ["mumax", "Ks", "Xmax", "Yxs"]),
}

NAME_TO_ID = {name: mid for mid, (name, _) in SIMPLE_MODELS.items()}


def get_param_names(model_id: int) -> list:
    """Ordered kinetic-parameter names for a mechanism."""
    return list(SIMPLE_MODELS[model_id][1])


# ── Specific growth rate mu(.) for each mechanism ────────────────────────────

def _mu_monod(X, S, p):
    return p["mumax"] * S / (p["Ks"] + S)


def _mu_contois(X, S, p):
    return p["mumax"] * S / (p["Kc"] * X + S)


def _mu_substrate_inhibition(X, S, p):
    return p["mumax"] * S / (p["Ks"] + S + S ** 2 / p["KI"])


def _mu_biomass_inhibition(X, S, p):
    # clip the (1 - X/Xmax) factor at 0 so biomass cannot go negative once X>Xmax
    return p["mumax"] * S / (p["Ks"] + S) * np.maximum(1.0 - X / p["Xmax"], 0.0)


_MU_FUNCS = {
    "monod":                _mu_monod,
    "contois":              _mu_contois,
    "substrate_inhibition": _mu_substrate_inhibition,
    "biomass_inhibition":   _mu_biomass_inhibition,
}


def compute_mu(model_id: int, X, S, params) -> np.ndarray:
    """
    True specific growth rate mu for a mechanism.

    Parameters
    ----------
    model_id : int
    X, S     : scalars or arrays
    params   : dict OR ordered sequence matching get_param_names(model_id)
    """
    name = SIMPLE_MODELS[model_id][0]
    p = _as_param_dict(model_id, params)
    return _MU_FUNCS[name](X, S, p)


# ── ODE right-hand sides (odeint signature) ──────────────────────────────────

def _as_param_dict(model_id: int, params) -> dict:
    if isinstance(params, dict):
        return params
    names = get_param_names(model_id)
    return {n: float(v) for n, v in zip(names, params)}


def _make_rhs(model_id: int):
    name = SIMPLE_MODELS[model_id][0]
    mu_fn = _MU_FUNCS[name]

    def rhs(y, t, k, feed_fn, Sin):
        y = np.maximum(y, 0.0)
        X, S, V = y
        V = max(V, 1e-9)
        p = _as_param_dict(model_id, k)

        F = float(feed_fn(t))                 # L/h
        mu = mu_fn(X, S, p)

        D = F / V
        dXdt = mu * X - D * X
        dSdt = -(1.0 / p["Yxs"]) * mu * X + D * (Sin - S)
        dVdt = F
        return [dXdt, dSdt, dVdt]

    return rhs


# Pre-built RHS callables, keyed by id
_RHS = {mid: _make_rhs(mid) for mid in SIMPLE_MODELS}


def get_simple_model(model_id: int):
    """Return the odeint RHS callable rhs(y, t, k, feed_fn, Sin) for a mechanism."""
    return _RHS[model_id]
