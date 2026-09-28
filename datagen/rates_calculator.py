"""
Compute observable specific rates from ODE trajectories.

For in silico data the rates are computed exactly by evaluating the model's
RHS function at each timepoint — no numerical differentiation is needed.
This avoids the noise amplification problem that makes derivative estimation
unreliable for sparse experimental data.

The feed-corrected observable specific rates are:

    mu_obs(t) = dX/dt / X  +  F(t)/V(t)
              = mu1(t)                         [no-death models]
              = mu1(t) - kd                    [models with cell death]

    q_S1(t) = -[dS1/dt - F/V * (S1_feed - S1)] / X   [g S1 / g X / h]
    q_S2(t) = -[dS2/dt - F/V * (S2_feed - S2)] / X   [g S2 / g X / h]
    q_P(t)  =  [dP/dt  + F/V * P            ] / X    [g P  / g X / h]

These are the intrinsic kinetic quantities that depend only on the mechanism,
not on the feed profile. The classifier trained on these features generalises
across different feeding strategies.
"""

import numpy as np
from kinetic_models.feed_fun import create_feed_rate_function, get_feed_values


def compute_specific_rates(
    trajectory: np.ndarray,
    t_span: np.ndarray,
    params,
    model_fn,
    exter_df,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute [mu_obs, q_S1, q_S2, q_P] at each timepoint.

    Parameters
    ----------
    trajectory : (T, 5) array  [X, S1, S2, P, V(L)]
    t_span     : (T,) time array
    params     : kinetic parameter vector for the model
    model_fn   : callable f(state, t, params, exter_df) → [dX, dS1, dS2, dP, dV]
    exter_df   : synthetic experiment DataFrame
    eps        : floor for biomass/volume to prevent division by zero

    Returns
    -------
    rates : (T, 4) array  [mu_obs, q_S1, q_S2, q_P]
    """
    S1_feed, S2_feed = get_feed_values(exter_df)
    feed_func = create_feed_rate_function(exter_df)

    T = len(t_span)
    rates = np.full((T, 4), np.nan)

    for i in range(T):
        state = trajectory[i]   # [X1, S1, S2, P, V]
        t = t_span[i]

        # Evaluate ODE right-hand side exactly
        dstate = model_fn(state, t, params, exter_df)
        dXdt, dS1dt, dS2dt, dPdt, _ = dstate

        X1 = max(float(state[0]), eps)
        S1 = float(state[1])
        S2 = float(state[2])
        P  = float(state[3])
        V  = max(float(state[4]), eps)

        Fs = feed_func(t) / 1000.0    # mL/h → L/h
        FV = Fs / V                    # dilution rate, h^-1

        mu_obs = dXdt / X1 + FV
        q_S1   = -(dS1dt - FV * (S1_feed - S1)) / X1
        q_S2   = -(dS2dt - FV * (S2_feed - S2)) / X1
        q_P    =  (dPdt  + FV * P)               / X1

        rates[i] = [mu_obs, q_S1, q_S2, q_P]

    return rates
