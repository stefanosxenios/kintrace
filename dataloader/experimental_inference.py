"""
Preprocessing pipeline for experimental data → LSTM classifier input.

Takes a cleaned experimental DataFrame and produces a tensor ready for
the classifier, handling the key challenge of sparse measurements:
  - Only ~12 offline timepoints vs 100-point in silico training sequences
  - Noisy derivative estimation (GPR or cubic spline)

Two derivative methods available via `method` parameter:
  'spline' : Piecewise cubic spline (scipy.interpolate.CubicSpline)
             Faster, deterministic, can oscillate on very sparse/noisy data
  'gpr'    : Gaussian Process Regression (sklearn)
             Smoother, probabilistic uncertainty, better for noisy sparse data
             Tunable via gpr_alpha (noise level) and gpr_length_scale

Feature columns produced per dataset_type:
  'concentrations' : [X, S1, S2, P, V, F/V]              — no differentiation needed
  'rates'          : [mu_obs, q_S1, q_S2, q_P, F/V]      — derivatives required
  'hybrid'         : both combined (10 features)

Usage:
    from dataloader.experimental_inference import prepare_experimental_input
    tensor = prepare_experimental_input(
        df=cleaned_BC12,
        dataset_type='hybrid',
        scaler=scaler,          # from training
        seq_len=100,
        method='gpr',
        S1_feed=82.99,
        S2_feed=260.27,
    )
    # tensor shape: (1, seq_len, n_features)
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline


# ── Feed concentration reader ─────────────────────────────────────────────────

def get_feed_concentrations(df):
    """
    Extract S1_feed (glucose) and S2_feed (xylose) from the DataFrame's feed row.

    The feed row is identified by Sample name ending in '_Feed' or by having
    NaN in 'Time (h)'. Falls back to BC12 reference values if not found.

    Returns
    -------
    S1_feed : float  glucose equivalent in feed [g/L]
    S2_feed : float  xylose in feed [g/L]
    """
    # Try to find a feed row (NaN time or sample name contains '_Feed')
    feed_row = df[df['Time (h)'].isna()]
    if feed_row.empty and 'Sample name' in df.columns:
        feed_row = df[df['Sample name'].astype(str).str.contains('_Feed', na=False)]

    if not feed_row.empty:
        S1 = feed_row['Glucose'].values[0]
        S2 = feed_row['Xylose'].values[0]
        if np.isfinite(S1) and np.isfinite(S2):
            return float(S1), float(S2)

    # Fallback: BC12 reference values
    import warnings
    warnings.warn(
        "Feed row not found in DataFrame — using BC12 reference values "
        "(S1_feed=82.99, S2_feed=260.27). Pass the full DataFrame including "
        "the feed row to get experiment-specific concentrations.",
        stacklevel=3,
    )
    return 82.99, 260.27


# ── Smoothing & differentiation ───────────────────────────────────────────────

def smooth_and_differentiate(
    t: np.ndarray,
    y: np.ndarray,
    method: str = 'spline',
    n_dense: int = 300,
    gpr_alpha: float = 0.05,
    gpr_length_scale: float = None,
    gpr_length_scale_bounds: tuple = (1.0, 200.0),
):
    """
    Fit a smooth curve to sparse (t, y) observations and return
    smoothed values + derivative on a dense evaluation grid.

    Parameters
    ----------
    t                      : (n,) time points (no NaN, sorted)
    y                      : (n,) observed values (no NaN)
    method                 : 'spline' or 'gpr'
    n_dense                : number of points in dense output grid
    gpr_alpha              : GPR noise variance (larger → smoother fit)
    gpr_length_scale       : RBF length scale in hours (None = auto-optimise)
    gpr_length_scale_bounds: bounds for RBF length scale optimisation

    Returns
    -------
    t_dense  : (n_dense,) dense time grid
    y_smooth : (n_dense,) smoothed values
    dydt     : (n_dense,) derivative dy/dt
    """
    t_dense = np.linspace(t[0], t[-1], n_dense)

    if method == 'spline':
        cs = CubicSpline(t, y, bc_type='not-a-knot')
        y_smooth = cs(t_dense)
        dydt     = cs(t_dense, 1)          # first derivative

    elif method == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, WhiteKernel, ConstantKernel as C
        )

        # Default length scale: ~25% of time range
        if gpr_length_scale is None:
            gpr_length_scale = (t[-1] - t[0]) * 0.25

        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(gpr_length_scale, gpr_length_scale_bounds)
            + WhiteKernel(noise_level=gpr_alpha, noise_level_bounds=(1e-5, 10.0))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6,
            n_restarts_optimizer=3, normalize_y=True,
        )
        gpr.fit(t.reshape(-1, 1), y)
        y_smooth = gpr.predict(t_dense.reshape(-1, 1))

        # Derivative via central finite differences on the smooth posterior
        dydt = np.gradient(y_smooth, t_dense)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'spline' or 'gpr'.")

    return t_dense, y_smooth, dydt


# ── F/V computation ───────────────────────────────────────────────────────────

def compute_FV_from_df(df, t_eval: np.ndarray) -> np.ndarray:
    """
    Compute feed dilution rate F(t)/V(t) [h^-1] at arbitrary time points.

    Feed rate is treated as a step function (last-observation-carried-forward).
    Volume is linearly interpolated.

    Parameters
    ----------
    df     : cleaned experimental DataFrame (measurement rows only, no feed row)
    t_eval : (n_eval,) time points at which to evaluate F/V

    Returns
    -------
    FV : (n_eval,) dilution rate h^-1
    """
    t_meas   = df['Time (h)'].values.astype(float)
    V_mL     = df['Reactor volume (mL)'].values.astype(float)
    F_mLh    = df['Feed rate (mL/h)'].fillna(0.0).values.astype(float)

    # Interpolate volume (linear, more physical than step)
    V_interp = np.interp(t_eval, t_meas, V_mL)

    # Step-interpolate feed rate (last observation carried forward)
    F_interp = np.zeros(len(t_eval))
    for i, te in enumerate(t_eval):
        idx = np.searchsorted(t_meas, te, side='right') - 1
        idx = max(0, min(idx, len(F_mLh) - 1))
        F_interp[i] = F_mLh[idx]

    # F/V: both in mL, the 1/1000 factors cancel
    FV = F_interp / np.maximum(V_interp, 1e-6)   # h^-1

    return FV


# ── Specific rates from experimental data ─────────────────────────────────────

def compute_rates_from_df(
    df,
    method: str = 'spline',
    S1_feed: float = None,
    S2_feed: float = None,
    n_dense: int = 300,
    gpr_alpha: float = 0.05,
    gpr_length_scale: float = None,
    eps: float = 1e-6,
):
    """
    Compute feed-corrected specific rates from an experimental DataFrame.

    Rates:
        mu_obs = dX/dt / X + F/V
        q_S1   = -(dS1/dt - F/V*(S1_feed - S1)) / X
        q_S2   = -(dS2/dt - F/V*(S2_feed - S2)) / X
        q_P    = (dP/dt + F/V*P) / X

    Parameters
    ----------
    df              : cleaned DataFrame (measurement rows, no feed row)
    method          : 'spline' or 'gpr'
    S1_feed         : glucose equivalent concentration in feed [g/L]
    S2_feed         : xylose concentration in feed [g/L]
    n_dense         : dense grid size for smoothing
    gpr_alpha       : GPR noise level
    gpr_length_scale: RBF length scale (None = auto)
    eps             : floor for biomass to avoid division by zero

    Returns
    -------
    t_dense : (n_dense,) time grid
    rates   : (n_dense, 4) array [mu_obs, q_S1, q_S2, q_P]
    FV      : (n_dense,)   dilution rate
    """
    # Auto-read feed concentrations from the DataFrame if not supplied
    if S1_feed is None or S2_feed is None:
        S1_feed, S2_feed = get_feed_concentrations(df)

    # --- Extract state variables, drop NaN rows per variable ---
    def _clean(col):
        sub = df[['Time (h)', col]].dropna()
        return sub['Time (h)'].values.astype(float), sub[col].values.astype(float)

    # Compute Glu_eq if not present (Glucose + Ethanol * 1.2)
    if 'Glu_eq' not in df.columns:
        df = df.copy()
        df['Glu_eq'] = df['Glucose'].fillna(0) + df['Ethanol'].fillna(0) * 1.2

    t_X,  X  = _clean('DW (g/L)')
    t_S1, S1 = _clean('Glu_eq')
    t_S2, S2 = _clean('Xylose')
    t_P,  P  = _clean('Xylitol')

    # Dense time grid (union of all variable time ranges)
    t0 = min(t_X[0],  t_S1[0],  t_S2[0],  t_P[0])
    tf = max(t_X[-1], t_S1[-1], t_S2[-1], t_P[-1])
    t_dense = np.linspace(t0, tf, n_dense)

    smooth_kw = dict(
        method=method, n_dense=n_dense,
        gpr_alpha=gpr_alpha, gpr_length_scale=gpr_length_scale,
    )

    # Smooth & differentiate each variable
    _, X_s,  dXdt  = smooth_and_differentiate(t_X,  X,  **smooth_kw)
    _, S1_s, dS1dt = smooth_and_differentiate(t_S1, S1, **smooth_kw)
    _, S2_s, dS2dt = smooth_and_differentiate(t_S2, S2, **smooth_kw)
    _, P_s,  dPdt  = smooth_and_differentiate(t_P,  P,  **smooth_kw)

    # F/V on the dense grid
    FV = compute_FV_from_df(df, t_dense)

    # Specific rates
    X_safe  = np.maximum(X_s,  eps)
    mu_obs  = dXdt  / X_safe + FV
    q_S1    = -(dS1dt - FV * (S1_feed - S1_s)) / X_safe
    q_S2    = -(dS2dt - FV * (S2_feed - S2_s)) / X_safe
    q_P     = (dPdt  + FV * P_s)               / X_safe

    rates = np.stack([mu_obs, q_S1, q_S2, q_P], axis=1)   # (n_dense, 4)
    return t_dense, rates, FV


# ── Main inference preprocessor ───────────────────────────────────────────────

def prepare_experimental_input(
    df,
    dataset_type: str = 'concentrations',
    scaler=None,
    seq_len: int = 100,
    method: str = 'spline',
    S1_feed: float = None,
    S2_feed: float = None,
    n_dense: int = 300,
    gpr_alpha: float = 0.05,
    gpr_length_scale: float = None,
    eps: float = 1e-6,
):
    """
    Transform an experimental DataFrame into a classifier-ready tensor.

    Parameters
    ----------
    df           : cleaned experimental DataFrame (from FedBatchDataLoader)
    dataset_type : 'concentrations', 'rates', or 'hybrid'
    scaler       : fitted StandardScaler from training (None = no scaling)
    seq_len      : output sequence length (must match training seq_len)
    method       : derivative method — 'spline' or 'gpr'
    S1_feed      : glucose equivalent in feed [g/L]
    S2_feed      : xylose in feed [g/L]
    n_dense      : dense interpolation grid (intermediate step)
    gpr_alpha    : GPR noise level (higher = smoother, less fit to data)
    gpr_length_scale : GPR RBF length scale in hours (None = auto)
    eps          : biomass floor for rate computation

    Returns
    -------
    tensor  : (1, seq_len, n_features) float32 torch.Tensor
    t_out   : (seq_len,) output time array
    debug   : dict with intermediate arrays for inspection/plotting
    """
    # Auto-read feed concentrations from the full df (before dropping feed row)
    if S1_feed is None or S2_feed is None:
        S1_feed, S2_feed = get_feed_concentrations(df)

    # Drop the feed row (NaN time) and sort by time
    meas = df[df['Time (h)'].notna()].copy()
    meas = meas.sort_values('Time (h)').reset_index(drop=True)

    if 'Glu_eq' not in meas.columns:
        meas['Glu_eq'] = meas['Glucose'].fillna(0) + meas['Ethanol'].fillna(0) * 1.2

    t0 = meas['Time (h)'].iloc[0]
    tf = meas['Time (h)'].iloc[-1]
    t_out = np.linspace(t0, tf, seq_len)

    debug = {}

    # ── Concentrations block ──────────────────────────────────────────
    if dataset_type in ('concentrations', 'hybrid'):
        def _interp(col):
            sub = meas[['Time (h)', col]].dropna()
            t_c = sub['Time (h)'].values.astype(float)
            y_c = sub[col].values.astype(float)
            # Spline for smooth interpolation to output grid
            cs  = CubicSpline(t_c, y_c, bc_type='not-a-knot')
            return np.maximum(cs(t_out), 0.0)

        X_out  = _interp('DW (g/L)')
        S1_out = _interp('Glu_eq')
        S2_out = _interp('Xylose')
        P_out  = _interp('Xylitol')
        # Volume MUST be in litres here. The in-silico training set stores V in L
        # (datagen/generate_v2.py converts V0_mL/1000 before integrating), so feeding
        # raw millilitres puts this channel ~3000 sigma outside the regressor's global
        # standardiser, saturating the network and collapsing its output to a constant.
        # The classifier is unaffected because it min-max scales each trajectory to its
        # own range, which cancels the unit error -- which is why this went unnoticed.
        V_out  = _interp('Reactor volume (mL)') / 1000.0
        FV_out = compute_FV_from_df(meas, t_out)   # F/V is a ratio: mL/mL cancels, h^-1

        conc_block = np.stack([X_out, S1_out, S2_out, P_out, V_out, FV_out], axis=1)
        debug['concentrations'] = conc_block
        debug['t_out'] = t_out

    # ── Rates block ──────────────────────────────────────────────────
    if dataset_type in ('rates', 'hybrid'):
        t_dense, rates_dense, FV_dense = compute_rates_from_df(
            meas, method=method,
            S1_feed=S1_feed, S2_feed=S2_feed,
            n_dense=n_dense, gpr_alpha=gpr_alpha,
            gpr_length_scale=gpr_length_scale, eps=eps,
        )
        # Resample dense rates to output grid via interpolation
        from scipy.interpolate import interp1d
        rates_out = np.zeros((seq_len, 4))
        for j in range(4):
            f = interp1d(t_dense, rates_dense[:, j], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            rates_out[:, j] = f(t_out)

        FV_out_rates = np.interp(t_out, t_dense, FV_dense)

        # rates features: [mu_obs, q_S1, q_S2, q_P, F/V]
        rates_block = np.concatenate(
            [rates_out, FV_out_rates.reshape(-1, 1)], axis=1
        )
        debug['rates'] = rates_block
        debug['t_dense'] = t_dense
        debug['rates_dense'] = rates_dense

    # ── Assemble final feature array ─────────────────────────────────
    if dataset_type == 'concentrations':
        features = conc_block                                # (seq_len, 6)
    elif dataset_type == 'rates':
        features = rates_block                               # (seq_len, 5)
    elif dataset_type == 'hybrid':
        features = np.concatenate([conc_block, rates_block], axis=1)  # (seq_len, 11)
    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

    # ── Scale ────────────────────────────────────────────────────────
    if scaler is not None:
        features = scaler.transform(features)

    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, F)
    return tensor, t_out, debug


# ── Convenience: run inference and print top-k predictions ───────────────────

def classify_experiment(
    model,
    df,
    label_names: list,
    dataset_type: str = 'concentrations',
    scaler=None,
    seq_len: int = 100,
    method: str = 'spline',
    S1_feed: float = None,
    S2_feed: float = None,
    top_k: int = 5,
    device: str = 'cpu',
    **kwargs,
):
    """
    End-to-end: preprocess experimental data → run classifier → print top-k.

    Returns
    -------
    pred_class  : int, top-1 predicted class index
    probs       : (n_classes,) numpy array of softmax probabilities
    label_names : list of class name strings
    """
    tensor, t_out, debug = prepare_experimental_input(
        df, dataset_type=dataset_type, scaler=scaler, seq_len=seq_len,
        method=method, S1_feed=S1_feed, S2_feed=S2_feed, **kwargs,
    )
    model.to(device)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        pred_class, confidence, probs = model.predict_with_confidence(tensor)

    pred_class = pred_class.item()
    confidence = confidence.item()
    probs_np   = probs.squeeze(0).cpu().numpy()

    top_k      = min(top_k, len(label_names))
    top_idx    = np.argsort(probs_np)[::-1][:top_k]

    print(f"Top-{top_k} predictions (method: {method}):")
    for rank, idx in enumerate(top_idx):
        bar = '█' * int(probs_np[idx] * 30)
        print(f"  {rank+1}. {label_names[idx]:<12s}  {probs_np[idx]:.4f}  {bar}")

    return pred_class, probs_np, debug
