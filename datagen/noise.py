"""
Gaussian noise injection for simulated trajectories.

Noise is scaled to the dynamic range of each feature:
    sigma_j = noise_pct * (max_j - min_j)

This ensures consistent relative noise across features with very
different absolute magnitudes (e.g. V in L vs X in g/L).

For concentrations:  values are clipped to >= 0 after noise addition.
For rates:           no clipping (rates can legitimately be negative,
                     e.g. net negative growth rate when kd > mu).
"""

import numpy as np


def add_gaussian_noise(
    data: np.ndarray,
    noise_pct: float = 0.05,
    clip_zero: bool = True,
    seed: int = None,
) -> np.ndarray:
    """
    Add percentage-based Gaussian noise scaled per feature.

    Parameters
    ----------
    data       : (T, F) single trajectory or (N, T, F) batch
    noise_pct  : fraction of dynamic range used as noise std (default 0.05 = 5%)
    clip_zero  : if True, clip output to >= 0 (use for concentrations, not rates)
    seed       : random seed

    Returns
    -------
    noisy : same shape as input
    """
    rng = np.random.default_rng(seed)

    if data.ndim == 2:
        # Single trajectory (T, F): compute range over time axis
        drange = data.max(axis=0) - data.min(axis=0)
    elif data.ndim == 3:
        # Batch (N, T, F): compute range over (N, T) axes
        drange = data.max(axis=(0, 1)) - data.min(axis=(0, 1))
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    # Floor to avoid zero-scale noise on constant features (e.g. P0 at t=0)
    scale = noise_pct * np.maximum(drange, 1e-6)

    noise = rng.normal(0.0, scale, size=data.shape)
    noisy = data + noise

    if clip_zero:
        noisy = np.clip(noisy, 0.0, None)

    return noisy
