"""
Data loading for Case Study 2 (dual-substrate xylitol bioconversion).

The mechanism library is the 11 models 21-31 (datagen/param_ranges_v2.py),
5-state system [X, S1, S2, P, V]. Representations produced by
datagen/generate_v2.py:
    concentrations : [X, S1, S2, P, V, F/V]   (6)
    rates          : [mu, qS1, qS2, qP, F/V]  (5)   (F/V appended from conc)
    hybrid         : concatenation            (11)

This module provides the CASE-STUDY-2 classifier data with the *new*
scale-invariant setup: per-trajectory min-max normalisation (nondimensional),
so mechanism identification depends on trajectory shape, not absolute
concentration magnitude. A `global` StandardScaler option is kept for ablation.

The Stage-2 regressor is unchanged from the v2 pipeline and reuses
`dataloader.in_silico_loader_v2.load_regressor_dataloaders` (global scaling,
targets min-max scaled to the LHS bounds).
"""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from datagen.param_ranges_v2 import PARAM_RANGES, get_param_names, get_param_bounds

MODEL_IDS = list(range(21, 32))                      # 21 .. 31
LABEL_NAMES = [f"model{m}" for m in MODEL_IDS]
ID_TO_CLASS = {m: i for i, m in enumerate(MODEL_IDS)}
CLASS_TO_ID = {i: m for m, i in ID_TO_CLASS.items()}


# ── scale-invariant (nondimensional) normalisation ───────────────────────────

def normalise_per_trajectory(X, eps=1e-3):
    """Per-trajectory, per-channel min-max to [0,1]. X: (N,T,F) -> (N,T,F) float32."""
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    rng = mx - mn
    safe = rng > (eps * np.maximum(np.abs(mx), 1.0))
    out = np.where(safe, (X - mn) / np.where(safe, rng, 1.0), 0.0)
    return out.astype(np.float32)


class PerTrajectoryScaler:
    """Scaler-like object so the real-data inference path can apply the same
    scale-invariant transform. `.transform` accepts (T,F) or (N,T,F)."""

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:                              # single trajectory (T,F)
            return normalise_per_trajectory(X[None])[0]
        return normalise_per_trajectory(X)

    # StandardScaler-compatible no-ops (nothing is fit)
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)


# ── in-silico loading ────────────────────────────────────────────────────────

def _load_rep(data_dir, rep, model_ids, max_per_model=None):
    """dict model_id -> (N,T,F). Uses mmap and slices `max_per_model` BEFORE
    materialising, so a capped run never reads the whole (large) array into RAM.
    Appends F/V to the 4-channel rates rep, matching dataloader.in_silico_loader_v2."""
    out = {}
    for mid in model_ids:
        X = np.load(os.path.join(data_dir, rep, f"model{mid}.npy"), mmap_mode="r")
        if max_per_model is not None:
            X = X[:max_per_model]
        X = np.asarray(X)                                        # materialise (slice only)
        if rep == "rates" and X.shape[2] == 4:
            conc = np.load(os.path.join(data_dir, "concentrations", f"model{mid}.npy"),
                           mmap_mode="r")
            if max_per_model is not None:
                conc = conc[:max_per_model]
            X = np.concatenate([X, np.asarray(conc[:, :, 5:6])], axis=2)   # F/V = conc idx 5
        out[mid] = X
    return out


def load_classifier_data(data_dir, rep="hybrid", seq_len=100, scaling="per_traj",
                         train_frac=0.70, val_frac=0.15, batch_size=256, seed=42,
                         max_per_model=None):
    """
    Build Stage-1 classifier train/val/test loaders for the 11-mechanism system.

    scaling : "per_traj" (scale-invariant, the new setup) or "global" (ablation).
    Returns (train_loader, val_loader, test_loader, transform, meta, (X_test, y_test)).
    """
    per = _load_rep(data_dir, rep, MODEL_IDS, max_per_model=max_per_model)
    Xs, ys = [], []
    for mid in MODEL_IDS:
        X = per[mid]
        Xs.append(X)
        ys.append(np.full(len(X), ID_TO_CLASS[mid], dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    T = X.shape[1]
    if seq_len < T:
        idx = np.round(np.linspace(0, T - 1, seq_len)).astype(int)
        X = X[:, idx, :]
    F = X.shape[2]

    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=1 - train_frac, stratify=y, random_state=seed)
    rel = val_frac / (1 - train_frac)
    Xval, Xte, yval, yte = train_test_split(
        Xtmp, ytmp, test_size=1 - rel, stratify=ytmp, random_state=seed)

    if scaling == "global":
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr.reshape(-1, F)).reshape(Xtr.shape).astype(np.float32)
        Xval = sc.transform(Xval.reshape(-1, F)).reshape(Xval.shape).astype(np.float32)
        Xte = sc.transform(Xte.reshape(-1, F)).reshape(Xte.shape).astype(np.float32)
        transform = sc
    else:  # per_traj (default, scale-invariant)
        Xtr, Xval, Xte = (normalise_per_trajectory(a) for a in (Xtr, Xval, Xte))
        transform = PerTrajectoryScaler()

    def _dl(Xa, ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa, dtype=torch.float32), torch.tensor(ya))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

    meta = dict(input_dim=F, n_classes=len(MODEL_IDS), label_names=LABEL_NAMES,
                label_map={m: ID_TO_CLASS[m] for m in MODEL_IDS},
                dataset_type=rep, seq_len=seq_len, scaling=scaling)
    return _dl(Xtr, ytr, True), _dl(Xval, yval, False), _dl(Xte, yte, False), \
        transform, meta, (Xte.astype(np.float32), yte)


def param_meta():
    """param_names, bounds and counts for the regressor heads (models 21-31)."""
    names = {mid: get_param_names(mid) for mid in MODEL_IDS}
    bounds = {}
    for mid in MODEL_IDS:
        lbs, ubs = get_param_bounds(mid)
        bounds[mid] = {"lbs": lbs, "ubs": ubs}
    counts = {mid: len(PARAM_RANGES[mid]) for mid in MODEL_IDS}
    return names, bounds, counts
