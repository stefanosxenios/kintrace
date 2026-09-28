"""
Data loading and preparation for the single-substrate benchmark.

Consumes the arrays written by datagen/generate_simple.py and produces:
  * a stacked, scaled dataset + labels for the Stage-1 classifier
  * per-mechanism scaled tensors + normalised parameter targets for the
    Stage-2 shared-encoder regressor

Feature scaling: a single StandardScaler is fit on the TRAIN split only,
across all mechanisms, and reused everywhere (classifier, regressor, and the
inference path) so the encoder always sees inputs on the same scale.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from kinetic_models.simple_models import SIMPLE_MODELS, get_param_names
from datagen.param_ranges_simple import get_param_bounds

MODEL_IDS = list(SIMPLE_MODELS.keys())
# contiguous 0..K-1 class indices for the classifier
ID_TO_CLASS = {mid: i for i, mid in enumerate(MODEL_IDS)}
CLASS_TO_ID = {i: mid for mid, i in ID_TO_CLASS.items()}
CLASS_NAMES = [SIMPLE_MODELS[CLASS_TO_ID[i]][0] for i in range(len(MODEL_IDS))]


def _rep_dir(root, rep, dataset="generated_simple"):
    return os.path.join(root, "data", dataset, rep)


def load_raw(root, rep="hybrid", dataset="generated_simple"):
    """Return dict model_id -> (X (N,T,F), params (N,P) df, conditions df)."""
    d = _rep_dir(root, rep, dataset)
    out = {}
    for mid in MODEL_IDS:
        X = np.load(os.path.join(d, f"model{mid}.npy"))
        params = pd.read_csv(os.path.join(d, f"params_model{mid}.csv"))
        cond = pd.read_csv(os.path.join(d, f"conditions_model{mid}.csv"))
        out[mid] = (X, params, cond)
    return out


def fit_feature_scaler(train_arrays):
    """Fit a StandardScaler over the pooled (N*T, F) training features."""
    pooled = np.concatenate([a.reshape(-1, a.shape[-1]) for a in train_arrays], axis=0)
    scaler = StandardScaler().fit(pooled)
    return scaler


def apply_scaler(X, scaler):
    N, T, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)


def normalise_per_trajectory(X, eps=1e-3):
    """
    Scale-invariant ("nondimensional") normalisation: per trajectory, per channel,
    min-max map to [0,1] over time. Removes absolute concentration magnitude so the
    classifier sees only the SHAPE of the dynamics. Channels that are ~constant
    (range < eps of their own scale) map to 0 to avoid amplifying noise.

    X : (N, T, F)  ->  (N, T, F) float32 in [0,1]
    """
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    rng = mx - mn
    safe = rng > (eps * np.maximum(np.abs(mx), 1.0))
    out = np.where(safe, (X - mn) / np.where(safe, rng, 1.0), 0.0)
    return out.astype(np.float32)


def normalise_params(params_df, model_id):
    """Map physical params -> [0,1] using the LHS bounds (regressor targets)."""
    lbs, ubs = get_param_bounds(model_id)
    lbs, ubs = np.array(lbs), np.array(ubs)
    vals = params_df[get_param_names(model_id)].values
    return (vals - lbs) / (ubs - lbs)


def denormalise_params(y_norm, model_id):
    lbs, ubs = get_param_bounds(model_id)
    lbs, ubs = np.array(lbs), np.array(ubs)
    return y_norm * (ubs - lbs) + lbs


def make_splits(root, rep="hybrid", val_frac=0.15, test_frac=0.15, seed=42,
                max_per_model=None, dataset="generated_simple", scaling="global"):
    """
    Build train/val/test splits, a feature transform, and everything both
    stages need. Returns a dict.

    max_per_model : optionally cap the number of trajectories used per mechanism.
    scaling       : "global"   -> StandardScaler fit on train (encodes absolute scale)
                    "per_traj" -> per-trajectory min-max (scale-invariant / nondimensional)
    """
    raw = load_raw(root, rep, dataset)

    # per-model index splits (stratified by construction: one model at a time)
    per_model = {}
    train_arrays = []
    for mid in MODEL_IDS:
        X, params, cond = raw[mid]
        N = X.shape[0]
        if max_per_model is not None and N > max_per_model:
            N = max_per_model
            X = X[:N]; params = params.iloc[:N].reset_index(drop=True); cond = cond.iloc[:N].reset_index(drop=True)
        idx = np.arange(N)
        idx_tr, idx_tmp = train_test_split(idx, test_size=val_frac + test_frac,
                                           random_state=seed)
        rel = test_frac / (val_frac + test_frac)
        idx_val, idx_te = train_test_split(idx_tmp, test_size=rel, random_state=seed)
        per_model[mid] = dict(X=X, params=params, cond=cond,
                              tr=idx_tr, val=idx_val, te=idx_te)
        train_arrays.append(X[idx_tr])

    scaler = fit_feature_scaler(train_arrays)

    if scaling == "per_traj":
        transform = normalise_per_trajectory
    else:
        transform = lambda X: apply_scaler(X, scaler)

    return dict(per_model=per_model, scaler=scaler, rep=rep, scaling=scaling,
                transform=transform, model_ids=MODEL_IDS,
                input_dim=train_arrays[0].shape[-1])


def classifier_arrays(splits, which):
    """Stack all mechanisms into (X_transformed, y_class) for a split."""
    transform = splits["transform"]
    Xs, ys = [], []
    for mid in MODEL_IDS:
        pm = splits["per_model"][mid]
        idx = pm[which]
        Xs.append(transform(pm["X"][idx]))
        ys.append(np.full(len(idx), ID_TO_CLASS[mid], dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def regressor_arrays(splits, which):
    """Per-mechanism dict model_id -> (X_transformed, y_norm) for a split."""
    transform = splits["transform"]
    out = {}
    for mid in MODEL_IDS:
        pm = splits["per_model"][mid]
        idx = pm[which]
        X = transform(pm["X"][idx])
        y = normalise_params(pm["params"].iloc[idx], mid)
        out[mid] = (X.astype(np.float32), y.astype(np.float32))
    return out
