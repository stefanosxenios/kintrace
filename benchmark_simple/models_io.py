"""Load trained Stage-1 / Stage-2 checkpoints for the simple benchmark.

Stage 2 comes in two flavours, both exposing the SAME call signature
``model(X, mid)`` so everything downstream (refinement, OOD figures, attention
overlays) works with either:

  * shared encoder      — one BiLSTM+attention encoder, one head per mechanism
                          (single ``regressor.pt``)
  * per-mechanism bank  — one independent BiLSTM+attention network per
                          mechanism, trained only on that mechanism's
                          trajectories (``regressor_permech_m{mid}.pt``)

``load_regressor`` auto-detects: if a directory has no ``regressor.pt`` but does
have the full set of per-mechanism checkpoints, it returns the bank. Passing a
directory as ``ckpt`` forces the bank.
"""

import os
import torch

from benchmark_simple import data as D
from kinetic_models.simple_models import get_param_names
from LSTMs.lstm_classifier import LSTMClassifier
from LSTMs.lstm_regressor import SharedEncoderRegressor


def load_classifier(results_dir, input_dim, device="cpu", hidden_dim=64, num_layers=2,
                    ckpt=None):
    """Load a Stage-1 classifier. By default reads `results_dir/classifier.pt`;
    pass `ckpt` to load a specific checkpoint (e.g. the renamed paper-bundle
    weights in results/paper_benchmark/models/classifier_nondim_hybrid.pt)."""
    path = ckpt or os.path.join(results_dir, "classifier.pt")
    m = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                       output_dim=len(D.MODEL_IDS), dropout=0.3,
                       bidirectional=True, use_attention=True)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


PERMECH_PATTERN = "regressor_permech_m{mid}.pt"


class PerMechRegressorBank(torch.nn.Module):
    """Dedicated per-mechanism regressors behind the shared-encoder interface.

    Holds one independent network per mechanism and dispatches ``bank(X, mid)``
    to that mechanism's own weights, so it is a drop-in replacement for
    SharedEncoderRegressor everywhere in the pipeline.
    """

    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleDict({str(k): v for k, v in nets.items()})

    def forward(self, x, mid):
        return self.nets[str(mid)](x, mid)


def has_per_mechanism(results_dir, pattern=PERMECH_PATTERN):
    """True if `results_dir` holds a complete set of per-mechanism checkpoints."""
    if not results_dir or not os.path.isdir(results_dir):
        return False
    return all(os.path.exists(os.path.join(results_dir, pattern.format(mid=mid)))
               for mid in D.MODEL_IDS)


def load_per_mechanism_regressor(results_dir, input_dim, device="cpu", hidden_dim=64,
                                 num_layers=2, pattern=PERMECH_PATTERN):
    """Load the bank of dedicated per-mechanism Stage-2 regressors."""
    nets = {}
    for mid in D.MODEL_IDS:
        n_params = len(get_param_names(mid))
        m = SharedEncoderRegressor(input_dim=input_dim, hidden_dim=hidden_dim,
                                   num_layers=num_layers,
                                   model_param_counts={mid: n_params},
                                   dropout=0.3, bidirectional=True, use_attention=True)
        m.load_state_dict(torch.load(os.path.join(results_dir, pattern.format(mid=mid)),
                                     map_location=device))
        nets[mid] = m.to(device).eval()
    return PerMechRegressorBank(nets).to(device).eval()


def load_regressor(results_dir, input_dim, device="cpu", hidden_dim=64, num_layers=2,
                   ckpt=None):
    """Load a Stage-2 regressor, shared-encoder or per-mechanism bank.

    By default reads `results_dir/regressor.pt`, falling back to the
    per-mechanism bank in the same directory when that file is absent. Pass
    `ckpt` as a file for a specific shared checkpoint, or as a DIRECTORY to
    force loading the per-mechanism bank from it.
    """
    if ckpt and os.path.isdir(ckpt):
        return load_per_mechanism_regressor(ckpt, input_dim, device, hidden_dim, num_layers)
    path = ckpt or os.path.join(results_dir, "regressor.pt")
    if ckpt is None and not os.path.exists(path) and has_per_mechanism(results_dir):
        return load_per_mechanism_regressor(results_dir, input_dim, device,
                                            hidden_dim, num_layers)
    counts = {mid: len(get_param_names(mid)) for mid in D.MODEL_IDS}
    m = SharedEncoderRegressor(input_dim=input_dim, hidden_dim=hidden_dim,
                               num_layers=num_layers, model_param_counts=counts,
                               dropout=0.3, bidirectional=True, use_attention=True)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()
