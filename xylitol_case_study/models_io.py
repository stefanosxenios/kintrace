"""Load Case Study 2 classifier / regressor checkpoints."""

import os
import torch
from LSTMs.lstm_classifier import LSTMClassifier
from LSTMs.lstm_regressor import SharedEncoderRegressor

HIDDEN_DIM = 128       # v2 pipeline uses 128 for the dual-substrate system
NUM_LAYERS = 2


def build_classifier(input_dim, n_classes, hidden=HIDDEN_DIM, layers=NUM_LAYERS):
    return LSTMClassifier(input_dim=input_dim, hidden_dim=hidden, num_layers=layers,
                          output_dim=n_classes, dropout=0.3,
                          bidirectional=True, use_attention=True)


def build_regressor(input_dim, model_param_counts, hidden=HIDDEN_DIM, layers=NUM_LAYERS):
    return SharedEncoderRegressor(input_dim=input_dim, hidden_dim=hidden, num_layers=layers,
                                  model_param_counts=model_param_counts, dropout=0.3,
                                  bidirectional=True, use_attention=True)


def load_classifier(path, input_dim, n_classes, device="cpu", **kw):
    m = build_classifier(input_dim, n_classes, **kw)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def load_regressor(path, input_dim, model_param_counts, device="cpu", **kw):
    m = build_regressor(input_dim, model_param_counts, **kw)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


class PerMechRegressorBank(torch.nn.Module):
    """One independently trained regressor per mechanism, behind the same
    interface as SharedEncoderRegressor (forward / predict_physical take the
    mechanism id and dispatch to that mechanism's own network)."""

    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleDict({str(k): v for k, v in nets.items()})

    def forward(self, x, model_id):
        return self.nets[str(model_id)](x, model_id)

    def predict_physical(self, x, model_id, lbs, ubs):
        return self.nets[str(model_id)].predict_physical(x, model_id, lbs, ubs)


def load_permech_bank(results_dir, input_dim, model_param_counts, device="cpu",
                      hidden=HIDDEN_DIM, layers=NUM_LAYERS):
    """Load regressor_permech_model{mid}.pt for every mechanism in the library."""
    nets = {}
    for mid, n_params in model_param_counts.items():
        m = build_regressor(input_dim, {int(mid): n_params}, hidden=hidden, layers=layers)
        path = os.path.join(results_dir, f"regressor_permech_model{int(mid)}.pt")
        m.load_state_dict(torch.load(path, map_location=device))
        nets[int(mid)] = m.to(device).eval()
    return PerMechRegressorBank(nets).to(device).eval()
