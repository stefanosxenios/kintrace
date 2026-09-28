"""
Shared-encoder LSTM regressor for bioprocess kinetic parameter estimation.

Architecture
------------
                 ┌─────────────────────────────┐
  (batch,T,F) → │  Shared BiLSTM + Attention   │ → context (batch, hidden*2)
                 └─────────────────────────────┘
                           │
              ┌────────────┼────────────┐
          head_3        head_21     head_20  ...  (15 model-specific heads)
              │
    Linear → ReLU → Dropout → Linear → Sigmoid
              │
    output ∈ [0,1]  (normalised parameter space)

Why normalise to [0,1]?
  - Each kinetic parameter lives on a very different scale
    (mumax1 ∈ [0.2,1.2] vs KP ∈ [10,100]).
  - MSE in raw space is dominated by large-scale parameters.
  - Normalising y to [0,1] using the LHS sample bounds makes
    the loss fair across all parameters.
  - Sigmoid output guarantees predictions stay within the
    sampled bounds — physically impossible values are impossible.
  - For evaluation, inverse-transform back to physical space.

Training
--------
  One optimiser step per "round": accumulate gradients from one
  mini-batch per model, then step. The shared encoder is updated
  by signals from all 15 models simultaneously each step.
"""

import torch
import torch.nn as nn


# ── Reuse the same attention module as the classifier ────────────────────────

class Attention(nn.Module):
    """Additive attention over LSTM output sequence."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out : (batch, T, hidden_dim)
        scores  = self.score(lstm_out)           # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)   # (batch, T, 1)
        context = (weights * lstm_out).sum(dim=1) # (batch, hidden_dim)
        return context, weights.squeeze(-1)


# ── Model-specific head ───────────────────────────────────────────────────────

class ModelHead(nn.Module):
    """
    Small MLP head for one kinetic model.

    Input  : context vector from the shared encoder (batch, encoder_dim)
    Output : (batch, n_params)  values in [0, 1] (sigmoid activated)

    The [0,1] output is a normalised representation of the kinetic
    parameters. Inverse-transform to physical space at evaluation time:
        physical = output * (ub - lb) + lb
    """

    def __init__(self, encoder_dim: int, n_params: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_params),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(context))   # (batch, n_params) ∈ [0,1]


# ── Shared encoder + model-specific heads ────────────────────────────────────

class SharedEncoderRegressor(nn.Module):
    """
    Shared BiLSTM encoder + one head per kinetic model.

    Parameters
    ----------
    input_dim      : number of input features per timestep
    hidden_dim     : LSTM hidden size (per direction)
    num_layers     : number of stacked LSTM layers
    model_param_counts : dict  model_id (int) → n_params (int)
    dropout        : dropout applied inside LSTM and in heads
    bidirectional  : if True, use bidirectional LSTM
    use_attention  : if True, attention pooling over all timesteps
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        model_param_counts: dict,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        num_directions = 2 if bidirectional else 1

        # ── Shared encoder ──────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        encoder_dim = hidden_dim * num_directions

        if use_attention:
            self.attention = Attention(encoder_dim)

        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # ── Model-specific heads ────────────────────────────────────
        # ModuleDict requires string keys
        self.heads = nn.ModuleDict({
            str(mid): ModelHead(encoder_dim, n_params, dropout)
            for mid, n_params in model_param_counts.items()
        })

        self.encoder_dim = encoder_dim

    # ── Encode ───────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, return_attention: bool = False):
        """
        Run the shared LSTM encoder and return the context vector.

        Parameters
        ----------
        x : (batch, T, input_dim)

        Returns
        -------
        context : (batch, encoder_dim)
        weights : (batch, T)  attention weights — only if return_attention=True
        """
        lstm_out, (hn, _) = self.lstm(x)

        if self.use_attention:
            context, weights = self.attention(lstm_out)
        else:
            if self.bidirectional:
                context = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                context = hn[-1]
            weights = None

        context = self.encoder_norm(context)

        if return_attention:
            return context, weights
        return context

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        model_id: int,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (batch, T, input_dim)
        model_id : integer kinetic model ID (e.g. 21, 3, 20 …)

        Returns
        -------
        params_norm : (batch, n_params)  predicted parameters in [0, 1]
        """
        key = str(model_id)
        if key not in self.heads:
            raise KeyError(
                f"No head for model_id={model_id}. "
                f"Available: {list(self.heads.keys())}"
            )

        if return_attention:
            context, weights = self.encode(x, return_attention=True)
            return self.heads[key](context), weights

        context = self.encode(x)
        return self.heads[key](context)

    # ── Convenience ──────────────────────────────────────────────────

    def predict_physical(
        self,
        x: torch.Tensor,
        model_id: int,
        lbs: torch.Tensor,
        ubs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict parameters in physical units.

        Parameters
        ----------
        lbs, ubs : (n_params,) lower/upper bound tensors (on same device as x)

        Returns
        -------
        params : (batch, n_params) in physical units
        """
        norm = self.forward(x, model_id)          # (batch, n_params) ∈ [0,1]
        return norm * (ubs - lbs) + lbs
