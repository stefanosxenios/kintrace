"""
LSTM Classifier for bioprocess kinetic mechanism identification.

Changes from v1:
  - Attention layer over all hidden states (not just final hn[-1])
  - Bidirectional LSTM option
  - Softmax probability / confidence output
  - Optional attention weight return for interpretability
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention over an LSTM output sequence.
    Learns a scalar score for each timestep and returns a weighted context vector.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        """
        Parameters
        ----------
        lstm_out : (batch, T, hidden_dim)

        Returns
        -------
        context : (batch, hidden_dim)  weighted sum over time
        weights : (batch, T)           attention weights (sum to 1)
        """
        scores  = self.score(lstm_out)              # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)       # (batch, T, 1)
        context = (weights * lstm_out).sum(dim=1)    # (batch, hidden_dim)
        return context, weights.squeeze(-1)


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier with optional attention and bidirectionality.

    Parameters
    ----------
    input_dim     : number of input features per timestep
    hidden_dim    : LSTM hidden size
    num_layers    : number of stacked LSTM layers
    output_dim    : number of classes
    dropout       : dropout probability (applied between layers and in head)
    bidirectional : if True, use bidirectional LSTM (doubles effective hidden dim)
    use_attention : if True, use attention pooling instead of last hidden state
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.bidirectional  = bidirectional
        self.use_attention  = use_attention
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        head_input_dim = hidden_dim * self.num_directions

        if use_attention:
            self.attention = Attention(head_input_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Parameters
        ----------
        x                : (batch, T, input_dim)
        return_attention : if True, also return attention weights

        Returns
        -------
        logits  : (batch, output_dim)
        weights : (batch, T)  — only if return_attention=True
        """
        lstm_out, (hn, _) = self.lstm(x)
        # lstm_out: (batch, T, hidden_dim * num_directions)

        if self.use_attention:
            context, weights = self.attention(lstm_out)
        else:
            if self.bidirectional:
                # Concatenate last forward and last backward hidden states
                context = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                context = hn[-1]
            weights = None

        logits = self.classifier(context)

        if return_attention:
            return logits, weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities. Shape: (batch, output_dim)."""
        return torch.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(self, x: torch.Tensor):
        """
        Returns
        -------
        pred_class : (batch,)  predicted class index
        confidence : (batch,)  probability of predicted class
        probs      : (batch, output_dim)  full probability distribution
        """
        probs = self.predict_proba(x)
        confidence, pred_class = probs.max(dim=-1)
        return pred_class, confidence, probs
