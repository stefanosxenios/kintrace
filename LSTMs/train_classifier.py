"""
Training loop for the LSTM kinetic mechanism classifier.

Changes from v1:
  - ReduceLROnPlateau learning rate scheduler
  - Gradient clipping (clip_grad_norm = 1.0)
  - Returns full history: train loss, val loss, val accuracy per epoch
  - Cleaned duplicate imports
  - Optional class weights for imbalanced datasets
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_classifier(
    model,
    train_loader,
    val_loader,
    n_epochs: int = 150,
    patience: int = 15,
    lr: float = 1e-3,
    lr_patience: int = 7,
    lr_factor: float = 0.5,
    clip_grad_norm: float = 1.0,
    class_weights: torch.Tensor = None,
    device: str = "cpu",
    use_early_stopping: bool = True,
    verbose: bool = True,
):
    """
    Train the LSTMClassifier.

    Parameters
    ----------
    model           : LSTMClassifier instance
    train_loader    : DataLoader for training set
    val_loader      : DataLoader for validation set
    n_epochs        : maximum training epochs
    patience        : early stopping patience (epochs without val loss improvement)
    lr              : initial learning rate
    lr_patience     : epochs before LR is reduced on plateau
    lr_factor       : factor to multiply LR by when reducing (0.5 = halve)
    clip_grad_norm  : max gradient norm (set to None to disable)
    class_weights   : optional (n_classes,) tensor for imbalanced datasets
    device          : "cpu", "cuda", or "mps"
    use_early_stopping : whether to apply early stopping
    verbose         : print epoch summaries

    Returns
    -------
    model       : best model (by val loss)
    history     : dict with keys train_loss, val_loss, val_acc, lr
    """
    model.to(device)

    criterion = CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor,
        patience=lr_patience, verbose=False,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_state    = None
    counter       = 0

    for epoch in range(1, n_epochs + 1):

        # ── Training ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                val_loss += criterion(logits, yb).item()
                preds    = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct / total
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if verbose:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"Acc {val_acc:.4f} | LR {current_lr:.2e}"
            )

        # ── LR scheduling ─────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Early stopping ────────────────────────────────────────────
        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                counter       = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best val loss {best_val_loss:.4f})")
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history
