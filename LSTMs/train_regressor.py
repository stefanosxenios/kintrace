"""
Training loop for the SharedEncoderRegressor.

Training strategy — interleaved multi-model batches:
  Each optimiser step accumulates gradients from one mini-batch
  per model before updating. This ensures the shared encoder
  receives simultaneous gradient signals from all 15 models,
  learning a representation that is useful across all of them.

  One "epoch" = one full pass through the shortest model's dataset.
  Longer datasets are cycled (infinite iterator) so every model
  contributes equally each step.
"""

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _cycle(loader):
    """Infinite iterator over a DataLoader."""
    while True:
        for batch in loader:
            yield batch


def train_shared_regressor(
    model,
    train_loaders: dict,
    val_loaders: dict,
    n_epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    lr_patience: int = 8,
    lr_factor: float = 0.5,
    clip_grad_norm: float = 1.0,
    device: str = "cpu",
    use_early_stopping: bool = True,
    verbose: bool = True,
):
    """
    Train the SharedEncoderRegressor.

    Parameters
    ----------
    model         : SharedEncoderRegressor instance
    train_loaders : dict  model_id (int) → DataLoader  (X_norm, y_norm)
    val_loaders   : dict  model_id (int) → DataLoader
    n_epochs      : maximum training epochs
    patience      : early stopping patience (epochs without improvement)
    lr            : initial learning rate
    lr_patience   : epochs before LR is halved on plateau
    lr_factor     : LR reduction factor
    clip_grad_norm: max gradient norm (None to disable)
    device        : 'cpu', 'cuda', or 'mps'
    verbose       : print epoch summaries

    Returns
    -------
    model   : best model by average validation loss
    history : dict  train_loss / val_loss / val_loss_per_model / lr  (per epoch)
    """
    model.to(device)
    model_ids = list(train_loaders.keys())

    optimizer  = Adam(model.parameters(), lr=lr)
    scheduler  = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor,
        patience=lr_patience, verbose=False,
    )
    criterion  = MSELoss()

    # Steps per epoch = batches in the smallest dataset
    steps_per_epoch = min(len(loader) for loader in train_loaders.values())
    if steps_per_epoch == 0:
        raise ValueError(
            "A training split is smaller than one batch (loaders use drop_last=True). "
            "Lower --batch, or generate more trajectories per mechanism.")

    # Infinite cycled iterators so longer datasets wrap around
    train_iters = {mid: _cycle(loader) for mid, loader in train_loaders.items()}

    history = {
        "train_loss": [], "val_loss": [],
        "val_loss_per_model": {mid: [] for mid in model_ids},
        "lr": [],
    }
    best_val_loss = float("inf")
    best_state    = None
    counter       = 0

    for epoch in range(1, n_epochs + 1):

        # ── Training ──────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=device)

            for mid in model_ids:
                xb, yb = next(train_iters[mid])
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb, mid)
                step_loss = step_loss + criterion(preds, yb)

            # Average over models before backward
            (step_loss / len(model_ids)).backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )
            optimizer.step()
            epoch_loss += step_loss.item() / len(model_ids)

        train_loss = epoch_loss / steps_per_epoch

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_losses_per_model = {}
        with torch.no_grad():
            for mid, loader in val_loaders.items():
                mid_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds    = model(xb, mid)
                    mid_loss += criterion(preds, yb).item()
                val_losses_per_model[mid] = mid_loss / len(loader)

        val_loss = sum(val_losses_per_model.values()) / len(val_losses_per_model)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for mid in model_ids:
            history["val_loss_per_model"][mid].append(val_losses_per_model[mid])

        if verbose:
            worst_mid = max(val_losses_per_model, key=val_losses_per_model.get)
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train {train_loss:.5f} | Val {val_loss:.5f} | "
                f"Worst model{worst_mid}={val_losses_per_model[worst_mid]:.5f} | "
                f"LR {current_lr:.2e}"
            )

        scheduler.step(val_loss)

        # ── Early stopping ────────────────────────────────────────
        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(
                            f"  Early stopping at epoch {epoch} "
                            f"(best val {best_val_loss:.5f})"
                        )
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history
