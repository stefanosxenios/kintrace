"""
Dedicated per-mechanism regressors for Case Study 2 — one BiLSTM+attention
network per mechanism (21-31), identical architecture, separate weights, trained
only on that mechanism's data. Alternative to the shared-encoder regressor.

Reuses the v2 regressor dataloaders (global scaling, targets min-max scaled to
the LHS bounds). Saves each model to results/xylitol_use_case/<tag>/ and writes
per-parameter R^2 to _regressor_permechanism.json.

Run standalone:
  python xylitol_case_study/train_per_mechanism.py --tag main --epochs 120
or via run.py:
  python xylitol_case_study/run.py --stage regressor --regressor per_mechanism --tag main --epochs 120
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from xylitol_case_study.models_io import build_regressor, HIDDEN_DIM
from LSTMs.train_regressor import train_shared_regressor
from dataloader.in_silico_loader_v2 import load_regressor_dataloaders

DATA_DIR = os.path.join(_ROOT, "data", "generated_v2")


def _r2(model, loader, mid, names, lbs, ubs, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device), mid).cpu().numpy()); trues.append(yb.numpy())
    P = np.concatenate(preds) * (ubs - lbs) + lbs
    Y = np.concatenate(trues) * (ubs - lbs) + lbs
    out = {}
    for k, nm in enumerate(names):
        ss_res = np.sum((Y[:, k] - P[:, k]) ** 2); ss_tot = np.sum((Y[:, k] - Y[:, k].mean()) ** 2)
        out[nm] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return out


def train_all(rep="hybrid", tag="main", epochs=120, seq_len=100, batch=256,
              hidden=HIDDEN_DIM, lr=1e-3, seed=42, device="cpu"):
    OUT = os.path.join(_ROOT, "results", "xylitol_use_case", tag)
    os.makedirs(OUT, exist_ok=True)
    tr, val, te, x_scaler, meta = load_regressor_dataloaders(
        DATA_DIR, dataset_type=rep, seq_len=seq_len, batch_size=batch, seed=seed)

    # Real-data inference needs the input scaler and the per-mechanism parameter
    # bounds; write them here so this path is self-sufficient.
    with open(os.path.join(OUT, "regressor_x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    reg_meta = {"input_dim": meta["input_dim"], "seq_len": seq_len,
                "param_names": {str(k): v for k, v in meta["param_names"].items()},
                "model_param_counts": {str(k): v for k, v in meta["model_param_counts"].items()},
                "bounds": {str(k): v for k, v in meta["bounds"].items()}}
    json.dump(reg_meta, open(os.path.join(OUT, "regressor_meta.json"), "w"), indent=2)

    results_path = os.path.join(OUT, "_regressor_permechanism.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for mid in meta["model_ids"]:
        ckpt = os.path.join(OUT, f"regressor_permech_model{mid}.pt")
        if os.path.exists(ckpt) and str(mid) in results:
            print(f"skip model{mid} (exists)"); continue
        n_params = meta["model_param_counts"][mid]
        torch.manual_seed(seed)
        model = build_regressor(meta["input_dim"], {mid: n_params}, hidden=hidden)
        model, _ = train_shared_regressor(
            model, {mid: tr[mid]}, {mid: val[mid]}, n_epochs=epochs,
            patience=max(epochs, 15), lr=lr, device=device, verbose=False)
        torch.save(model.state_dict(), ckpt)
        names = meta["param_names"][mid]
        lbs = np.array(meta["bounds"][mid]["lbs"]); ubs = np.array(meta["bounds"][mid]["ubs"])
        r2 = _r2(model, te[mid], mid, names, lbs, ubs, device)
        results[str(mid)] = {"r2": r2, "r2_mean": float(np.nanmean(list(r2.values())))}
        json.dump(results, open(results_path, "w"), indent=2)
        print(f"model{mid}: mean R2 = {results[str(mid)]['r2_mean']:.3f}")

    mean = float(np.mean([results[k]["r2_mean"] for k in results]))
    print(f"[regressor:per_mechanism] mean R2 across models = {mean:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="hybrid")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    train_all(rep=args.rep, tag=args.tag, epochs=args.epochs, seq_len=args.seq_len,
              batch=args.batch, hidden=args.hidden, lr=args.lr, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
