"""
End-to-end trainer for Case Study 2 (xylitol, 11 mechanisms 21-31).

Stage 1 : scale-invariant (per-trajectory) classifier over the 11 mechanisms.
Stage 2 : parameter regressor, in TWO flavours selectable with --regressor:
            shared        -> one shared-encoder network with 11 heads
            per_mechanism -> 11 dedicated LSTMs (see train_per_mechanism.py)

Results are written to results/xylitol_use_case/<tag>/.

Usage
-----
  # scale-invariant classifier
  python xylitol_case_study/run.py --stage classifier --rep hybrid --scaling per_traj --tag main --epochs 120
  # shared-encoder regressor
  python xylitol_case_study/run.py --stage regressor  --rep hybrid --regressor shared --tag main --epochs 120
  # dedicated per-mechanism regressors
  python xylitol_case_study/run.py --stage regressor  --rep hybrid --regressor per_mechanism --tag main --epochs 120
  python xylitol_case_study/run.py --stage assemble --tag main
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from xylitol_case_study import data as XD
from xylitol_case_study.models_io import build_classifier, build_regressor, HIDDEN_DIM, NUM_LAYERS
from LSTMs.train_classifier import train_classifier
from LSTMs.train_regressor import train_shared_regressor
from dataloader.in_silico_loader_v2 import load_regressor_dataloaders
from datagen.param_ranges_v2 import get_param_names, get_param_bounds

DATA_DIR = os.path.join(_ROOT, "data", "generated_v2")


def out_dir(tag):
    d = os.path.join(_ROOT, "results", "xylitol_use_case", tag)
    os.makedirs(d, exist_ok=True)
    return d


# ── Stage 1: classifier ──────────────────────────────────────────────────────

def run_classifier(args, device):
    OUT = out_dir(args.tag)
    tr, val, te, transform, meta, (Xte, yte) = XD.load_classifier_data(
        DATA_DIR, rep=args.rep, seq_len=args.seq_len, scaling=args.scaling,
        batch_size=args.batch, seed=args.seed, max_per_model=args.max_per_model)

    torch.manual_seed(args.seed)
    model = build_classifier(meta["input_dim"], meta["n_classes"], hidden=args.hidden)
    model, _ = train_classifier(model, tr, val, n_epochs=args.epochs, patience=max(args.epochs, 15),
                                lr=args.lr, device=device, verbose=True)

    model.eval()
    with torch.no_grad():
        pred, conf, _ = model.predict_with_confidence(torch.tensor(Xte).to(device))
    pred = pred.cpu().numpy(); acc = float((pred == yte).mean())
    K = meta["n_classes"]
    cm = np.zeros((K, K), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1

    torch.save(model.state_dict(), os.path.join(OUT, "classifier.pt"))
    with open(os.path.join(OUT, "classifier_scaler.pkl"), "wb") as f:
        pickle.dump(transform, f)
    json.dump({"test_acc": acc, "confusion": cm.tolist(), "label_names": meta["label_names"],
               "n_classes": K, "input_dim": meta["input_dim"], "scaling": args.scaling,
               "rep": args.rep, "seq_len": args.seq_len},
              open(os.path.join(OUT, "_classifier.json"), "w"), indent=2)
    _plot_confusion(cm, acc, meta["label_names"], OUT)
    print(f"[classifier] test accuracy = {acc:.3f}  ({args.scaling}, {args.rep})")
    return model


def _plot_confusion(cm, acc, names, OUT):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Stage-1 confusion (11 mechanisms, acc {acc:.2f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "confusion.png"), dpi=150)
    plt.close(fig)


# ── Stage 2: regressor ───────────────────────────────────────────────────────

def _r2_per_model(model, test_loaders, meta, device):
    out = {}
    model.eval()
    for mid, loader in test_loaders.items():
        names = meta["param_names"][mid]
        lbs = np.array(meta["bounds"][mid]["lbs"]); ubs = np.array(meta["bounds"][mid]["ubs"])
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                p = model(xb.to(device), mid).cpu().numpy()
                preds.append(p); trues.append(yb.numpy())
        P = np.concatenate(preds); Y = np.concatenate(trues)
        Pp = P * (ubs - lbs) + lbs; Yp = Y * (ubs - lbs) + lbs
        r2 = {}
        for k, nm in enumerate(names):
            ss_res = np.sum((Yp[:, k] - Pp[:, k]) ** 2); ss_tot = np.sum((Yp[:, k] - Yp[:, k].mean()) ** 2)
            r2[nm] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        out[str(mid)] = {"r2": r2, "r2_mean": float(np.nanmean(list(r2.values())))}
    return out


def run_regressor_shared(args, device):
    OUT = out_dir(args.tag)
    tr, val, te, x_scaler, meta = load_regressor_dataloaders(
        DATA_DIR, dataset_type=args.rep, seq_len=args.seq_len,
        batch_size=args.batch, seed=args.seed)
    torch.manual_seed(args.seed)
    model = build_regressor(meta["input_dim"], meta["model_param_counts"], hidden=args.hidden)
    model, _ = train_shared_regressor(model, tr, val, n_epochs=args.epochs,
                                      patience=max(args.epochs, 15), lr=args.lr,
                                      device=device, verbose=True)
    torch.save(model.state_dict(), os.path.join(OUT, "regressor_shared.pt"))
    with open(os.path.join(OUT, "regressor_x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    # meta with str keys for json
    reg_meta = {"input_dim": meta["input_dim"], "seq_len": args.seq_len,
                "param_names": {str(k): v for k, v in meta["param_names"].items()},
                "model_param_counts": {str(k): v for k, v in meta["model_param_counts"].items()},
                "bounds": {str(k): v for k, v in meta["bounds"].items()}}
    json.dump(reg_meta, open(os.path.join(OUT, "regressor_meta.json"), "w"), indent=2)
    r2 = _r2_per_model(model, te, meta, device)
    json.dump(r2, open(os.path.join(OUT, "_regressor_shared.json"), "w"), indent=2)
    mean = float(np.mean([r2[k]["r2_mean"] for k in r2]))
    print(f"[regressor:shared] mean R2 across models = {mean:.3f}")
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["classifier", "regressor", "assemble"])
    ap.add_argument("--rep", default="hybrid", choices=["concentrations", "rates", "hybrid"])
    ap.add_argument("--scaling", default="per_traj", choices=["per_traj", "global"],
                    help="classifier input scaling (per_traj = scale-invariant, the new setup)")
    ap.add_argument("--regressor", default="per_mechanism", choices=["per_mechanism", "shared"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-model", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    print(f"device={device}  stage={args.stage}  rep={args.rep}  scaling={args.scaling}")

    if args.stage == "classifier":
        run_classifier(args, device)
    elif args.stage == "regressor":
        if args.regressor == "shared":
            run_regressor_shared(args, device)
        else:
            from xylitol_case_study.train_per_mechanism import train_all
            train_all(rep=args.rep, tag=args.tag, epochs=args.epochs, seq_len=args.seq_len,
                      batch=args.batch, hidden=args.hidden, lr=args.lr, seed=args.seed,
                      device=device)
    elif args.stage == "assemble":
        OUT = out_dir(args.tag)
        summary = {}
        for name in ["_classifier.json", "_regressor_shared.json", "_regressor_permechanism.json"]:
            p = os.path.join(OUT, name)
            if os.path.exists(p):
                summary[name] = json.load(open(p))
        json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)
        print(f"assembled results.json in {OUT}")


if __name__ == "__main__":
    main()
