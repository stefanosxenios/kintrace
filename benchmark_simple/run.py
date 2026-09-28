"""
End-to-end runner for the single-substrate LSTM benchmark.

Pipeline
--------
  1. load + split + scale the in-silico datasets (benchmark_simple.data)
  2. Stage 1 : train the BiLSTM+attention classifier (4-way mechanism ID)
               -> confusion matrix, accuracy, confidence calibration
  3. Stage 2 : train the parameter regressor -> per-parameter R^2 / relative
               error scatter. Two variants via --regressor:
                 per_mechanism (default) one dedicated BiLSTM+attention network
                                         per mechanism, separate weights
                 shared                  one encoder, one head per mechanism
  4. Refinement (EXP3 gate) : least-squares fit from LSTM-init vs
               naive / random / LHS-multistart baselines
               -> convergence success, nfev, wall-clock, SSE table + plots

All artefacts are written to results/simple_benchmark/.

Reuses, unchanged:
  LSTMs/lstm_classifier.py, LSTMs/train_classifier.py
  LSTMs/lstm_regressor.py,  LSTMs/train_regressor.py

Usage
-----
  python benchmark_simple/run.py                             # full run
  python benchmark_simple/run.py --epochs 40 --refine-n 60   # quicker
  python benchmark_simple/run.py --regressor shared          # old Stage-2
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from benchmark_simple import data as D
from benchmark_simple import fitting as F
from kinetic_models.simple_models import get_param_names, SIMPLE_MODELS
from LSTMs.lstm_classifier import LSTMClassifier
from LSTMs.train_classifier import train_classifier
from LSTMs.lstm_regressor import SharedEncoderRegressor
from LSTMs.train_regressor import train_shared_regressor
from benchmark_simple.models_io import PerMechRegressorBank

OUT = os.path.join(_ROOT, "results", "simple_benchmark")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 : classifier
# ─────────────────────────────────────────────────────────────────────────────

def run_classifier(splits, device, epochs, batch, seed, eval_only=False):
    Xtr, ytr = D.classifier_arrays(splits, "tr")
    Xval, yval = D.classifier_arrays(splits, "val")
    Xte, yte = D.classifier_arrays(splits, "te")

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    torch.manual_seed(seed)
    model = LSTMClassifier(input_dim=splits["input_dim"], hidden_dim=64,
                           num_layers=2, output_dim=len(D.MODEL_IDS),
                           dropout=0.3, bidirectional=True, use_attention=True)
    ckpt = os.path.join(OUT, "classifier.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.to(device)
        print("  loaded classifier checkpoint")
    if not eval_only:
        model, hist = train_classifier(
            model, loader(Xtr, ytr, True), loader(Xval, yval, False),
            n_epochs=epochs, patience=max(epochs, 10), lr=1e-3, device=device, verbose=True)
        torch.save(model.state_dict(), ckpt)

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        pred, conf, probs = model.predict_with_confidence(
            torch.tensor(Xte, dtype=torch.float32).to(device))
    pred = pred.cpu().numpy(); conf = conf.cpu().numpy()
    acc = float((pred == yte).mean())

    K = len(D.MODEL_IDS)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1

    # Confidence calibration (reliability): bin by confidence, compare to accuracy
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    calib = []
    for b in range(10):
        m = bin_idx == b
        if m.sum() > 0:
            calib.append((0.5 * (bins[b] + bins[b + 1]),
                          float((pred[m] == yte[m]).mean()), int(m.sum())))

    _plot_confusion(cm, acc)
    _plot_calibration(calib)
    return model, dict(test_acc=acc, confusion=cm.tolist(),
                       class_names=D.CLASS_NAMES,
                       mean_confidence=float(conf.mean()))


def _plot_confusion(cm, acc):
    names = D.CLASS_NAMES
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right"); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Stage 1 — mechanism ID (test acc = {acc:.3f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stage1_confusion.png"), dpi=150)
    plt.close(fig)


def _plot_calibration(calib):
    if not calib:
        return
    c = np.array(calib)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(c[:, 0], c[:, 1], "o-", color="#1f77b4", label="model")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Stage 1 — confidence calibration"); ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stage1_calibration.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 : regressor
# ─────────────────────────────────────────────────────────────────────────────

def run_regressor(splits, device, epochs, batch, seed, eval_only=False, ckpt_path=None):
    """Stage 2, shared-encoder variant: one BiLSTM+attention encoder with one
    head per mechanism, trained jointly on every mechanism's trajectories."""
    tr = D.regressor_arrays(splits, "tr")
    val = D.regressor_arrays(splits, "val")

    def loaders(dct, shuffle):
        out = {}
        for mid, (X, y) in dct.items():
            ds = TensorDataset(torch.tensor(X), torch.tensor(y))
            out[mid] = DataLoader(ds, batch_size=batch, shuffle=shuffle)
        return out

    param_counts = {mid: len(get_param_names(mid)) for mid in D.MODEL_IDS}
    torch.manual_seed(seed)
    model = SharedEncoderRegressor(input_dim=splits["input_dim"], hidden_dim=64,
                                   num_layers=2, model_param_counts=param_counts,
                                   dropout=0.3, bidirectional=True, use_attention=True)
    ckpt = ckpt_path or os.path.join(OUT, "regressor.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.to(device)
        print(f"  loaded regressor checkpoint: {ckpt}")
    if not eval_only:
        model, hist = train_shared_regressor(
            model, loaders(tr, True), loaders(val, False),
            n_epochs=epochs, patience=max(epochs, 15), lr=1e-3, device=device, verbose=True)
        torch.save(model.state_dict(), ckpt)

    return model, eval_regressor(model, splits, device)


def run_regressor_permech(splits, device, epochs, batch, seed, eval_only=False):
    """Stage 2, per-mechanism variant: one independent BiLSTM+attention network
    per mechanism (identical architecture, separate weights), each trained only
    on its own mechanism's trajectories. Returned as a PerMechRegressorBank, so
    it is interchangeable with the shared encoder downstream.

    Checkpoints are `<OUT>/regressor_permech_m{mid}.pt`; an existing one is
    loaded and training continues from it (same resume semantics as Stage 1).
    """
    tr = D.regressor_arrays(splits, "tr")
    val = D.regressor_arrays(splits, "val")
    nets = {}
    for mid in D.MODEL_IDS:
        n_params = len(get_param_names(mid))
        torch.manual_seed(seed)
        m = SharedEncoderRegressor(input_dim=splits["input_dim"], hidden_dim=64,
                                   num_layers=2, model_param_counts={mid: n_params},
                                   dropout=0.3, bidirectional=True, use_attention=True)
        ckpt = os.path.join(OUT, f"regressor_permech_m{mid}.pt")
        if os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"  loaded per-mechanism checkpoint: {ckpt}")
        m.to(device)
        if not eval_only:
            def one_loader(arr, shuffle):
                X, y = arr
                ds = TensorDataset(torch.tensor(X), torch.tensor(y))
                return {mid: DataLoader(ds, batch_size=batch, shuffle=shuffle)}
            print(f"  mechanism {mid} ({SIMPLE_MODELS[mid][0]}) — "
                  f"{tr[mid][0].shape[0]} train trajectories")
            m, _ = train_shared_regressor(
                m, one_loader(tr[mid], True), one_loader(val[mid], False),
                n_epochs=epochs, patience=max(epochs, 15), lr=1e-3,
                device=device, verbose=True)
            torch.save(m.state_dict(), ckpt)
        nets[mid] = m
    bank = PerMechRegressorBank(nets).to(device)
    return bank, eval_regressor(bank, splits, device)


def eval_regressor(model, splits, device):
    """Per-parameter recovery (R^2 + relative error) on the test split, plus the
    stage2_recovery.png scatter. Works for either Stage-2 variant."""
    te = D.regressor_arrays(splits, "te")
    model.eval()
    per_param = {}
    fig, axes = plt.subplots(1, len(D.MODEL_IDS), figsize=(4.2 * len(D.MODEL_IDS), 4))
    for ax, mid in zip(np.atleast_1d(axes), D.MODEL_IDS):
        X, y = te[mid]
        with torch.no_grad():
            pred_norm = model(torch.tensor(X).to(device), mid).cpu().numpy()
        true_phys = D.denormalise_params(y, mid)
        pred_phys = D.denormalise_params(pred_norm, mid)
        names = get_param_names(mid)
        r2s, rels = {}, {}
        for k, nm in enumerate(names):
            t, p = true_phys[:, k], pred_phys[:, k]
            ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rel = float(np.mean(np.abs(t - p) / np.maximum(np.abs(t), 1e-9)))
            r2s[nm] = float(r2); rels[nm] = rel
            # scatter normalised for a shared axis
            tn = (t - t.min()) / (t.max() - t.min() + 1e-9)
            pn = (p - t.min()) / (t.max() - t.min() + 1e-9)
            ax.scatter(tn, pn, s=6, alpha=0.35, label=f"{nm} R²={r2:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(SIMPLE_MODELS[mid][0]); ax.set_xlabel("true (norm)")
        ax.set_ylabel("pred (norm)"); ax.legend(fontsize=7)
        per_param[mid] = dict(r2=r2s, rel_err=rels)
    fig.suptitle("Stage 2 — parameter recovery (test)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stage2_recovery.png"), dpi=150)
    plt.close(fig)
    return per_param


# ─────────────────────────────────────────────────────────────────────────────
# Refinement benchmark (EXP3 gate)
# ─────────────────────────────────────────────────────────────────────────────

def run_refinement(splits, reg_model, device, refine_n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for mid in D.MODEL_IDS:
        pm = splits["per_model"][mid]
        te_idx = pm["te"]
        pick = rng.choice(te_idx, size=min(refine_n, len(te_idx)), replace=False)
        Xraw = pm["X"]; cond = pm["cond"]
        Xscaled = splits["transform"](Xraw[pick])
        with torch.no_grad():
            pred_norm = reg_model(torch.tensor(Xscaled, dtype=torch.float32).to(device),
                                  mid).cpu().numpy()
        lstm_phys = D.denormalise_params(pred_norm, mid)
        pnames = get_param_names(mid)
        lbs, ubs = F.get_param_bounds(mid)
        span = np.maximum(np.array(ubs, float) - np.array(lbs, float), 1e-12)
        for r, gi in enumerate(pick):
            Xobs = Xraw[gi, :, 0]; Sobs = Xraw[gi, :, 1]
            true_theta = pm["params"].iloc[gi][pnames].values.astype(float)
            res = F.benchmark_one(mid, cond.iloc[gi], Xobs, Sobs, lstm_phys[r],
                                  multistart_k=5, seed=int(gi))
            for strat, rr in res.items():
                # parameter-recovery accuracy: normalised distance of the fitted
                # parameters from ground truth (each parameter scaled by its box
                # width so all contribute comparably). Lower = better recovery.
                nerr = float(np.sqrt(np.mean(
                    ((np.asarray(rr["params"], float) - true_theta) / span) ** 2)))
                rows.append(dict(model_id=mid, mechanism=SIMPLE_MODELS[mid][0],
                                 sample=int(gi), strategy=strat, sse=rr["sse"],
                                 nfev=rr["nfev"], wall_s=rr["wall_s"],
                                 param_err=nerr, success=rr["success"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "refinement_raw.csv"), index=False)

    summary = (df.groupby("strategy")
                 .agg(success_rate=("success", "mean"),
                      median_nfev=("nfev", "median"),
                      median_wall_s=("wall_s", "median"),
                      median_sse=("sse", "median"),
                      median_param_err=("param_err", "median"))
                 .reindex(["lstm", "naive", "random", "lhs_multistart"]))
    summary.to_csv(os.path.join(OUT, "refinement_summary.csv"))
    _plot_refinement(df, summary)
    return summary


def _plot_refinement(df, summary):
    order = ["lstm", "naive", "random", "lhs_multistart"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].bar(order, summary.loc[order, "success_rate"], color="#1f77b4")
    axes[0].set_title("Convergence success rate\n(reaches best SSE on trajectory)")
    axes[0].set_ylim(0, 1)
    axes[1].bar(order, summary.loc[order, "median_nfev"], color="#ff7f0e")
    axes[1].set_title("Median # function evals (fitting cost)")
    data = [df[df.strategy == s]["sse"].values for s in order]
    axes[2].boxplot(data, labels=order, showfliers=False)
    axes[2].set_yscale("log"); axes[2].set_title("Final SSE (log)")
    perr = [df[df.strategy == s]["param_err"].values for s in order]
    axes[3].boxplot(perr, labels=order, showfliers=False)
    axes[3].set_title("Parameter recovery error\n(normalised |fit - true|)")
    for ax in axes:
        for lab in ax.get_xticklabels():
            lab.set_rotation(20)
    fig.suptitle("Refinement — LSTM-init vs baselines")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "refinement_comparison.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="hybrid", choices=["concentrations", "rates", "hybrid"])
    ap.add_argument("--regressor", default="per_mechanism",
                    choices=["per_mechanism", "shared"],
                    help="Stage-2 variant: per_mechanism = one dedicated network per "
                         "mechanism (default); shared = one encoder with per-mechanism heads")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--refine-n", type=int, default=40, help="test trajs/mechanism to refine")
    ap.add_argument("--max-per-model", type=int, default=None)
    ap.add_argument("--stage", default="all",
                    choices=["all", "classifier", "regressor", "refine", "assemble"],
                    help="run one stage (enables checkpoint-based resume across calls)")
    ap.add_argument("--dataset", default="generated_simple",
                    help="dataset dir under data/ (e.g. generated_simple_t12)")
    ap.add_argument("--scaling", default="global", choices=["global", "per_traj"],
                    help="per_traj = scale-invariant (nondimensional) input")
    ap.add_argument("--tag", default="",
                    help="suffix for the results dir (e.g. t12 -> results/simple_benchmark_t12)")
    ap.add_argument("--out-dir", default="",
                    help="explicit output dir for artifacts/figures (overrides --tag; "
                         "e.g. results/paper_benchmark/figures)")
    ap.add_argument("--reg-ckpt", default="",
                    help="explicit Stage-2 regressor checkpoint for the refine stage, "
                         "shared variant only (e.g. results/paper_benchmark/models/"
                         "regressor_global_hybrid.pt); defaults to <out-dir>/regressor.pt. "
                         "Ignored with --regressor per_mechanism, which always reads "
                         "<out-dir>/regressor_permech_m*.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global OUT
    if args.out_dir:
        OUT = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_ROOT, args.out_dir)
    elif args.tag:
        OUT = os.path.join(_ROOT, "results", f"simple_benchmark_{args.tag}")
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    print(f"device={device} rep={args.rep} stage={args.stage} dataset={args.dataset} out={OUT}")

    splits = D.make_splits(_ROOT, rep=args.rep, seed=args.seed,
                           max_per_model=args.max_per_model, dataset=args.dataset,
                           scaling=args.scaling)
    print(f"input_dim={splits['input_dim']}  mechanisms={splits['model_ids']}")

    def _save(key, obj):
        p = os.path.join(OUT, f"_{key}.json")
        json.dump(obj, open(p, "w"), indent=2)

    if args.stage in ("all", "classifier"):
        print("Stage 1: training classifier ...")
        _, clf_res = run_classifier(splits, device, args.epochs, args.batch, args.seed)
        print(f"  test accuracy = {clf_res['test_acc']:.3f}")
        _save("classifier", clf_res)

    if args.stage in ("all", "regressor"):
        print(f"Stage 2: training regressor ({args.regressor}) ...")
        if args.regressor == "per_mechanism":
            reg_model, reg_res = run_regressor_permech(splits, device, args.epochs,
                                                       args.batch, args.seed)
        else:
            reg_model, reg_res = run_regressor(splits, device, args.epochs,
                                               args.batch, args.seed)
        _save("regressor", reg_res)

    if args.stage in ("all", "refine"):
        print("Refinement benchmark ...")
        # regressor must exist (checkpoint) — reload if this is an isolated call
        if args.stage == "refine":
            if args.regressor == "per_mechanism":
                reg_model, _ = run_regressor_permech(splits, device, args.epochs,
                                                     args.batch, args.seed, eval_only=True)
            else:
                reg_model, _ = run_regressor(splits, device, args.epochs, args.batch,
                                             args.seed, eval_only=True,
                                             ckpt_path=(args.reg_ckpt or None))
        summary = run_refinement(splits, reg_model, device, args.refine_n, args.seed)
        print(summary.to_string())
        summary.reset_index().to_json(os.path.join(OUT, "_refinement.json"), orient="records")

    if args.stage in ("all", "assemble"):
        clf_res = json.load(open(os.path.join(OUT, "_classifier.json")))
        reg_res = json.load(open(os.path.join(OUT, "_regressor.json")))
        refine = json.load(open(os.path.join(OUT, "_refinement.json")))
        with open(os.path.join(OUT, "results.json"), "w") as f:
            json.dump({"classifier": clf_res, "regressor_per_param": reg_res,
                       "refinement": refine, "config": vars(args)}, f, indent=2)
        print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
