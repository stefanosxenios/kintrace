"""
Training-results figure: Stage-1 confusion matrix (a) beside Stage-2 per-parameter
R^2 heatmap (b). Reads the evaluation JSON that run.py writes.

Run:
  python benchmark_simple/training_matrices.py \
      --clf-dir results/simple_benchmark_rec --reg-dir results/simple_benchmark_rec
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf-dir", default=os.path.join(_ROOT, "results", "simple_benchmark_rec"))
    ap.add_argument("--reg-dir", default=os.path.join(_ROOT, "results", "simple_benchmark_rec"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "simple_benchmark", "training_matrices.png"))
    args = ap.parse_args()

    clf = json.load(open(os.path.join(args.clf_dir, "_classifier.json")))
    reg = json.load(open(os.path.join(args.reg_dir, "_regressor.json")))
    names = clf["class_names"]; short = [n.replace("_inhibition", "_inhib") for n in names]
    cm = np.array(clf["confusion"], float); cmn = cm / cm.sum(1, keepdims=True)
    params = ["mumax", "Ks", "Kc", "KI", "Xmax", "Yxs"]
    plabel = [r"$\mu_{max}$", "$K_S$", "$K_c$", "$K_I$", "$X_{max}$", "$Y_{XS}$"]
    R2 = np.full((4, len(params)), np.nan)
    for mid in ["1", "2", "3", "4"]:
        for p, v in reg[mid]["r2"].items():
            R2[int(mid) - 1, params.index(p)] = v

    fig = plt.figure(figsize=(13, 4.4))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.35], wspace=0.32)
    axc = fig.add_subplot(gs[0])
    im = axc.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    axc.set_xticks(range(4)); axc.set_yticks(range(4))
    axc.set_xticklabels(short, rotation=35, ha="right", fontsize=9); axc.set_yticklabels(short, fontsize=9)
    for i in range(4):
        for j in range(4):
            axc.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if cmn[i, j] > 0.5 else "black")
    axc.set_xlabel("Predicted"); axc.set_ylabel("True")
    axc.set_title(f"(a) Stage-1 confusion (test acc {clf['test_acc']:.2f})", fontsize=11)
    fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04)
    axr = fig.add_subplot(gs[1])
    im2 = axr.imshow(np.clip(R2, 0, 1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axr.set_xticks(range(len(params))); axr.set_yticks(range(4))
    axr.set_xticklabels(plabel, fontsize=11); axr.set_yticklabels(short, fontsize=9)
    for i in range(4):
        for j in range(len(params)):
            axr.text(j, i, (f"{R2[i, j]:.2f}" if not np.isnan(R2[i, j]) else "–"),
                     ha="center", va="center", fontsize=9,
                     color=("#888888" if np.isnan(R2[i, j]) else "black"))
    axr.set_title("(b) Stage-2 parameter recovery ($R^2$)", fontsize=11)
    fig.colorbar(im2, ax=axr, fraction=0.046, pad=0.04)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
