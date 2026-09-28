"""
In-silico training figures for Case Study 2 (reads the JSON that run.py /
train_per_mechanism.py write to results/xylitol_use_case/<tag>/):

  confusion.png              11x11 Stage-1 confusion matrix
  r2_heatmap.png             Stage-2 per-parameter R^2 across the 11 mechanisms
  regressor_comparison.png   shared encoder vs dedicated per-mechanism (mean R^2/model)

Run:  python xylitol_case_study/figures.py --tag main
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# canonical union of parameter names across models 21-31 (for the heatmap columns)
PARAM_ORDER = ["mumax1", "mumax2", "YXS1", "YPS2", "KS1", "KP",
               "kd", "KSI1", "KPI1", "KSI2", "KPI2"]
PLAB = {"mumax1": r"$\mu_{m1}$", "mumax2": r"$\mu_{m2}$", "YXS1": "$Y_{XS1}$",
        "YPS2": "$Y_{PS2}$", "KS1": "$K_{S1}$", "KP": "$K_P$", "kd": "$k_d$",
        "KSI1": "$K_{SI1}$", "KPI1": "$K_{PI1}$", "KSI2": "$K_{SI2}$", "KPI2": "$K_{PI2}$"}


def plot_confusion(clf_json, out):
    cm = np.array(clf_json["confusion"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = [n.replace("model", "M") for n in clf_json["label_names"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Stage-1 confusion (11 mechanisms, test acc {clf_json['test_acc']:.2f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _r2_matrix(reg_json):
    mids = sorted(int(k) for k in reg_json)
    M = np.full((len(mids), len(PARAM_ORDER)), np.nan)
    for r, mid in enumerate(mids):
        for p, v in reg_json[str(mid)]["r2"].items():
            if p in PARAM_ORDER:
                M[r, PARAM_ORDER.index(p)] = v
    return mids, M


def plot_r2_heatmap(reg_json, out, title="Stage-2 parameter recovery ($R^2$)"):
    mids, M = _r2_matrix(reg_json)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(np.clip(M, 0, 1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(PARAM_ORDER))); ax.set_yticks(range(len(mids)))
    ax.set_xticklabels([PLAB[p] for p in PARAM_ORDER], fontsize=10)
    ax.set_yticklabels([f"M{m}" for m in mids], fontsize=9)
    for i in range(len(mids)):
        for j in range(len(PARAM_ORDER)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="#bbbbbb")
    ax.set_title(title); fig.colorbar(im, fraction=0.03, pad=0.02)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_regressor_comparison(shared_json, permech_json, out):
    mids = sorted(set(int(k) for k in shared_json) | set(int(k) for k in permech_json))
    sh = [shared_json.get(str(m), {}).get("r2_mean", np.nan) for m in mids]
    pm = [permech_json.get(str(m), {}).get("r2_mean", np.nan) for m in mids]
    x = np.arange(len(mids)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - w / 2, sh, w, color="#7f7f7f", label="shared encoder")
    ax.bar(x + w / 2, pm, w, color="#1f77b4", label="per-mechanism")
    ax.set_xticks(x); ax.set_xticklabels([f"M{m}" for m in mids])
    ax.set_ylabel("mean $R^2$ across parameters"); ax.set_ylim(0, 1.05)
    ax.set_title("Stage-2 regressor: shared encoder vs dedicated per-mechanism LSTMs")
    ax.grid(alpha=0.2, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    D = os.path.join(_ROOT, "results", "xylitol_use_case", args.tag)

    clf_p = os.path.join(D, "_classifier.json")
    if os.path.exists(clf_p):
        plot_confusion(json.load(open(clf_p)), os.path.join(D, "confusion.png"))
        print("saved confusion.png")

    pm_p = os.path.join(D, "_regressor_permechanism.json")
    if os.path.exists(pm_p):
        plot_r2_heatmap(json.load(open(pm_p)), os.path.join(D, "r2_heatmap.png"),
                        "Stage-2 parameter recovery ($R^2$) — per-mechanism regressors")
        print("saved r2_heatmap.png")


if __name__ == "__main__":
    main()
