"""Case study 2 report (virtual and real xylitol runs): statistics, figures and tables."""
import os
os.environ.setdefault("GFORTRAN_UNBUFFERED_ALL", "1")
import sys
import pickle

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from report_common import (load_all, savefig, Macros, write_tex, fmt, pct, sci, W1, W2, BLUE, ORANGE, AQUA, YELLOW,
                           INK, GREY, LIGHT, HERE, RES)

sys.path.insert(0, HERE)
from kinid import case2 as c2  # noqa: E402

WORKBOOK = os.path.join(HERE, "..", "..", "data", "FedBatch_xylitol_in_YP.xlsx")
CHL = [r"$X$", r"$S_1$", r"$S_2$", r"$P$"]
CHN = ["biomass", "glucose eq.", "xylose", "xylitol"]
PNAMES = {"mumax1": r"$\mu_{\max,1}$ [h$^{-1}$]", "mumax2": r"$\mu_{\max,2}$ [h$^{-1}$]", "YXS1": r"$Y_{XS_1}$ [g g$^{-1}$]",
          "YPS2": r"$Y_{PS_2}$ [g g$^{-1}$]", "KS1": r"$K_{S_1}$ [g L$^{-1}$]", "KP": r"$K_{S_2}$ [g L$^{-1}$]",
          "kd": r"$k_d$ [h$^{-1}$]", "KSI1": r"$K_{SI_1}$ [g L$^{-1}$]", "KPI1": r"$K_{PI_1}$ [g L$^{-1}$]",
          "KSI2": r"$K_{SI_2}$ [g L$^{-1}$]", "KPI2": r"$K_{PI_2}$ [g L$^{-1}$]"}


def num(x, sign=False):
    """Number with a proper minus sign, usable in and out of math mode."""
    return f"\\ensuremath{{{x:+.1f}}}" if sign else f"\\ensuremath{{{x:.1f}}}"


def entry_of(name, rec):
    """Rebuild a simulable entry (library or SR) from a stored record."""
    if rec["library"]:
        return c2.LIBRARY[name], rec["theta"]
    l1, l2 = rec["laws"]
    death = name.endswith("+kd")
    th = rec["theta"]
    kd = th[-3] if death else 0.0
    e = c2.sr_entry(l1, l2, death, list(th[-2:]), kd)
    return e, e["centre"]


def best(recs, library):
    names = [m for m, r in recs.items() if r["library"] == library]
    return min(names, key=lambda m: recs[m]["BIC"]) if names else None


def latex_law(expr_str, Y=None, tol=0.01):
    """LaTeX of a discovered law. With measured states Y (rows: S1, S2, P), terms that never contribute more
    than `tol` of the numerator or denominator over the measured states are omitted."""
    S1, S2, P = sp.symbols("S1 S2 P", positive=True)
    e = sp.sympify(expr_str, locals=dict(S1=S1, S2=S2, P=P))
    if Y is not None:
        num, den = sp.fraction(e, exact=True)
        vals = {S1: Y[:, 0], S2: Y[:, 1], P: Y[:, 2]}

        def prune(expr):
            terms = sp.Add.make_args(sp.expand(expr))
            f = [np.abs(np.broadcast_to(sp.lambdify(list(vals), t, "numpy")(*vals.values()), Y[:, 0].shape))
                 for t in terms]
            tot = np.sum(f, 0)
            keep = [t for t, v in zip(terms, f) if np.nanmax(v / np.where(tot > 0, tot, 1)) >= tol]
            return sp.Add(*keep)
        e = prune(num) / prune(den)
    return sp.latex(e, symbol_names={S1: "S_1", S2: "S_2", P: "P"}).replace("+ 1.0}", "+ 1}").replace("\\frac", "\\dfrac")


# ── virtual runs ─────────────────────────────────────────────────────────────
def virtual(M):
    V = load_all("virtual/*.pkl")
    rows, out = [], {}
    for vname, mech in [("V18", "M31"), ("V15", "M28")]:
        rs = [r for k, r in V.items() if k.startswith(vname)]
        if not rs:
            continue
        sel, w, short, adeq, dB, fd, ws = [], [], [], [], [], [], []
        for r in rs:
            lib = r["library"]
            b = min(lib, key=lambda m: lib[m]["BIC"])
            sel.append(b)
            w.append(lib[mech]["weight"])
            short.append(lib[mech]["weight"] > 0.1)
            n = r["n"]
            adeq.append(lib[b]["chi2"] <= n + 3 * np.sqrt(2 * n))
            a = r["all"]
            bl, bs = best(a, True), best(a, False)
            d = a[bs]["BIC"] - a[bl]["BIC"]
            dB.append(d)
            fd.append(d < -10 and a[bs]["chi2"] <= n + 3 * np.sqrt(2 * n))
        acc = np.mean([s == mech for s in sel])
        others = sorted({s for s in sel if s != mech})
        out[vname] = dict(acc=acc, w=np.mean(w), short=np.mean(short), n=len(rs), others=others)
        tag = "Veighteen" if vname == "V18" else "Vfifteen"
        M[f"{tag}Acc"], M[f"{tag}W"], M[f"{tag}Short"] = pct(acc), fmt(np.mean(w)), pct(np.mean(short))
        M[f"{tag}N"], M[f"{tag}Adeq"] = len(rs), pct(np.mean(adeq))
        M[f"{tag}FD"] = int(np.sum(fd))
        M[f"{tag}dBICmin"], M[f"{tag}dBICmed"] = num(np.min(dB)), num(np.median(dB))
        M[f"{tag}NodeLoss"] = sci(np.median([r['node']['loss'] for r in rs]))
        meanw = {m: np.mean([r["library"][m]["weight"] for r in rs]) for m in rs[0]["library"]}
        alts = sorted([m for m in meanw if m != mech and meanw[m] >= 0.05], key=lambda m: -meanw[m])
        M[f"{tag}Alts"] = ", ".join(f"{m} ({meanw[m]:.2f})" for m in alts) or "none"
        wrong = {s: sel.count(s) for s in others}
        rows.append(f"{vname} & {mech} & {pct(acc)} & {fmt(np.mean(w))} & {pct(np.mean(short))} & "
                    f"{', '.join(f'{k} ({v})' for k, v in wrong.items()) or '--'} & {num(np.median(dB))} & {int(np.sum(fd))} \\\\")
    write_tex("tab_c2_virtual.tex", "\n".join(rows) + "\n")
    # form of the best proposed conversion law over all virtual analyses
    mu2 = [best(r["all"], False).split("mu2[")[1].split("]")[0] for r in V.values()]
    M["VirtSRn"] = len(mu2)
    M["VirtSRexp"] = sum("exp(-P/K)" in f for f in mu2)
    M["VirtSRexpMonod"] = sum(f == "S2*exp(-P/K) + mu2" for f in mu2)
    # death term in the best proposed law where the generating mechanism has one (M31) and not otherwise (M28)
    M["VirtSRdeath"] = sum(best(r["all"], False).endswith("+kd") == (k.startswith("V18")) for k, r in V.items())
    # true parameters of the virtual runs
    truth = {v: dict(zip(c2.LIBRARY[m]["names"], V[f"{v}_0"]["theta_true"]))
             for v, m in [("V18", "M31"), ("V15", "M28")] if f"{v}_0" in V}
    lines = [f"{PNAMES[p]} & " + " & ".join(f"{truth[v][p]:.3g}" if p in truth[v] else "--" for v in truth) + " \\\\"
             for p in ["mumax1", "KS1", "YXS1", "mumax2", "KP", "YPS2", "KSI2", "KPI2", "kd"]]
    write_tex("tab_c2_virtual_params.tex", "\n".join(lines) + "\n")
    return V


def fig_virtual(V):
    runs = c2.load_real(WORKBOOK)
    fig, axes = plt.subplots(2, 6, figsize=(W2, 2.9))
    for row, (vname, base, mech) in enumerate([("V18", "BC18", "M31"), ("V15", "BC15", "M28")]):
        r = V.get(f"{vname}_0")
        if r is None:
            continue
        run = dict(runs[base])
        td = np.linspace(0, run["t_end"], 141)
        e = c2.LIBRARY[mech]
        x0_true = c2.initial_state(run["t"], run["Y"][:, None, :]).ravel()
        truth = c2.simulate(e, r["theta_true"], td, x0_true, [run])
        lib = r["library"]
        b = min(lib, key=lambda m: lib[m]["BIC"])
        fit = c2.simulate(c2.LIBRARY[b], lib[b]["theta"], td, lib[b]["x0"], [run])
        for j in range(4):
            ax = axes[row, j]
            ax.plot(r["t"], r["Y"][:, j], "o", ms=2.2, color=GREY, mew=0, label="data")
            ax.plot(td, truth[:, j], color=INK, lw=1.6, alpha=0.3, label=f"truth ({mech})")
            ax.plot(td, fit[:, j], color=BLUE, lw=1.0, label=f"selected ({b})")
            ax.set_title(f"{vname}: {CHN[j]}")
        # learned vs true rates
        nd = r["node"]
        keep = nd["keep"][0]
        tt = nd["t_dense"][keep]
        mu1t, mu2t = c2.library_rates(dict(zip(e["names"], r["theta_true"])),
                                      *[np.clip(c2.simulate(e, r["theta_true"], tt, x0_true, [run])[:, k], 0, None)
                                        for k in (1, 2, 3)])
        for k, (lt, ln) in enumerate([(mu1t, nd["mu1_dense"][keep, 0]), (mu2t, nd["mu2_dense"][keep, 0])]):
            ax = axes[row, 4 + k]
            ax.plot(tt, lt, color=INK, lw=1.6, alpha=0.3, label="true")
            ax.plot(tt, ln, color=ORANGE, lw=1.0, label="learned (hybrid)")
            ax.set_title(f"{vname}: " + (r"$\mu_1$ [h$^{-1}$]" if k == 0 else r"$\mu_2$ [h$^{-1}$]"))
    for ax in axes[-1]:
        ax.set_xlabel("time [h]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"g L$^{-1}$")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 4].get_legend_handles_labels()
    fig.legend(h1 + h2[1:], ["data", "truth", "selected mechanism", "learned rate (hybrid model)"], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(h_pad=0.5, w_pad=0.4, rect=(0, 0.05, 1, 1))
    savefig(fig, "c2_virtual")


# ── real runs ────────────────────────────────────────────────────────────────
def real(M):
    R = load_all("real/*.pkl")
    runs = c2.load_real(WORKBOOK)
    rows, prow, srrows = [], {}, []
    order = [k for k in ["BC12", "BC15", "BC18", "BC19", "joint3", "joint4"] if k in R]
    label = {"joint3": "BC12+BC15+BC18", "joint4": "all four"}
    for k in order:
        r = R[k]
        lib = r["library"]
        srt = sorted(lib, key=lambda m: lib[m]["BIC"])
        b = srt[0]
        short = [(m, lib[m]["weight"]) for m in srt if lib[m]["weight"] > 0.1]
        a = r["all"]
        bs = best(a, False)
        d = a[bs]["BIC"] - a[b]["BIC"]
        # per-species SD of the best library fit
        rr = [runs[x] for x in r["runs"]]
        t, Yf, Y3 = c2.data_matrix(rr)
        e = c2.LIBRARY[b]
        Ys = c2.simulate(e, lib[b]["theta"], t, lib[b]["x0"], rr)
        res = (Ys - Yf).reshape(len(t), len(rr), 4)
        sd = [np.sqrt(np.nanmean(res[..., j] ** 2)) for j in range(4)]
        rows.append(f"{label.get(k, k)} & {r['n']} & {b} ({lib[b]['weight']:.2f}) & "
                    f"{', '.join(f'{m} ({w:.2f})' for m, w in short)} & "
                    f"{' & '.join(f'{s:.2f}' for s in sd)} & {num(d, True)} \\\\")
        prow[k] = (b, dict(zip(e["names"], lib[b]["theta"])))
        f1, f2 = a[bs]["formula"]
        Ym = np.vstack([np.c_[x["Y"][:, 1], x["Y"][:, 2], x["Y"][:, 3]] for x in rr])
        Ym = Ym[np.all(np.isfinite(Ym), 1)]
        srrows.append((label.get(k, k), d, latex_law(f1, Ym), latex_law(f2, Ym), "kd" in bs.split()[-1]))
        tag = {"BC12": "BCtwelve", "BC15": "BCfifteen", "BC18": "BCeighteen", "BC19": "BCnineteen",
               "joint3": "JointThree", "joint4": "JointFour"}[k]
        M[f"{tag}Sel"], M[f"{tag}W"] = b, fmt(lib[b]["weight"])
        M[f"{tag}SRdBIC"] = num(d, True)
        M[f"{tag}N"] = r["n"]
        M[f"{tag}Minutes"] = f"{r['seconds'] / 60:.0f}"
        # conversion rate of the selected mechanism at a common reference state, S2 = 30 and P = 50 g/L
        M[f"{tag}MuRef"] = f"{c2.library_rates(prow[k][1], 0.0, 30.0, 50.0)[1]:.2f}"
        M[f"{tag}Kd"] = f"{prow[k][1].get('kd', 0.0):.3f}"
        # fitted initial biomass minus the first measurement, per run
        M[f"{tag}DXzero"] = ", ".join(f"{lib[b]['x0'][4 * i] - x['Y'][0, 0]:.1f}" for i, x in enumerate(rr))
        # residual SD per run and species of the selected mechanism
        R[k]["sd_run"] = np.sqrt(np.nanmean(res ** 2, axis=0))
    write_tex("tab_c2_real.tex", "\n".join(rows) + "\n")
    # parameters of the M31 fit (the mechanism selected for most analyses) and of each selected mechanism
    allnames = ["mumax1", "mumax2", "YXS1", "YPS2", "KS1", "KP", "kd", "KSI2", "KPI2"]
    lines = []
    for p in allnames:
        vals = []
        for k in order:
            r = R[k]
            if "M31" in r["library"]:
                vals.append(f"{dict(zip(c2.LIBRARY['M31']['names'], r['library']['M31']['theta']))[p]:.3g}")
        lines.append(f"{PNAMES[p]} & " + " & ".join(vals) + " \\\\")
    write_tex("tab_c2_real_params.tex", "\n".join(lines) + "\n")
    write_tex("tab_c2_real_params_head.tex", " & ".join(label.get(k, k) for k in order) + "\n")
    # SR laws
    lines = [f"{lab} & ${l1}$ & ${l2}$ & {'yes' if kd else 'no'} & {num(d, True)} \\\\" for lab, d, l1, l2, kd in srrows]
    write_tex("tab_c2_real_sr.tex", "\n".join(lines) + "\n")
    return R, runs


def rss_by_species(r, runs, mech="M31"):
    """Residual sum of squares and counts per species (summed over the runs of the analysis) of a library fit."""
    rr = [runs[x] for x in r["runs"]]
    t, Yf, _ = c2.data_matrix(rr)
    rec = r["library"][mech]
    Y = c2.simulate(c2.LIBRARY[mech], rec["theta"], t, rec["x0"], rr)
    res = ((Y - Yf) ** 2).reshape(len(t), len(rr), 4)
    cnt = np.isfinite(Yf).reshape(len(t), len(rr), 4)
    return np.nansum(res, (0, 1)), cnt.sum((0, 1))


def shared_vs_separate(R, runs, joint, parts, mech="M31"):
    """BIC of shared kinetics (the joint fit) minus BIC of separate kinetics (the fits of `parts`), both with
    one error SD per species pooled over all runs. Negative: shared kinetics are preferred."""
    rj, nj = rss_by_species(R[joint], runs, mech)
    rs = sum(rss_by_species(R[p], runs, mech)[0] for p in parts)
    n = nj.sum()
    k = R[joint]["library"][mech]["k"]
    crit = lambda rss: float(np.sum(nj * np.log(rss / nj)))
    return (crit(rj) + k * np.log(n)) - (crit(rs) + k * len(parts) * np.log(n))


def fig_real(R, runs):
    names = ["BC12", "BC15", "BC18", "BC19"]
    fig, axes = plt.subplots(4, 4, figsize=(W2, 5.6))
    td_all = {}
    for i, k in enumerate(names):
        run = runs[k]
        td = np.linspace(0, run["t_end"], 200)
        r = R.get(k)
        for j in range(4):
            ax = axes[i, j]
            ax.plot(run["t"], run["Y"][:, j], "o", ms=2.6, color=INK, mew=0, label="measured", zorder=4)
        if r is not None:
            lib = r["library"]
            b = min(lib, key=lambda m: lib[m]["BIC"])
            y = c2.simulate(c2.LIBRARY[b], lib[b]["theta"], td, lib[b]["x0"], [run])
            nd = r["node"]
            keep = nd["keep"][0]
            a = r["all"]
            bs = best(a, False)
            es, ths = entry_of(bs, a[bs])
            ysr = c2.simulate(es, ths, td, a[bs]["x0"], [run])
            for j in range(4):
                axes[i, j].plot(nd["t_dense"][keep], nd["y_dense"][keep, 0, j], color=LIGHT, lw=1.6,
                                label="hybrid neural ODE")
                axes[i, j].plot(td, y[:, j], color=BLUE, lw=1.0, label="best library mechanism, run alone")
                if ysr is not None:
                    axes[i, j].plot(td, ysr[:, j], color=AQUA, lw=1.0, label="best symbolic-regression law, run alone")
            axes[i, 0].set_ylabel(f"{k}\n" + r"g L$^{-1}$")
        J = R.get("joint3")
        if J is not None and k in J["runs"]:
            lib = J["library"]
            b = min(lib, key=lambda m: lib[m]["BIC"])
            rr = [runs[x] for x in J["runs"]]
            ti = J["runs"].index(k)
            tgrid = np.linspace(0, max(x["t_end"] for x in rr), 300)
            y = c2.simulate(c2.LIBRARY[b], lib[b]["theta"], tgrid, lib[b]["x0"], rr)
            for j in range(4):
                axes[i, j].plot(tgrid, y[:, 4 * ti + j], color=ORANGE, lw=1.0, ls="--",
                                label="best library mechanism, joint (BC12, BC15, BC18)")
    for j in range(4):
        axes[0, j].set_title(f"{CHN[j]} {CHL[j]}")
        axes[-1, j].set_xlabel("time [h]")
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=0.5, w_pad=0.5, rect=(0, 0.06, 1, 1))
    savefig(fig, "c2_real")


def fig_real_weights(R):
    order = [k for k in ["BC12", "BC15", "BC18", "BC19", "joint3", "joint4"] if k in R]
    labels = {"joint3": "joint: BC12, 15, 18", "joint4": "joint: all four"}
    mechs = list(c2.LIBRARY)
    W = np.array([[R[k]["library"][m]["weight"] for m in mechs] for k in order])
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    im = ax.imshow(W, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mechs)))
    ax.set_xticklabels(mechs, rotation=60, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels.get(k, k) for k in order])
    for i, j in zip(*np.nonzero(W >= 0.05)):
        ax.text(j, i, f"{W[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                color="white" if W[i, j] > 0.6 else INK)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("BIC weight")
    cb.outline.set_visible(False)
    savefig(fig, "c2_real_weights")


def main():
    M = Macros("numbers_c2.tex")
    V = virtual(M)
    if V:
        fig_virtual(V)
    R, runs = real(M)
    if R:
        # shared versus separate kinetics (M31) for groups of runs
        lines = []
        for joint, tag in [("joint2", "SharedTwo"), ("joint3b", "SharedThreeB"), ("joint3", "SharedThree"),
                           ("joint4", "SharedFour")]:
            if joint in R and all(x in R for x in R[joint]["runs"]):
                parts = R[joint]["runs"]
                d = shared_vs_separate(R, runs, joint, parts)
                k = R[joint]["library"]["M31"]["k"]
                M[tag] = num(d, True)
                lines.append(f"{', '.join(parts)} & {R[joint]['n']} & {k} & {k * len(parts)} & {num(d, True)} \\\\")
        write_tex("tab_c2_shared.tex", "\n".join(lines) + "\n")
        fig_real(R, runs)
        fig_real_weights(R)
    M.write()
    print("case 2 report written;", len(V), "virtual,", len(R), "real")


if __name__ == "__main__":
    main()
