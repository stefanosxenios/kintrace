"""Case study 1 report: statistics over noise realisations, figures and tables."""
import os
os.environ.setdefault("GFORTRAN_UNBUFFERED_ALL", "1")
import sys

import numpy as np
import matplotlib.pyplot as plt

from report_common import (load_all, savefig, Macros, write_tex, fmt, pct, sci, bic_weights, W1, W2,
                           BLUE, ORANGE, AQUA, YELLOW, INK, GREY, LIGHT, RUNC, HERE)

sys.path.insert(0, HERE)
from kinid import case1 as c1  # noqa: E402

LIB = list(c1.LIBRARY)
EXPS = list(c1.TRUE) + ["hidden"]
NICE = {"monod": "Monod", "contois": "Contois", "haldane": "Haldane", "biomass_inh": "Logistic",
        "tessier": "Tessier", "moser": "Moser", "aiba": "Aiba", "hidden": "Hidden"}


def nice(m):
    base, kd = (m[:-3], True) if m.endswith("+kd") else (m, False)
    return NICE[base] + (r"+$k_d$" if kd else "")


TRUE_TERMS = {"monod": {"S", "mu"}, "contois": {"S", "mu*X"}, "haldane": {"S", "mu", "mu*S^2"},
              "biomass_inh": {"S", "S*X", "mu"}, "tessier": {"S", "S*exp(-S/K)"}, "aiba": {"S*exp(-S/K)", "mu"},
              "moser": None, "hidden": {"S", "S*X", "mu", "mu*S^2"}}


def true_terms(e):
    return TRUE_TERMS[e[:-3] if e.endswith("+kd") else e]


def best(recs, library):
    names = [m for m, r in recs.items() if r["library"] == library]
    return min(names, key=lambda m: recs[m]["BIC"]) if names else None


def collect():
    R = load_all("case1/*.pkl")
    seeds = sorted({int(k.rsplit("_", 1)[1]) for k in R})
    by = {(k.rsplit("_", 1)[0], int(k.rsplit("_", 1)[1])): v for k, v in R.items()}
    return by, seeds


def selection_stats(by, seeds, mode, cands, exps):
    """Accuracy, weight of the true mechanism and shortlist inclusion; mean weight matrix."""
    acc, wtrue, short, W = {}, {}, {}, np.zeros((len(exps), len(cands)))
    for i, e in enumerate(exps):
        a, w, s, cnt = [], [], [], 0
        for sd in seeds:
            if (e, sd) not in by:
                continue
            res = by[(e, sd)][mode]
            ws = bic_weights(res["library"], cands, res["n"])
            W[i] += [ws[c] for c in cands]
            cnt += 1
            if e in cands:
                a.append(max(ws, key=ws.get) == e)
                w.append(ws[e])
                s.append(ws[e] > 0.1)
        W[i] /= max(cnt, 1)
        if e in cands and a:
            acc[e], wtrue[e], short[e] = np.mean(a), np.mean(w), np.mean(s)
    return acc, wtrue, short, W


def sr_stats(by, seeds, mode):
    """Per experiment and seed: dBIC (best SR - best library), rediscovery of the true term set, acceptance."""
    rows = {}
    for e in EXPS:
        for sd in seeds:
            if (e, sd) not in by:
                continue
            res = by[(e, sd)][mode]
            recs = res["all"]
            bl, bs = best(recs, True), best(recs, False)
            d = recs[bs]["BIC"] - recs[bl]["BIC"]
            tt = true_terms(e)
            death = e.endswith("+kd")
            redisc = tt is not None and set(recs[bs]["law"]["terms"]) == tt and (bs.endswith("+kd") == death)
            # rediscovery among SR models within 2 BIC units of the best SR model
            near = [m for m, r in recs.items() if not r["library"] and r["BIC"] <= recs[bs]["BIC"] + 2]
            redisc_near = tt is not None and any(set(recs[m]["law"]["terms"]) == tt for m in near)
            accept = d < c1.ACCEPT_DBIC and recs[bs]["chi2"] <= c1.chi2_ok(res["n"])
            rows[(e, sd)] = dict(dbic=d, redisc=redisc, redisc_near=redisc_near, accept=accept,
                                 adequate=res["library_adequate"], esc=res.get("escalated", False),
                                 best_sr=bs, formula=recs[bs]["formula"], chi2_lib=recs[bl]["chi2"],
                                 chi2_sr=recs[bs]["chi2"], n=res["n"], overall=min(recs, key=lambda m: recs[m]["BIC"]))
    return rows


# ── figures ──────────────────────────────────────────────────────────────────
def fig_design(by):
    e = by[("hidden", 0)]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 1.9))
    for run in e["runs"]:
        c = RUNC[run["key"]]
        for j, ax in enumerate(axes[:2]):
            ax.plot(c1.T_OBS, run["Y"][:, j], "o", ms=1.8, color=c, alpha=0.6, mew=0)
            ax.plot(c1.T_OBS, run["truth"][:, j], "-", color=c, lw=1.0, label=f"run {run['key']}")
    axes[0].set(xlabel="time [h]", ylabel=r"$X$ [g L$^{-1}$]")
    axes[1].set(xlabel="time [h]", ylabel=r"$S$ [g L$^{-1}$]")
    axes[0].legend(loc="upper left")
    for name, mk in [("contois", "o"), ("haldane", "s"), ("hidden", "^")]:
        for run in by[(name, 0)]["runs"]:
            axes[2].plot(run["truth"][:, 0], run["truth"][:, 1], "-", color=RUNC[run["key"]], lw=0.9)
            axes[2].plot(run["truth"][::8, 0], run["truth"][::8, 1], mk, ms=2.5, color=RUNC[run["key"]], mew=0)
    axes[2].set(xlabel=r"$X$ [g L$^{-1}$]", ylabel=r"$S$ [g L$^{-1}$]")
    h = [plt.Line2D([], [], ls="", marker=mk, color=GREY, ms=3, label=NICE[n]) for n, mk in
         [("contois", "o"), ("haldane", "s"), ("hidden", "^")]]
    axes[2].legend(handles=h, loc="upper right")
    for ax, lab in zip(axes, "abc"):
        ax.set_title(f"({lab})", loc="left")
    fig.tight_layout(w_pad=1.2)
    savefig(fig, "c1_design")


def fig_learned_mu(by):
    fig, axes = plt.subplots(3, 4, figsize=(W2, 3.9), sharex=True)
    for ax, e in zip(axes.ravel(), EXPS):
        res = by[(e, 0)]
        mu_fn, p = c1.truth_of(e)
        d = c1.DESIGN["a"]
        tr = c1.simulate_runs(mu_fn, p, c1.T_DENSE, [d["X0"], d["S0"]], [d])
        ax.plot(c1.T_DENSE, mu_fn(tr[:, 0], np.clip(tr[:, 1], 0, None), p), color=INK, lw=1.6, alpha=0.35,
                label="true")
        for mode, col, lab in [("single", ORANGE, "run a"), ("joint", BLUE, "runs a, b, c")]:
            node = res[mode]["node"]
            ax.plot(c1.T_DENSE, node["mur"][0], color=col, lw=1.0, label=f"learned from {lab}")
        ax.set_title(nice(e))
    for ax in axes[-1]:
        ax.set_xlabel("time [h]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mu$ [h$^{-1}$]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0.04, 1, 1))
    savefig(fig, "c1_learned_mu")


def fig_confusion(Ws, Wj):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.3), sharey=True)
    for ax, W, title in [(axes[0], Ws, "(a) one run (a)"), (axes[1], Wj, "(b) three runs (a, b, c), joint")]:
        im = ax.imshow(W, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(LIB)))
        ax.set_xticklabels([nice(m) for m in LIB], rotation=60, ha="right")
        ax.set_yticks(range(len(EXPS)))
        ax.set_yticklabels([nice(e) for e in EXPS])
        ax.set_title(title, loc="left")
        ax.set_xlabel("candidate mechanism")
        for i, j in zip(*np.nonzero(W >= 0.05)):
            ax.text(j, i, f"{W[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                    color="white" if W[i, j] > 0.6 else INK)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("generating mechanism")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("mean BIC weight")
    cb.outline.set_visible(False)
    savefig(fig, "c1_confusion")


def fig_sr(srs, srj):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    for ax, rows, title in [(axes[0], srs, "(a) one run (a)"), (axes[1], srj, "(b) three runs, joint")]:
        for i, e in enumerate(EXPS):
            d = np.array([r["dbic"] for (ee, _), r in rows.items() if ee == e])
            red = np.array([r["redisc_near"] for (ee, _), r in rows.items() if ee == e])
            y = i + np.linspace(-0.25, 0.25, len(d))
            ax.scatter(np.sign(d) * np.log10(1 + np.abs(d)), y, s=7, c=np.where(red, AQUA, GREY), lw=0, zorder=3)
        ax.axvline(-np.log10(11), color=ORANGE, lw=0.8, ls="-")
        ax.axvline(0, color=LIGHT, lw=0.6)
        ticks = [-1000, -100, -10, 0, 10, 100]
        ax.set_xticks([np.sign(t) * np.log10(1 + abs(t)) for t in ticks])
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticks(range(len(EXPS)))
        ax.set_yticklabels([nice(e) for e in EXPS])
        ax.invert_yaxis()
        ax.set_xlabel(r"$\Delta$BIC = best SR law $-$ best library mechanism")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="#e1e0d9", lw=0.5)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, color=AQUA, label="true term set recovered"),
         plt.Line2D([], [], ls="", marker="o", ms=3, color=GREY, label="other law"),
         plt.Line2D([], [], color=ORANGE, lw=0.8, label=r"acceptance, $\Delta$BIC $=-10$")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    savefig(fig, "c1_sr")


def fig_hidden(by, loop):
    """Hidden law: best library fit vs discovered law on the three runs; and the later run."""
    res = by[("hidden", 0)]
    recs = res["joint"]["all"]
    bl, bs = best(recs, True), best(recs, False)
    runs = c1.make_runs("hidden", 0)
    lib_entry = c1.LIBRARY[bl]
    sr_e = c1.sr_entry(recs[bs]["law"], bs.endswith("+kd"), 0.5, 0.0)
    y_lib = c1.simulate(lib_entry, recs[bl]["theta"], c1.T_DENSE, recs[bl]["x0"], runs)
    y_sr = c1.simulate(sr_e, recs[bs]["theta"], c1.T_DENSE, recs[bs]["x0"], runs)
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.2))
    for i, run in enumerate(runs):
        for j in range(2):
            ax = axes[j, i]
            ax.plot(c1.T_OBS, run["Y"][:, j], "o", ms=1.8, color=GREY, mew=0, label="data")
            ax.plot(c1.T_DENSE, y_lib[:, 2 * i + j], color=ORANGE, lw=1.0, label=f"best library: {nice(bl)}")
            ax.plot(c1.T_DENSE, y_sr[:, 2 * i + j], color=BLUE, lw=1.0, label="discovered law")
            if j == 0:
                ax.set_title(f"run {run['key']}")
        axes[1, i].set_xlabel("time [h]")
    axes[0, 0].set_ylabel(r"$X$ [g L$^{-1}$]")
    axes[1, 0].set_ylabel(r"$S$ [g L$^{-1}$]")
    h, l = axes[0, 0].get_legend_handles_labels()
    # later run (seed 0)
    lp = loop.get("0")
    if lp is not None and lp["later"] is not None:
        later = c1.make_runs("hidden", 0, keys=("d",))
        base, plus = lp["later"]["base"], lp["later"]["plus"]
        b0 = min(base, key=lambda m: base[m]["BIC"])
        yb = c1.simulate(c1.LIBRARY[b0], base[b0]["theta"], c1.T_DENSE, base[b0]["x0"], later)
        new = c1.accepted_entry(lp["acceptance"]["name"], res["joint"]["all"][lp["acceptance"]["name"]])
        yp = c1.simulate(new, plus["SR-1"]["theta"], c1.T_DENSE, plus["SR-1"]["x0"], later)
        for j in range(2):
            ax = axes[j, 3]
            ax.plot(c1.T_OBS, later[0]["Y"][:, j], "o", ms=1.8, color=GREY, mew=0)
            ax.plot(c1.T_DENSE, yb[:, j], color=ORANGE, lw=1.0, label=f"original library: {nice(b0)}")
            ax.plot(c1.T_DENSE, yp[:, j], color=BLUE, lw=1.0, label="enlarged library: SR-1")
        axes[0, 3].set_title("later run d (one run)")
        axes[1, 3].set_xlabel("time [h]")
        h2, l2 = axes[0, 3].get_legend_handles_labels()
        h, l = h + h2, l + l2
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(h_pad=0.5, w_pad=0.5, rect=(0, 0.07, 1, 1))
    savefig(fig, "c1_hidden")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    by, seeds = collect()
    M = Macros("numbers_c1.tex")
    M["ConeSeeds"] = len(seeds)
    acc4, w4, s4, _ = selection_stats(by, seeds, "single", c1.LIB4, c1.LIB4)
    accs, ws, ss, Ws = selection_stats(by, seeds, "single", LIB, EXPS)
    accj, wj, sj, Wj = selection_stats(by, seeds, "joint", LIB, EXPS)
    libexp = list(c1.TRUE)
    M["ConeAccFour"] = pct(np.mean([acc4[e] for e in c1.LIB4]))
    M["ConeAccSingle"] = pct(np.mean([accs[e] for e in libexp]))
    M["ConeAccJoint"] = pct(np.mean([accj[e] for e in libexp]))
    M["ConeShortSingle"] = pct(np.mean([ss[e] for e in libexp]))
    M["ConeShortJoint"] = pct(np.mean([sj[e] for e in libexp]))
    M["ConeWtrueSingle"] = fmt(np.mean([ws[e] for e in libexp]))
    M["ConeWtrueJoint"] = fmt(np.mean([wj[e] for e in libexp]))
    # adequacy
    ad_s = [by[(e, sd)]["single"]["library_adequate"] for e in libexp for sd in seeds if (e, sd) in by]
    ad_j = [by[(e, sd)]["joint"]["library_adequate"] for e in libexp for sd in seeds if (e, sd) in by]
    hs = [by[("hidden", sd)]["single"]["library_adequate"] for sd in seeds if ("hidden", sd) in by]
    hj = [by[("hidden", sd)]["joint"]["library_adequate"] for sd in seeds if ("hidden", sd) in by]
    M["ConeAdeqLibSingle"], M["ConeAdeqLibJoint"] = pct(np.mean(ad_s)), pct(np.mean(ad_j))
    M["ConeHiddenFlagSingle"], M["ConeHiddenFlagJoint"] = sum(not x for x in hs), sum(not x for x in hj)
    M["ConeHiddenN"] = len(hs)
    # node
    loss_s = [by[(e, sd)]["single"]["node"]["loss"] for e in EXPS for sd in seeds if (e, sd) in by]
    loss_j = [by[(e, sd)]["joint"]["node"]["loss"] for e in EXPS for sd in seeds if (e, sd) in by]
    M["ConeNodeLossSingle"], M["ConeNodeLossJoint"] = sci(np.median(loss_s)), sci(np.median(loss_j))
    tsec = [by[(e, sd)]["joint"]["seconds"] + by[(e, sd)]["single"]["seconds"] for e in EXPS for sd in seeds if (e, sd) in by]
    M["ConeMinutes"] = f"{np.median(tsec) / 60:.1f}"
    # selection table
    lines = []
    for e in libexp:
        a4 = pct(acc4[e]) if e in acc4 else "--"
        lines.append(f"{nice(e)} & {a4} & {pct(accs[e])} & {fmt(ws[e])} & {pct(accj[e])} & {fmt(wj[e])} \\\\")
    lines.append("\\midrule")
    lines.append(f"Mean & {pct(np.mean(list(acc4.values())))} & {pct(np.mean([accs[e] for e in libexp]))} & "
                 f"{fmt(np.mean([ws[e] for e in libexp]))} & {pct(np.mean([accj[e] for e in libexp]))} & "
                 f"{fmt(np.mean([wj[e] for e in libexp]))} \\\\")
    write_tex("tab_c1_selection.tex", "\n".join(lines) + "\n")
    # symbolic regression
    srs, srj = sr_stats(by, seeds, "single"), sr_stats(by, seeds, "joint")
    for tag, rows in [("Single", srs), ("Joint", srj)]:
        lib_rows = [r for (e, _), r in rows.items() if e != "hidden"]
        M[f"ConeFalseDisc{tag}"] = sum(r["accept"] for r in lib_rows)
        M[f"ConeLibCases{tag}"] = len(lib_rows)
        hid = [r for (e, _), r in rows.items() if e == "hidden"]
        M[f"ConeHiddenFound{tag}"] = sum(r["redisc_near"] for r in hid)
        M[f"ConeHiddenBest{tag}"] = sum(r["overall"].startswith("SR") for r in hid)
        M[f"ConeHiddenAccepted{tag}"] = sum(r["accept"] for r in hid)
    # rediscovery of library laws (joint)
    red = {}
    for e in libexp:
        rr = [r for (ee, _), r in srj.items() if ee == e]
        red[e] = (np.mean([r["redisc_near"] for r in rr]), np.median([r["dbic"] for r in rr]))
    lines = [f"{nice(e)} & {pct(red[e][0]) if true_terms(e) is not None else '--'} & {red[e][1]:.1f} \\\\"
             for e in libexp]
    write_tex("tab_c1_sr.tex", "\n".join(lines) + "\n")
    M["ConeAibaRedisc"] = pct(red["aiba"][0])
    M["ConeTessierRedisc"] = pct(red["tessier"][0])
    # escalation on hidden (joint)
    esc = [by[("hidden", sd)]["joint"] for sd in seeds if ("hidden", sd) in by]
    M["ConeEscalated"] = sum(r.get("escalated", False) for r in esc)
    M["ConeScreened"] = int(np.median([r.get("n_screened", 0) for r in esc if r.get("escalated")] or [0]))
    hid = [r for (e, _), r in srj.items() if e == "hidden"]
    # structurally equivalent: S replaced by S exp(-S/K) with exp(-Smax/K) >= 0.8 over the observed substrate range
    equiv, kvals = 0, []
    for sd in seeds:
        rec = by[("hidden", sd)]["joint"]["all"]
        bs = best(rec, False)
        law = rec[bs]["law"]
        terms = set(law["terms"])
        smax = max(np.nanmax(r_["Y"][:, 1]) for r_ in by[("hidden", sd)]["runs"])
        if terms == (TRUE_TERMS["hidden"] - {"S"}) | {"S*exp(-S/K)"} and np.exp(-smax / law["q"]["K"]) >= 0.8:
            equiv += 1
            kvals.append(law["q"]["K"])
    M["ConeHiddenEquivJoint"] = equiv
    if kvals:
        M["ConeHiddenKmin"], M["ConeHiddenKmax"] = f"{min(kvals):.0f}", f"{max(kvals):.0f}"
    M["ConeHiddenChiSR"] = f"{np.median([r['chi2_sr'] for r in hid]):.0f}"
    M["ConeHiddenChiLib"] = f"{np.median([r['chi2_lib'] for r in hid]):.0f}"
    M["ConeChiOK"] = f"{c1.chi2_ok(294):.0f}"
    M["ConeChiOKsingle"] = f"{c1.chi2_ok(98):.0f}"
    hs_ = [r for (e, _), r in srs.items() if e == "hidden"]
    M["ConeHiddenChiLibSingle"] = f"{np.median([r['chi2_lib'] for r in hs_]):.0f}"
    M["ConeHiddenChiSRSingle"] = f"{np.median([r['chi2_sr'] for r in hs_]):.0f}"
    # example discovered law (seed 0), as LaTeX, and its physical parameters
    import sympy as sp
    from kinid.sr import sr_formula
    rec0 = by[("hidden", 0)]["joint"]["all"]
    bs0 = best(rec0, False)
    law0 = rec0[bs0]["law"]
    Xs, Ss = sp.symbols("X S", positive=True)
    expr = sr_formula(c1.SR_C1, law0)
    write_tex("c1_hidden_law_tex.tex", sp.latex(expr) + "\n")
    if set(law0["terms"]) == TRUE_TERMS["hidden"]:
        c = dict(zip(law0["terms"], law0["coef"]))
        M["ConeHidMumax"] = fmt(c["S"])
        M["ConeHidXmax"] = f"{-c['S'] / c['S*X']:.1f}"
        M["ConeHidKs"] = fmt(-c["mu"])
        M["ConeHidKi"] = f"{-1 / c['mu*S^2']:.1f}"
    # before escalation: was the true structure shortlisted, was the best shortlisted law adequate?
    short_true, pre_ok, errs_a, errs_bc = 0, 0, [], []
    for sd in seeds:
        j = by[("hidden", sd)]["joint"]
        short_true += any(set(law["terms"]) == TRUE_TERMS["hidden"] for law, _, _ in j["sr_candidates"])
        pre = [m for m in j["all"] if not j["all"][m]["library"] and not m.startswith("SR (screened)")]
        pre_ok += min(j["all"][m]["chi2"] for m in pre) <= c1.chi2_ok(j["n"])
        for k, key in enumerate("abc"):
            d = c1.DESIGN[key]
            tr = c1.simulate_runs(c1.hidden_mu, c1.HIDDEN_P, c1.T_DENSE, [d["X0"], d["S0"]], [d])
            mt = c1.hidden_mu(tr[:, 0], np.clip(tr[:, 1], 0, None), c1.HIDDEN_P)
            e_ = np.sqrt(np.mean((j["node"]["mur"][k] - mt) ** 2)) / mt.max()
            (errs_a if key == "a" else errs_bc).append(e_)
    M["ConeHiddenShortTrue"], M["ConeHiddenPreOK"] = short_true, pre_ok
    M["ConeHiddenErrAmed"], M["ConeHiddenErrAmax"] = pct(np.median(errs_a)), pct(np.max(errs_a))
    M["ConeHiddenErrBCmax"] = pct(np.max(errs_bc))
    # loop
    from report_common import load_all as la
    loop = la("case1_loop/*.pkl")
    if loop:
        acc = [lp["acceptance"]["accepted"] for lp in loop.values() if lp["acceptance"]]
        M["ConeLoopN"], M["ConeLoopAccepted"] = len(loop), sum(acc)
        later = [lp["later"] for lp in loop.values() if lp["later"]]
        sel_base = [min(L["base"], key=lambda m: L["base"][m]["BIC"]) for L in later]
        sel_plus = [min(L["plus"], key=lambda m: L["plus"][m]["BIC"]) for L in later]
        M["ConeLaterPlus"] = sum(s == "SR-1" for s in sel_plus)
        better = [L["plus"]["SR-1"]["chi2"] < min(L["base"][m]["chi2"] for m in L["base"]) for L in later]
        M["ConeLaterBetterChi"] = sum(better)
        dchi = [min(L["base"][m]["chi2"] for m in L["base"]) - L["plus"]["SR-1"]["chi2"] for L in later]
        M["ConeLaterDchiMax"] = f"{max(dchi):.0f}"
        M["ConeLaterStrong"] = sum(L["plus"]["SR-1"]["weight"] > 0.9 for L in later)
        M["ConeLaterWplus"] = fmt(np.median([L["plus"]["SR-1"]["weight"] for L in later]))
        M["ConeLaterBaseTop"] = nice(max(set(sel_base), key=sel_base.count))
        M["ConeLaterBaseCount"] = sel_base.count(max(set(sel_base), key=sel_base.count))
        steal = [s["weight_new"] for lp in loop.values() for s in lp["steal"].values()]
        changed = [s["selected"] != e for lp in loop.values() for e, s in lp["steal"].items()]
        M["ConeStealMax"] = fmt(max(steal)) if steal else "--"
        M["ConeStealChanged"] = sum(changed)
        M["ConeStealCases"] = len(changed)
    else:
        loop = {}
    # ablation
    ab = la("case1_ablation/*.pkl")
    if ab:
        lines = []
        for e in c1.LIB4:
            f = [r for r in ab.values() if r["name"] == e and r["factor"]]
            s = [r for r in ab.values() if r["name"] == e and not r["factor"]]
            lines.append(f"{nice(e)} & {np.median([r['loss'] for r in f]) * 1e3:.2f} & "
                         f"{np.median([r['loss'] for r in s]) * 1e3:.2f} & {np.median([r['mu_rel_err'] for r in f]):.2f} & "
                         f"{np.median([r['mu_rel_err'] for r in s]):.2f} & \\ensuremath{{{np.median([r['S_min'] for r in f]):.1f}}} & "
                         f"\\ensuremath{{{np.median([r['S_min'] for r in s]):.1f}}} \\\\")
        write_tex("tab_c1_ablation.tex", "\n".join(lines) + "\n")
    M.write()
    fig_design(by)
    fig_learned_mu(by)
    fig_confusion(Ws, Wj)
    fig_sr(srs, srj)
    fig_hidden(by, loop)
    print("case 1 report written;", len(by), "results")


if __name__ == "__main__":
    main()
