"""Main-text figures for paper_cms (fig_predictors, fig_dft_gaps). Run from the repository root:
    python3 paper_cms/figures_src/make_main_figures.py
Inputs: results/pred_*.csv, results/predictor_metrics.json, reports/cms_dft_extension/dft_vs_alignn.csv.
Output: paper_cms/figures/{fig_predictors,fig_dft_gaps}.{pdf,png}. The gap MAE is computed from the clipped (>= 0) ensemble mean, as in the tables."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#8A8A8A", "#111111"                     
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 4,
    "pdf.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150, "axes.grid": False})

def plab(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")

                                                                               
M = json.load(open("results/predictor_metrics.json"))
fig = plt.figure(figsize=(7.2, 4.5), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.045])
ax = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])], [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]]
cax = fig.add_subplot(gs[0, 3]); fig.add_subplot(gs[1, 3]).axis("off")

def parity(a, split, tgt, lim, unit, tag, title):
    d = pd.read_csv(f"results/pred_{split}_{tgt}.csv"); d = d[d.is_perovskite]
    p = d["mean"].clip(lower=0) if tgt == "bandgap" else d["mean"]
    mae = (p - d.y).abs().mean(); n = len(d)
    sc = a.scatter(d.y, p, c=np.log10(d["std"].clip(lower=1e-3)), s=5, cmap="viridis", vmin=-2.5, vmax=0, lw=0, alpha=.85, rasterized=True)
    a.plot(lim, lim, color=INK, lw=0.6, ls="--"); a.set(xlim=lim, ylim=lim); a.set_aspect("equal")
    if tgt == "bandgap":
        txt = f"$n$ = {n}\nMAE = {mae:.2f} eV\npredict 0: {d.y.abs().mean():.2f} eV\npredict median: {(d.y - d.y.median()).abs().mean():.2f} eV"
    else:
        txt = f"$n$ = {n}\nMAE = {mae:.3f} eV/atom\npredict mean: {M[f'{split}/{tgt}/perovskite_test']['baseline_mae']:.3f}"
    a.text(0.04, 0.96, txt, transform=a.transAxes, va="top", fontsize=6.3, linespacing=1.25,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    a.set_xlabel(f"MP label ({unit})"); a.set_ylabel(f"ALIGNN ({unit})"); a.set_title(title, loc="left", pad=3); plab(a, tag); return sc
sc = parity(ax[0][0], "random", "bandgap", (-0.3, 8), "eV", "a", "Gap, random split")
parity(ax[0][1], "Bdisjoint", "bandgap", (-0.3, 8), "eV", "b", "Gap, B-site-disjoint")
parity(ax[0][2], "random", "ehull", (-0.03, 1.0), "eV/atom", "c", r"$E_{\rm hull}$, random split")
parity(ax[1][0], "Bdisjoint", "ehull", (-0.03, 1.0), "eV/atom", "d", r"$E_{\rm hull}$, B-site-disjoint")
cb = fig.colorbar(sc, cax=cax); cb.set_label(r"$\log_{10}\sigma_{\rm ens}$ (eV)", fontsize=6.8); cb.ax.tick_params(labelsize=6, width=0.5, length=2); cb.outline.set_linewidth(0.4)

a = ax[1][1]; plab(a, "e")
for split, col, mk, lab in (("random", BLUE, "o", "random"), ("Bdisjoint", ORANGE, "s", "B-site-disjoint")):
    u = M[f"{split}/bandgap/perovskite_test"]["uncertainty"]; b = u["bins"]
    a.plot([x["mean_std"] for x in b], [x["mae"] for x in b], marker=mk, color=col, label=f"{lab}  ($\\rho_s$ = {u['spearman_std_abs_err']:.2f})")
a.set_xscale("log"); a.set(xlabel="mean ensemble std in quartile (eV)", ylabel="gap MAE in quartile (eV)"); a.set_title("Error vs ensemble spread", loc="left", pad=3)
a.legend(loc="upper left", handlelength=1.6)
a = ax[1][2]; plab(a, "f"); nom = [.5, .8, .95]
for split, col, mk, lab in (("random", BLUE, "o", "random"), ("Bdisjoint", ORANGE, "s", "B-site-disjoint")):
    cov = M[f"{split}/bandgap/perovskite_test"]["uncertainty"]["gaussian_coverage"]
    a.plot(nom, [cov[k] for k in ("50%", "80%", "95%")], marker=mk, color=col, label=lab)
a.plot([0.4, 1], [0.4, 1], color=INK, ls="--", lw=0.6); a.set(xlim=(0.4, 1.0), ylim=(0.2, 1.02), xlabel="nominal Gaussian coverage", ylabel="observed coverage")
a.set_title("Interval coverage (not calibrated)", loc="left", pad=3); a.legend(loc="lower right", handlelength=1.6)
fig.savefig("paper_cms/figures/fig_predictors.pdf"); fig.savefig("paper_cms/figures/fig_predictors.png"); plt.close(fig)

                                                                                  
d = pd.read_csv("reports/cms_dft_extension/dft_vs_alignn.csv").dropna(subset=["pbe_gap_nscf_eV"])
TEX = {"SrTiO3": r"SrTiO$_3$", "KNbO3": r"KNbO$_3$", "Cs2NaRhBr6": r"Cs$_2$NaRhBr$_6$", "Tl3AsCl6": r"Tl$_3$AsCl$_6$", "TlPbCl3": r"TlPbCl$_3$",
       "Cs3Mg2CdBr9": r"Cs$_3$Mg$_2$CdBr$_9$", "Tl2CuBiCl6": r"Tl$_2$CuBiCl$_6$", "Cs2LiAsI6": r"Cs$_2$LiAsI$_6$"}
d["tex"] = d.formula.map(TEX); ctrl = {"PRV-11", "PRV-09"}
def cat(r): return "transformed" if r.retention == "transformed" else ("control" if r.id in ctrl else "generated")
d["kind"] = d.apply(cat, axis=1)
CSTY = {"control": dict(color=GREY, marker="s", label="known compound (control)"), "generated": dict(color=BLUE, marker="o", label="generated, framework retained"),
        "transformed": dict(color=ORANGE, marker="D", label="generated, transformed on relaxation")}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1, 1.05]), constrained_layout=True)
a = ax[0]; lim = (0.4, 3.0); x = np.array(lim)
a.fill_between(x, x - 0.25, x + 0.25, color="#E8E8E8", lw=0, label=r"$\pm$0.25 eV (random-split MAE)"); a.plot(x, x, color=INK, lw=0.6, ls="--")
off = {"PRV-01": (-4, 7, "right"), "PRV-07": (6, -9, "left"), "PRV-03": (-6, -2, "right"), "PRV-06": (-3, 8, "right"), "PRV-11": (6, -2, "left"),
       "PRV-09": (6, -8, "left"), "CMB-15": (-6, 5, "right"), "PRV-14": (6, 3, "left")}
for _, r in d.iterrows():
    s = CSTY[r.kind]
    a.errorbar(r.pbe_gap_nscf_eV, r.alignn_gap_orig, yerr=r.alignn_gap_orig_std, fmt=s["marker"], color=s["color"], ms=4.5, mew=0.4, mec="white", capsize=1.5, elinewidth=0.7, zorder=3)
    dx, dy, ha = off[r.id]; a.annotate(r.id, (r.pbe_gap_nscf_eV, r.alignn_gap_orig), xytext=(dx, dy), textcoords="offset points", fontsize=5.8, ha=ha)
t = d[d.kind == "transformed"].iloc[0]                                                       
a.annotate("", xy=(t.pbe_gap_nscf_eV, t.alignn_gap_relaxed), xytext=(t.pbe_gap_nscf_eV, t.alignn_gap_orig), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8, ls=":"), zorder=2)
a.plot(t.pbe_gap_nscf_eV, t.alignn_gap_relaxed, marker="D", mfc="white", mec=ORANGE, mew=1.0, ms=4.5, ls="", zorder=3)
a.set(xlim=lim, ylim=lim, xlabel="PBE gap, relaxed structure (eV)", ylabel="ALIGNN gap (eV)"); a.set_aspect("equal"); plab(a, "a")
h = [Line2D([], [], ls="", **{k: v for k, v in CSTY[c].items()}, ms=4.5) for c in CSTY] + [Line2D([], [], ls="", marker="D", mfc="white", mec=ORANGE, mew=1.0, ms=4.5)]
a.legend(h + [plt.Rectangle((0, 0), 1, 1, fc="#E8E8E8")], [CSTY[c]["label"] for c in CSTY] + ["ALIGNN on relaxed geometry", r"$\pm$0.25 eV"], loc="upper left", fontsize=5.8, handletextpad=0.3, borderaxespad=0.2)
b = ax[1]; dd = d.sort_values("alignn_minus_dft_orig_eV").reset_index(drop=True); y = np.arange(len(dd)); plab(b, "b")
b.axvspan(-0.25, 0.25, color="#E8E8E8", lw=0); b.axvline(0, color=INK, lw=0.6)
for i, r in dd.iterrows():
    s = CSTY[r.kind]
    b.errorbar(r.alignn_minus_dft_orig_eV, i, xerr=r.alignn_gap_orig_std, fmt=s["marker"], color=s["color"], ms=4.5, mew=0.4, mec="white", capsize=1.5, elinewidth=0.7, zorder=3)
    if r.kind == "transformed":
        b.plot(r.alignn_minus_dft_relaxedpred_eV, i, marker="D", mfc="white", mec=ORANGE, mew=1.0, ms=4.5, ls="", zorder=3)
        b.annotate("", xy=(r.alignn_minus_dft_relaxedpred_eV - 0.02, i), xytext=(r.alignn_minus_dft_orig_eV + 0.03, i), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8, ls=":"))
b.set_yticks(y); b.set_yticklabels([f"{t}" for t in dd.tex], fontsize=7); b.set_ylim(-0.6, len(dd) - 0.4)
b.set_xlabel("ALIGNN $-$ PBE (eV); bars: ensemble std"); b.grid(axis="y", lw=0.3, color="#DDDDDD"); b.set_axisbelow(True)
mad = dd.alignn_minus_dft_orig_eV.abs().mean(); nin = int((dd.alignn_minus_dft_orig_eV.abs() <= 0.25).sum())
b.text(0.02, 0.98, f"starting geometry: mean $|\\Delta|$ = {mad:.2f} eV;\n{nin} of {len(dd)} within 0.25 eV", transform=b.transAxes, ha="left", va="top", fontsize=6.2)
fig.savefig("paper_cms/figures/fig_dft_gaps.pdf"); fig.savefig("paper_cms/figures/fig_dft_gaps.png"); plt.close(fig); print("ok")
