"""Figures from screening outputs: funnel, arm comparison, property map + retrospective selection benchmark, structure panel."""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Element
sys.path.insert(0, os.path.dirname(__file__))
import style as S
S.apply()
ARMS = json.load(open("work/arms.json"))
COL = {"gap14": S.CLAY, "gap14g4": S.GOLD, "base": S.BLUE, "enum": S.GREEN, "legacy": S.PLUM}
SHORT = {"gap14": "MatterGen\n(1.4 eV)", "gap14g4": "MatterGen\n(1.4 eV, \u03b3=4)", "base": "MatterGen\nunconditional", "enum": "Prototype\nsubstitution", "legacy": "Legacy 45-D\n+ prototypes"}
out = lambda n: (f"paper/figures/{n}.pdf", f"paper/figures/{n}.png")
def save(fig, n):
    p, q = out(n); fig.savefig(p); fig.savefig(q, dpi=300); plt.close(fig)

             
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
stages = ["generated", "parsed", "geometry", "ABX3 stoichiometry", "3D perovskite", "charge balanced", "tolerance window / unassessable", "unique"]
lab = ["generated", "valid periodic", "geometry", r"ABX$_3$", "3D perovskite", "charge balanced", "tolerance", "unique"]
for arm in ("gap14", "base"):
    f = pd.DataFrame(json.load(open(f"work/{arm}_funnel.json"))).set_index("stage").loc[stages]
    ax[0].plot(range(len(stages)), f.retained_total * 100, "o-", color=COL[arm], label=SHORT[arm].replace("\n", " "))
ax[0].set_yscale("log"); ax[0].set_xticks(range(len(stages))); ax[0].set_xticklabels(lab, rotation=40, ha="right", fontsize=7.5)
ax[0].set_ylabel("Retained of all generated (%)"); ax[0].set_title("(a) Generated structures"); ax[0].legend(); S.finish(ax[0], "y")
stages2 = stages[3:]
for k, arm in enumerate(("gap14", "gap14g4", "base", "enum", "legacy")):
    f = pd.DataFrame(json.load(open(f"work/{arm}_funnel.json"))).set_index("stage").loc[stages2]
    ax[1].bar(np.arange(len(stages2)) + k * 0.16 - 0.32, f.n / f.n.iloc[0] * 100, 0.16, color=COL[arm], label=SHORT[arm].replace("\n", " "))
ax[1].set_xticks(range(len(stages2))); ax[1].set_xticklabels(lab[3:], rotation=40, ha="right", fontsize=7.5)
ax[1].set_ylabel(r"Retained of ABX$_3$ structures (%)"); ax[1].set_title("(b) After stoichiometry"); S.finish(ax[1], "y")
save(fig, "fig_funnel")

                     
T = pd.read_csv("results/arm_comparison.csv").set_index("arm").loc[list(ARMS)]
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.7), constrained_layout=True)
panels = [("yield_unique_per_1000", "(a) Unique perovskites\nper 1000 structures"), ("viable_per_1000_generated", "(b) Qualifying per 1000\n(gap, hull, spread)"),
          ("frac_new_vs_both", "(c) Fraction structurally\nnew vs MP and Alex-MP-20"), ("frac_high_uncertainty", "(d) Fraction with\nspread > 0.3 eV")]
for a, (c, t) in zip(ax, panels):
    a.bar(range(len(T)), T[c], color=[COL[x] for x in T.index]); a.set_xticks(range(len(T))); a.set_xticklabels([SHORT[x] for x in T.index], fontsize=6.5, rotation=60, ha="right")
    a.set_title(t, fontsize=8.5); S.finish(a, "y")
    if c.startswith("yield") or c.startswith("viable"): a.set_yscale("log")
save(fig, "fig_arms")

                                             
pool = pd.read_csv("work/cand/gap14_pool.csv"); sl = pd.read_csv("results/ranked_table.csv"); R = pd.read_csv("results/retro_benchmark.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]; a.axvspan(1.1, 1.7, color=S.GOLD, alpha=.2, lw=0); a.axhline(0.10, color=S.INK, ls="--", lw=.8)
sc = a.scatter(pool.gap, pool.ehull, c=np.log10(pool.gap_std.clip(lower=1e-3)), cmap="viridis", s=14, vmin=-2.3, vmax=0, lw=0, alpha=.85)
sh = sl[sl.shortlist]; a.errorbar(sh.gap, sh.ehull, xerr=sh.gap_std, yerr=sh.ehull_std, fmt="none", ecolor=S.INK, lw=.8, zorder=3)
a.scatter(sh.gap, sh.ehull, marker="*", s=120, color=S.CLAY, edgecolor=S.INK, lw=.7, zorder=4, label="shortlist")
for _, r in sh.iterrows(): a.annotate(r.candidate[-2:], (r.gap, r.ehull), xytext=(4, 5), textcoords="offset points", fontsize=7)
a.set_xlabel(r"predicted $E_{\rm g}$ (eV)"); a.set_ylabel(r"predicted $E_{\rm hull}$ (eV/atom)"); a.set_ylim(-0.01, min(0.6, pool.ehull.max() * 1.05)); a.set_title("(a) Generated perovskites"); a.legend(loc="upper right")
cb = fig.colorbar(sc, ax=a, shrink=.8, pad=.01); cb.set_label(r"$\log_{10}\sigma_g$"); S.finish(a, "both")
b = ax[1]; r = R[(R.split == "random") & (R.k == 5)].set_index("method")
meths = ["random", "gap_only", "stability_aware", "uncertainty_aware", "proposed"]; nm = ["random", "gap only", "+hull gate", "+spread", "proposed"]
b.bar(range(5), r.loc[meths].hit_rate, color=[S.GREY if False else "#BFBBAE"] * 4 + [S.CLAY]); b.set_xticks(range(5)); b.set_xticklabels(nm, rotation=40, ha="right", fontsize=7.5)
b.set_ylabel("hit rate (true gap in 1.4±0.3 eV,\ntrue $E_{hull}$≤0.1), top-5"); b.set_title("(b) Retrospective test on\nheld-out MP perovskites"); S.finish(b, "y")
save(fig, "fig_map")

                      
def project(s, ncell=(2, 2, 1)):
    sc_ = s.copy(); sc_.make_supercell(ncell); return sc_
from matplotlib.patches import Circle
ELCOL = {}
def color(sym):
    import colorsys
    z = Element(sym).Z; h = (z * 0.61803) % 1; return colorsys.hls_to_rgb(h, 0.55, 0.55)
fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.2), constrained_layout=True)
axs = axs.ravel()
for extra in axs[len(sh):]: extra.axis("off")
X = {"O", "F", "S", "Cl", "Se", "Br", "Te", "I", "N"}
for a, (_, r) in zip(axs, sh.iterrows()):
    s = Structure.from_file(f"candidates/structures/{r.candidate}.cif"); sup = project(s, (2, 2, 1)); sup = Structure(sup.lattice, sup.species, sup.frac_coords % 1.0)
    C = sup.cart_coords; P = np.c_[C[:, 0] + 0.25 * C[:, 2], C[:, 1] + 0.15 * C[:, 2]]; z = C[:, 2]
    D = sup.distance_matrix; Bs = set((r.Bsite or "").split(","))
    for i, site in enumerate(sup):
        if site.specie.symbol not in Bs: continue
        xs = [j for j in range(len(sup)) if sup[j].specie.symbol in X and D[i, j] > 0.1]
        if not xs: continue
        dmin = min(D[i, j] for j in xs)
        for j in xs:
            if D[i, j] <= 1.25 * dmin and np.linalg.norm(C[i] - C[j]) <= 1.25 * dmin + 0.05: a.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="#777777", lw=.5, zorder=1)
    for i in np.argsort(z):
        sym = sup[i].specie.symbol; rad = 0.5 if sym in X else (0.42 if sym in Bs else 0.72)
        a.add_patch(Circle(P[i], rad, color=color(sym), ec="k", lw=.3, zorder=2 + z[i] / 100))
    a.set_aspect("equal"); a.autoscale_view(); a.axis("off")
    els = sorted({x.specie.symbol for x in sup}); a.set_title(f"{r.candidate}  {r.formula}  ({r.spacegroup})", fontsize=8.5)
    for k, e in enumerate(els): a.text(0.5 + (k - (len(els) - 1) / 2) * 0.2, -0.03, e, transform=a.transAxes, ha="center", va="top", fontsize=7, color=color(e), weight="bold")
save(fig, "fig_structures"); print("figures ok")
