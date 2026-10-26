"""Publication figures for the DFT extension. Run from the repo root after summarize_dft.py / collect_relaxed.py.
fig_dft_structures: initial vs relaxed structures (side by side, B-X polyhedra drawn via bonds) ; fig_dft_gaps: ALIGNN vs PBE gaps ; both -> paper_cms/figures/*.pdf,png"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

from pymatgen.core import Structure
d = pd.read_csv("reports/cms_dft_extension/dft_vs_alignn.csv"); r = pd.read_csv("reports/cms_dft_extension/structural_retention.csv").set_index("id")

                                                                        

                                                                                                                                           
import re
ELEM_COL = {"Cs": "#56B4E9", "Tl": "#CC79A7", "Na": "#F0E442", "Rh": "#7B3294", "As": "#8DA0CB", "Mg": "#E6AB02", "Cd": "#999999", "Pb": "#4D4D4D",
            "Br": "#D55E00", "Cl": "#009E73", "I": "#6A3D9A", "Sr": "#56B4E9", "Ti": "#0072B2", "K": "#56B4E9", "Nb": "#0072B2", "O": "#D55E00"}
ANION = {"O", "F", "Cl", "Br", "I", "S", "Se"}
def draw(ax, s, rep=(2, 2, 1)):
    sc = s.copy(); sc.make_supercell(list(rep)); xy = sc.cart_coords; syms = [x.specie.symbol for x in sc]
    for i, nbl in enumerate(sc.get_all_neighbors(3.2)):
        if syms[i] in ANION: continue
        ds = [n for n in nbl if n.specie.symbol in ANION]
        if not ds: continue
        m = min(n.nn_distance for n in ds)
        if m > 3.0: continue
        lo, hi = xy[:, :2].min(0) - 0.05, xy[:, :2].max(0) + 0.05
        for n in ds:
            if n.nn_distance <= 1.25 * m and (n.coords[:2] >= lo).all() and (n.coords[:2] <= hi).all(): ax.plot([xy[i, 0], n.coords[0]], [xy[i, 1], n.coords[1]], color="#777777", lw=0.6, zorder=1)
    for i in np.argsort(xy[:, 2]): ax.scatter(xy[i, 0], xy[i, 1], s=22 if syms[i] in ANION else 60, color=ELEM_COL.get(syms[i], "#AAAAAA"), edgecolor="white", lw=0.4, zorder=2 + xy[i, 2] / 100)
    ax.set_aspect("equal"); ax.axis("off"); return set(syms)
TEXF = lambda f: re.sub(r"(\d+)", r"$_{\1}$", f)
ids = ["PRV-07", "PRV-06", "CMB-15", "PRV-14"]; allsym = set()
fig, axs = plt.subplots(2, len(ids), figsize=(7.2, 3.9), constrained_layout=True)
for j, c in enumerate(ids):
    s0 = Structure.from_dict(json.load(open(f"dft_ext/structures/{c}_standardized.json"))); s1 = Structure.from_dict(json.load(open(f"dft_ext/structures/{c}_relaxed.json")))
    allsym |= draw(axs[0, j], s0); draw(axs[1, j], s1)
    axs[0, j].set_title(f"{c}  {TEXF(s0.composition.reduced_formula)}", fontsize=7.5, pad=2)
    axs[1, j].set_title(r.loc[c, "retention"], fontsize=6.8, pad=0, color="#D55E00" if r.loc[c, "retention"] == "transformed" else "#111111")
for k, lab in enumerate(("starting\nstructure", "PBE-relaxed\nstructure")): axs[k, 0].text(-0.04, 0.5, lab, transform=axs[k, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ELEM_COL.get(k, "#AAAAAA"), label=k, ms=5, mec="white", mew=0.4) for k in sorted(allsym)], loc="lower center", ncol=len(allsym), fontsize=6.8, frameon=False, handletextpad=0.1, columnspacing=1.0, bbox_to_anchor=(0.5, -0.045))
fig.savefig("paper_cms/figures/fig_dft_structures.pdf", bbox_inches="tight"); fig.savefig("paper_cms/figures/fig_dft_structures.png", bbox_inches="tight", dpi=400); plt.close(fig); print("figs ok")
