"""Figure: reference dataset and chemistry (MP perovskites vs Alex-MP-20 perovskites)."""
import collections, json, sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import style as S
S.apply()
lab = pd.read_csv("data/curated/labels.csv"); mp = lab[lab.is_perovskite]
al = pd.read_csv("data/curated/alex_abx3_labels.csv"); al = al[al.family == "3D corner-sharing perovskite"]
T = json.load(open("configs/perovskite.json")) if os.path.exists("configs/perovskite.json") else {"target": 1.4, "win": 0.3}

fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
                    
def top(col, n, df):
    c = collections.Counter(e for v in df[col] for e in v.split(",") if e); return c.most_common(n)
a = ax[0, 0]; y0 = 0; ticks = []; labs = []
cols = {"A": S.CLAY, "Bsite": S.BLUE, "X": S.GREEN}
for site, n in (("A", 8), ("Bsite", 8), ("X", 6)):
    items = top(site, n, mp)
    for k, (e, c) in enumerate(items):
        a.barh(y0 + k, c, color=cols[site], height=0.75); ticks.append(y0 + k); labs.append(e)
    a.text(0.98, 1 - (y0 + n * 0.5 + 0.5) / 29, {"A": "A site", "Bsite": "B site", "X": "X site"}[site], color=cols[site], fontsize=9, weight="bold", ha="right", va="center", transform=a.get_yaxis_transform() if False else a.transAxes)
    y0 += n + 1.2
a.set_yticks(ticks); a.set_yticklabels(labs, fontsize=8); a.invert_yaxis(); a.set_xlabel("MP perovskites (count)")
a.set_title("(a) Site chemistry"); S.finish(a, "x")
              
b = ax[0, 1]; bins = np.arange(0, 8.5, 0.25)
b.hist(mp.band_gap, bins=bins, color=S.BLUE, alpha=.85, label=f"MP, n={len(mp):,}")
b.hist(al.gap.dropna(), bins=bins, histtype="step", color=S.CLAY, lw=1.5, weights=np.full(al.gap.notna().sum(), len(mp) / al.gap.notna().sum()), label="Alex-MP-20 (scaled)")
b.axvspan(T["target"] - T["win"], T["target"] + T["win"], color=S.GOLD, alpha=.25, lw=0)
b.set_xlabel("PBE band gap (eV)"); b.set_ylabel("Count"); b.set_title("(b) Band-gap labels"); b.legend(loc="upper right"); S.finish(b, "y")
            
c = ax[1, 0]; c.hist(np.log10(mp.ehull.clip(lower=1e-3)), bins=40, color=S.GREEN, alpha=.85)
c.axvline(np.log10(0.05), color=S.INK, ls="--", lw=1); c.text(np.log10(0.05) + .05, c.get_ylim()[1] * .9, "0.05 eV/atom", fontsize=8)
c.set_xlabel(r"$\log_{10}\,E_{\rm hull}$ (eV/atom; clipped at $10^{-3}$)"); c.set_ylabel("Count"); c.set_title(r"(c) Stability labels"); S.finish(c, "y")
            
d = ax[1, 1]
rows = [("random: train", (lab.split_random == "train")), ("random: test", (lab.split_random == "test")),
        ("B-disjoint: train", (lab.split_Bdisjoint == "train")), ("B-disjoint: test", (lab.split_Bdisjoint == "test"))]
for k, (n, m) in enumerate(rows):
    d.barh(k, m.sum(), color=S.GREY if False else "#CFCBBE", height=.7); d.barh(k, (m & lab.is_perovskite).sum(), color=S.TEAL, height=.7)
    d.text(m.sum() + 300, k, f"{m.sum():,} ({(m & lab.is_perovskite).sum():,})", va="center", fontsize=8)
d.set_yticks(range(len(rows))); d.set_yticklabels([r[0] for r in rows], fontsize=8); d.invert_yaxis(); d.set_xlim(0, 36000)
d.set_xlabel("Structures (perovskites in teal)"); d.set_title("(d) Splits"); S.finish(d, "x")
fig.savefig("paper/figures/fig_dataset.pdf"); fig.savefig("paper/figures/fig_dataset.png", dpi=300)
print("ok")
