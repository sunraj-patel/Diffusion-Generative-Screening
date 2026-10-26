"""Chemistry of the generated, screened perovskites versus the MP and Alex-MP-20 perovskites. Writes results/chemistry.json and fig_chem."""
import json, os, sys, collections
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import style as S
S.apply()
g = pd.read_csv("work/cand/gap14_pool.csv").fillna(""); b = pd.read_csv("work/cand/base_pool.csv").fillna(""); e = pd.read_csv("work/cand/enum_pool.csv").fillna("")
lab = pd.read_csv("data/curated/labels.csv").fillna(""); mp = lab[lab.is_perovskite]
al = pd.read_csv("data/curated/alex_abx3_labels.csv").fillna(""); al = al[al.family == "3D corner-sharing perovskite"]
def share(df, col, top=None):
    c = collections.Counter(x for v in df[col] for x in set(v.split(",")) if x); n = len(df); return {k: v / n for k, v in c.items()}
def anion_class(x): return "oxide" if x == "O" else "fluoride" if x == "F" else "chloride/bromide/iodide" if set(x.split(",")) <= {"Cl", "Br", "I"} else "chalcogenide (S,Se,Te)" if set(x.split(",")) <= {"S", "Se", "Te"} else "mixed/other"
res = {}
for nm, df, xc in (("MatterGen gap-conditioned", g, "X"), ("MP perovskites", mp, "X"), ("Alex-MP-20 perovskites", al, None), ("Prototype substitution", e, "X")):
    if xc is None: res[nm] = {"n": len(df)}; continue
    cl = collections.Counter(anion_class(x) for x in df[xc]); res[nm] = {"n": len(df), **{k: v / len(df) for k, v in cl.items()}}
n_el = lambda df: df.formula.map(lambda f: len(__import__("pymatgen.core", fromlist=["Composition"]).Composition(f).elements))
res["double_or_mixed_fraction_gen"] = float((n_el(g) >= 4).mean()); res["double_or_mixed_fraction_MP"] = float((n_el(mp) >= 4).mean())
res["spacegroups_gen"] = g.spacegroup if "spacegroup" in g else None
gp = g.copy()
res["mean_gap_by_anion_gen"] = {k: float(v) for k, v in gp.groupby(gp.X.map(anion_class)).gap.mean().items()}
res["count_by_anion_gen"] = {k: int(v) for k, v in gp.X.map(anion_class).value_counts().items()}
topA = collections.Counter(x for v in g.A for x in v.split(",") if x).most_common(8); topB = collections.Counter(x for v in g.Bsite for x in v.split(",") if x).most_common(10)
res["top_A_gen"] = topA; res["top_B_gen"] = topB
trainB = set(x for v in mp.Bsite for x in v.split(",") if x); genB = set(x for v in g.Bsite for x in v.split(",") if x)
cntB = collections.Counter(x for v in mp.Bsite for x in v.split(",") if x)
rare = {k for k, v in cntB.items() if v <= 30}
res["gen_B_elements"] = len(genB); res["gen_B_not_in_MP_perovskites"] = sorted(genB - trainB)
res["frac_gen_with_rare_B"] = float(np.mean([bool(set(v.split(",")) & rare) for v in g.Bsite]))
res["frac_MP_with_rare_B"] = float(np.mean([bool(set(v.split(",")) & rare) for v in mp.Bsite]))
res["frac_gen_formula_new_vs_MP_and_Alex"] = float((~g.formula_in_MP_perovskites & ~g.formula_in_AlexMP20_train).mean())
g["viable"] = ((g.gap - 1.4).abs() <= .3) & (g.ehull <= .1) & (g.gap_std <= .3)
res["viable_by_anion"] = {k: int(v) for k, v in g[g.viable].X.map(anion_class).value_counts().items()}
res["viable_formulas"] = g[g.viable].formula.tolist()
json.dump(res, open("results/chemistry.json", "w"), indent=1, default=str)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.3), constrained_layout=True)
cls = ["oxide", "fluoride", "chloride/bromide/iodide", "chalcogenide (S,Se,Te)", "mixed/other"]; cc = [S.CLAY, S.BLUE, S.GREEN, S.PLUM, S.GOLD]
for k, nm in enumerate(("MP perovskites", "MatterGen gap-conditioned", "Prototype substitution")):
    bot = 0
    for c_, col in zip(cls, cc):
        v = res[nm].get(c_, 0); ax[0].bar(k, v, bottom=bot, color=col, label=c_ if k == 0 else None); bot += v
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["MP", "MatterGen\n(1.4 eV)", "Prototype\nsubst."], fontsize=7.5); ax[0].set_ylabel("Fraction"); ax[0].set_title("(a) Anion family", fontsize=9); fig.legend(*ax[0].get_legend_handles_labels(), loc="lower left", fontsize=6.5, ncol=5, frameon=False)
sg = g.groupby(g.X.map(anion_class)).gap
for k, c_ in enumerate(cls):
    v = g[g.X.map(anion_class) == c_].gap
    if len(v): ax[1].scatter(np.full(len(v), k) + np.random.RandomState(0).uniform(-.2, .2, len(v)), v, s=8, color=cc[k], lw=0, alpha=.8)
ax[1].axhspan(1.1, 1.7, color=S.GOLD, alpha=.2, lw=0); ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["O", "F", "Cl/Br/I", "S/Se/Te", "mixed"], fontsize=7.5); ax[1].set_ylabel(r"predicted $E_{\rm g}$ (eV)"); ax[1].set_title("(b) Gap by anion", fontsize=9); S.finish(ax[1], "y")
tb = res["top_B_gen"]; ax[2].barh(range(len(tb)), [v for _, v in tb][::-1], color=S.BLUE); ax[2].set_yticks(range(len(tb))); ax[2].set_yticklabels([k for k, _ in tb][::-1], fontsize=7.5); ax[2].set_xlabel("generated perovskites"); ax[2].set_title("(c) Common B sites", fontsize=9); S.finish(ax[2], "x")
fig.savefig("paper/figures/fig_chem.pdf"); fig.savefig("paper/figures/fig_chem.png", dpi=300)
print(json.dumps({k: v for k, v in res.items() if k not in ("spacegroups_gen", "viable_formulas")}, indent=1, default=str)[:3500]); print(res["viable_formulas"])

                                                                 
from scipy.stats import ks_2samp, spearmanr
out = {}
mpg = mp.band_gap.values
for arm in ("gap14", "base", "gap14g4", "enum", "legacy"):
    d = pd.read_csv(f"work/cand/{arm}_pool.csv")
    out[arm] = dict(n=len(d), median_pred_gap=float(d.gap.median()), frac_metal_pred=float((d.gap < 0.1).mean()), frac_in_window=float(((d.gap - 1.4).abs() <= .3).mean()),
                    ks_vs_MP_labels=float(ks_2samp(d.gap, mpg).statistic), ks_p=float(ks_2samp(d.gap, mpg).pvalue),
                    spearman_gap_ehull=float(spearmanr(d.gap, d.ehull).correlation), spearman_gap_std_gap=float(spearmanr(d.gap, d.gap_std).correlation),
                    spearman_novelty_std=float(spearmanr(d.novelty, d.gap_std).correlation))
out["MP_labels"] = dict(n=len(mp), median_gap=float(np.median(mpg)), frac_metal=float((mpg < 0.1).mean()), frac_in_window=float((np.abs(mpg - 1.4) <= .3).mean()))
json.dump(out, open("results/gap_distribution.json", "w"), indent=1); print(json.dumps(out, indent=1))
