"""Audit D/C: re-match generated unique perovskites against MP and Alex-MP-20 (same StructureMatcher as the pipeline, pipeline/crystal.py:177),
record WHICH reference entry matched, and compare the predictor on the generated geometry with the reference label."""
import gzip, json, warnings, numpy as np, pandas as pd
from collections import defaultdict
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings("ignore")
O = "reports/cms_audit/out/"
sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
g = pd.read_csv("work/cand/gap14_pool.csv", keep_default_na=False, na_values=[""]); g = g[g.unique & g.perovskite].copy()
pool = {}
for l in gzip.open("work/pool_gap14.jsonl.gz", "rt"):
    d = json.loads(l); pool[d["id"]] = Structure.from_dict(d["structure"])
lab = pd.read_csv("data/curated/labels.csv", keep_default_na=False, na_values=[""]).set_index("id")
alexl = pd.read_csv("data/curated/alex_abx3_labels.csv", keep_default_na=False, na_values=[""]).set_index("id")
need = set(g.formula.map(lambda f: __import__("pymatgen.core", fromlist=["Composition"]).Composition(f).reduced_formula))
def load(path, keep):
    D = defaultdict(list)
    for l in gzip.open(path, "rt"):
        d = json.loads(l); s = Structure.from_dict(d["structure"]); rf = s.composition.reduced_formula
        if rf in keep: D[rf].append((d["id"], s))
    return D
MP, AX = load("data/curated/structures.jsonl.gz", need), load("data/curated/alex_perovskites.jsonl.gz", need)
rows = []
for _, r in g.iterrows():
    s = pool[r.id]; rf = s.composition.reduced_formula
    mp = [i for i, t in MP.get(rf, []) if sm.fit(s, t)]; ax = [i for i, t in AX.get(rf, []) if sm.fit(s, t)]
    row = dict(id=r.id, formula=r.formula, pred_gap=r.gap, pred_std=r.gap_std, in_MP=bool(mp), in_Alex=bool(ax), mp_id=";".join(mp[:3]), alex_id=";".join(ax[:3]),
               mp_label=lab.loc[mp[0], "band_gap"] if mp else np.nan, mp_sg=lab.loc[mp[0], "spacegroup"] if mp else "",
               alex_label=alexl.loc[ax[0], "gap"] if ax and ax[0] in alexl.index else np.nan)
    rows.append(row)
R = pd.DataFrame(rows); R.to_csv(O + "generated_vs_reference.csv", index=False)
print("n", len(R), "in MP", R.in_MP.sum(), "in Alex", R.in_Alex.sum(), "Alex not MP", (R.in_Alex & ~R.in_MP).sum(), "either", (R.in_MP | R.in_Alex).sum(), "neither", (~(R.in_MP | R.in_Alex)).sum())
ax_only = R[R.in_Alex & ~R.in_MP]; print("Alex-not-MP with Alex gap label:", ax_only.alex_label.notna().sum(), "of", len(ax_only))
m = R[R.in_MP & R.mp_label.notna()].copy(); m["err"] = m.pred_gap - m.mp_label
print("\nGEOMETRY-SENSITIVITY: generated structures matching an MP entry (ALIGNN on generated geometry vs MP label of the relaxed MP entry) n=%d" % len(m))
print("  MAE %.3f  RMSE %.3f  median|err| %.3f  bias %.3f  Spearman %.3f" % (m.err.abs().mean(), np.sqrt((m.err**2).mean()), m.err.abs().median(), m.err.mean(), m[["pred_gap","mp_label"]].corr("spearman").iloc[0,1]))
w = m[(m.mp_label >= .5) & (m.mp_label <= 3)]; print("  label 0.5-3 eV n=%d MAE %.3f" % (len(w), w.err.abs().mean()))
print("  generated predicted <0.1 eV but label >=0.5:", int(((m.pred_gap < .1) & (m.mp_label >= .5)).sum()), "| label 0 but predicted >=0.5:", int(((m.mp_label == 0) & (m.pred_gap >= .5)).sum()))
a = ax_only.dropna(subset=["alex_label"]); 
if len(a): print("  Alex-only n=%d  MAE %.3f (Alex gap is a different DFT workflow)" % (len(a), (a.pred_gap - a.alex_label).abs().mean()))
print("\nlargest MP-matched disagreements:\n", m.reindex(m.err.abs().sort_values(ascending=False).index).head(10)[["formula","pred_gap","mp_label","mp_id","mp_sg"]].to_string())
