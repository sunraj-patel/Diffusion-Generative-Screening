"""Stage E: choose the final five (and a ranked table of 15) from the generated, band-gap-conditioned pool and export the package.
usage: python finalize.py ARM   (reads work/cand/ARM_pool.csv + unique structures; writes candidates/, results/shortlist.csv, results/ranked_table.csv)"""
import gzip, json, os, sys, glob
import numpy as np, pandas as pd
from pymatgen.core import Structure
sys.path.insert(0, os.path.dirname(__file__))
import rank, candidates
from crystal import classify_perovskite, tolerance_factor, charge_balance

arm = sys.argv[1]; uniq = sys.argv[2]
cfg = dict(target=1.4, hull_max=0.10, gap_std_max=0.30, gap_gate=0.60, k=5, n_table=15)
d = pd.read_csv(f"work/cand/{arm}_pool.csv")
structs = {}
for l in gzip.open(uniq, "rt"):
    r = json.loads(l); structs[r["id"]] = Structure.from_dict(r["structure"])
d["new_structure"] = ~(d.struct_in_MP | d.struct_in_AlexMP20)
elig = d[(d.gap_std <= cfg["gap_std_max"]) & ~d.struct_in_MP].copy()                                                                        
ranked = rank.proposed_rank(elig, cfg["target"], hull_max=cfg["hull_max"], gap_gate=cfg["gap_gate"], min_pool=0)
sel = rank.diverse_select(ranked, cfg["k"])
tier1 = pd.concat([sel, ranked[~ranked.id.isin(sel.id)]])
                                                                                                                                     
elig2 = d[(d.gap_std <= cfg["gap_std_max"]) & ~d.id.isin(tier1.id)].copy()
tier2 = rank.proposed_rank(elig2, cfg["target"], hull_max=cfg["hull_max"], gap_gate=0.8, min_pool=0)
tier1["tier"] = 1; tier2["tier"] = 2
table = pd.concat([tier1, tier2]).head(cfg["n_table"])
os.makedirs("candidates", exist_ok=True)
ex = candidates.export(table, structs, "candidates", "PRV")
out = table.reset_index(drop=True).copy(); out = pd.concat([out, ex], axis=1)
out["shortlist"] = [i < min(cfg["k"], len(sel)) for i in range(len(out))]
t = []; cb = []
for _, r in out.iterrows():
    s = structs[r.id]; c = classify_perovskite(s); t.append(tolerance_factor(s, c)); cb.append(str(charge_balance(s.composition.reduced_composition)[1]))
out["tolerance_factor"] = t; out["oxidation_guess"] = cb
def why(r):
    p = [f"predicted E_g {r.gap:.2f}±{r.gap_std:.2f} eV (target {cfg['target']})", f"predicted E_hull {r.ehull:.3f}±{r.ehull_std:.3f} eV/atom",
         f"{r.A or '-'}/{r.Bsite or '-'}/{r.X} family", ("tier 2 alternative: structure already present in MP perovskites (re-discovery of a known structure)" if r.tier == 2 else "structure absent from MP and Alex-MP-20 perovskite sets" if r.new_structure else "structure absent from MP perovskites but present in the generator training set Alex-MP-20")]
    p.append("composition " + ("present in MP perovskites" if r.formula_in_MP_perovskites else "absent from MP perovskites") + ", " + ("present" if r.formula_in_AlexMP20_train else "absent") + " in Alex-MP-20")
    if not (isinstance(r.tolerance_factor, float) and r.tolerance_factor == r.tolerance_factor): p.append("tolerance factor not assessable")
    return "; ".join(p)
def concerns(r):
    c = []
    els = set((r.A + "," + r.Bsite + "," + r.X).split(","))
    tox = sorted(els & {"Tl", "Pb", "Cd", "Hg", "As"})
    if tox: c.append("contains " + "/".join(tox) + " (toxicity)")
    if r.gap_std > 0.2: c.append(f"large ensemble spread on the gap ({r.gap_std:.2f} eV)")
    if not (isinstance(r.tolerance_factor, float) and r.tolerance_factor == r.tolerance_factor): c.append("tolerance factor not assessable (oxidation states or radii unavailable)")
    if r.struct_in_AlexMP20: c.append("structure present in the generator training set (Alex-MP-20)")
    if r.formula_in_MP_perovskites: c.append("composition already listed among MP perovskites")
    if len([e for e in r.Bsite.split(",") if e]) > 1 and len(set(r.Bsite.split(",")) & set(r.A.split(","))) > 0: c.append("same element on A and B sites")
    c.append("unrelaxed generator output; no DFT check of gap, hull energy, magnetism or dynamical stability")
    return "; ".join(c)
out["selection_rationale"] = out.apply(why, axis=1); out["unresolved_concerns"] = out.apply(concerns, axis=1)
os.makedirs("results", exist_ok=True)
out.to_csv("results/ranked_table.csv", index=False); out[out.shortlist].to_csv("results/shortlist.csv", index=False)
print(out[["candidate", "formula", "spacegroup", "gap", "gap_std", "ehull", "ehull_std", "tolerance_factor", "novelty", "shortlist"]].round(3).to_string())
