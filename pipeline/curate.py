"""Build curated datasets from raw MP downloads (data/raw) -> data/curated.
Outputs: structures.jsonl.gz (pymatgen dicts), labels.csv, splits.json, curation_report.json"""
import json, gzip, sys, os, collections
import numpy as np, pandas as pd
from pymatgen.core import Structure
sys.path.insert(0, os.path.dirname(__file__))
from crystal import classify_perovskite, tolerance_factor, charge_balance, dedup, ANIONS, make_matcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

RAW, OUT = "data/raw", "data/curated"
os.makedirs(OUT, exist_ok=True)
rep = {}

def load(p):
    L = [json.loads(l) for l in open(p)]
    return L[0], L[1:]

meta_a, A = load(f"{RAW}/mp_abc3.jsonl"); meta_b, B = load(f"{RAW}/mp_broad.jsonl")
rep["raw"] = dict(abc3=len(A), broad=len(B), mp_db_version=meta_a["db_version"], retrieved=meta_a["retrieved"])
seen, recs = set(), []
for r in A + B:
    if r["material_id"] in seen: continue
    seen.add(r["material_id"]); recs.append(r)
rows, structs = [], []
drop = collections.Counter()
for r in recs:
    s = Structure.from_dict(r["structure"])
    if not s.is_ordered: drop["disordered"] += 1; continue
    if r["band_gap"] is None or r["energy_above_hull"] is None: drop["missing_label"] += 1; continue
    els = {e.symbol for e in s.composition}
    if not (els & ANIONS): drop["no_anion"] += 1; continue
    if len(s) > 40: drop["gt40_sites"] += 1; continue
    c = classify_perovskite(s)
    rows.append(dict(id=r["material_id"], formula=s.composition.reduced_formula, nsites=len(s),
        spacegroup=r["symmetry"]["symbol"], sg_number=r["symmetry"]["number"], band_gap=r["band_gap"],
        ehull=r["energy_above_hull"], eform=r["formation_energy_per_atom"], theoretical=r["theoretical"],
        abx3=c["abx3"], family=c["family"], A=",".join(c["A_sites"] or []), Bsite=",".join(c["B_sites"] or []),
        X=",".join(sorted(els & ANIONS)), elements=",".join(sorted(els)),
        charge_balanced=charge_balance(s.composition.reduced_composition)[0], tolerance=tolerance_factor(s, c)))
    structs.append(s)
df = pd.DataFrame(rows); rep["dropped_before_dedup"] = dict(drop)
                                                                                
key = df.formula + "|" + df.sg_number.astype(str)
keep = np.ones(len(df), bool); m = make_matcher()
for k, idx in df.groupby(key).indices.items():
    idx = sorted(idx, key=lambda i: df.ehull.iloc[i]); reps = []
    for i in idx:
        if any(m.fit(structs[j], structs[i]) for j in reps): keep[i] = False
        else: reps.append(i)
rep["structural_duplicates_removed"] = int((~keep).sum())
df = df[keep].reset_index(drop=True); structs = [s for s, k in zip(structs, keep) if k]
df["is_perovskite"] = df.family == "3D corner-sharing perovskite"
                                                             
rng = np.random.RandomState(0)
groups = df.formula.unique(); rng.shuffle(groups)
n = len(groups); g_te = set(groups[: n // 10]); g_va = set(groups[n // 10: n // 5])
df["split_random"] = np.where(df.formula.isin(g_te), "test", np.where(df.formula.isin(g_va), "val", "train"))
                                                                                                                  
per = df[df.is_perovskite]
cnt = collections.Counter(b for x in per.Bsite for b in x.split(",") if b)
cand = [e for e, c in cnt.items() if 15 <= c <= 90]; rng.shuffle(cand)
held, tot = [], 0
for e in cand:
    if tot >= 0.12 * len(per): break
    held.append(e); tot += cnt[e]
H = set(held)
has_H = df.elements.apply(lambda x: bool(H & set(x.split(","))))
is_testB = df.is_perovskite & df.Bsite.apply(lambda x: bool(x) and set(x.split(",")) <= H)
df["split_Bdisjoint"] = np.where(is_testB, "test", np.where(has_H, "excluded", np.where(df.split_random == "val", "val", "train")))
rep["held_out_B_elements"] = sorted(H)
rep["counts"] = dict(total=len(df), abx3=int(df.abx3.sum()), perovskite=int(df.is_perovskite.sum()),
                     families=df[df.abx3].family.value_counts().to_dict(),
                     split_random=df.split_random.value_counts().to_dict(),
                     split_Bdisjoint=df.split_Bdisjoint.value_counts().to_dict(),
                     perovskite_test_random=int(((df.split_random == "test") & df.is_perovskite).sum()),
                     perovskite_test_B=int(is_testB.sum()), gap_zero_frac=float((df.band_gap == 0).mean()))
df.to_csv(f"{OUT}/labels.csv", index=False)
with gzip.open(f"{OUT}/structures.jsonl.gz", "wt") as f:
    for i, s in zip(df.id, structs): f.write(json.dumps({"id": i, "structure": s.as_dict()}) + "\n")
json.dump(rep, open(f"{OUT}/curation_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
