"""Compare the connectivity classifier with two simple baselines on the curated MP set and on a generated pool.
 (a) stoichiometry only: ordered, >=2 cations, anion:cation = 3:2 (the formula-level screen);
 (b) prototype matching: pymatgen StructureMatcher (anonymous, loose tolerances) against five perovskite prototypes taken from the curated set
     (cubic SrTiO3 mp-htd, Pnma CaTiO3 mp-fyp, R-3c LaAlO3 mp-eii, I4/mcm SrTiO3 mp-gwx, Fm-3m Cs2AgBiBr6 mp-cjjbe); the prototypes are excluded from scoring.
Usage: python baseline_comparison.py [POOL.jsonl.gz]   -> baseline_comparison.csv, baseline_disagreements.csv"""
import sys, json, gzip, csv, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
sys.path.insert(0, __file__.rsplit("/validation", 1)[0])
from perovtopo import Params, classify, PEROVSKITE, abx3_stoichiometry
HERE = __file__.rsplit("/", 1)[0]
PROTO = ["mp-htd", "mp-fyp", "mp-eii", "mp-gwx", "mp-cjjbe"]
meta = {r["mp_id"]: r for r in csv.DictReader(open(f"{HERE}/curated_set.csv"))}
structs = {}
for l in gzip.open(f"{HERE}/structures.jsonl.gz", "rt"):
    d = json.loads(l); structs[d["mp_id"]] = Structure.from_dict(d["structure"])
protos = [structs[m] for m in PROTO]
SM = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True, attempt_supercell=True)

def three(s):
    r = classify(s); d = classify(s, Params(strict_corner=True))
    stoich = bool(r["abx3"])
    proto = False
    if stoich:
        try: proto = any(SM.fit_anonymous(s, p) for p in protos)
        except Exception: proto = False
    return stoich, proto, r["family"] == PEROVSKITE, d["family"] == PEROVSKITE

def work(a): return three(a)

def conf(rows, col, truth):
    tp = sum(1 for r in rows if truth(r) and r[col]); fn = sum(1 for r in rows if truth(r) and not r[col])
    fp = sum(1 for r in rows if (not truth(r)) and r[col]); tn = sum(1 for r in rows if (not truth(r)) and not r[col])
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=round(tp / max(tp + fn, 1), 3), specificity=round(tn / max(tn + fp, 1), 3))

if __name__ == "__main__":
    ids = [m for m in sorted(meta) if m not in PROTO]
    with Pool() as pool: res = pool.map(work, [structs[m] for m in ids], chunksize=10)
    rows = [dict(id=m, tier=meta[m]["tier"], label=meta[m]["label"], stoich=r[0], proto=r[1], topo=r[2], strict=r[3], formula=meta[m]["formula"], sg=meta[m]["spacegroup"]) for m, r in zip(ids, res)]
    out = []
    for tier in ("1", "2", "all"):
        sub = [r for r in rows if tier == "all" or r["tier"] == tier]
        for col, name in (("stoich", "stoichiometry only"), ("proto", "prototype matching"), ("topo", "connectivity (default)"), ("strict", "connectivity (strict)")):
            out.append(dict(set=f"tier {tier}", method=name, **conf(sub, col, lambda r: r["label"] == "perovskite")))
    if len(sys.argv) > 1:
        pool_s = [Structure.from_dict(json.loads(l)["structure"]) for l in gzip.open(sys.argv[1], "rt") if json.loads(l).get("structure")]
        with Pool() as pool: gres = pool.map(work, pool_s, chunksize=10)
        out.append(dict(set=f"generated pool (n={len(gres)})", method="stoichiometry only", tp=sum(g[0] for g in gres)))
        out.append(dict(set=f"generated pool (n={len(gres)})", method="prototype matching", tp=sum(g[1] for g in gres)))
        out.append(dict(set=f"generated pool (n={len(gres)})", method="connectivity (default)", tp=sum(g[2] for g in gres)))
        out.append(dict(set=f"generated pool (n={len(gres)})", method="both prototype and connectivity", tp=sum(g[1] and g[2] for g in gres)))
        out.append(dict(set=f"generated pool (n={len(gres)})", method="connectivity only (no prototype match)", tp=sum(g[2] and not g[1] for g in gres)))
        out.append(dict(set=f"generated pool (n={len(gres)})", method="prototype only (no connectivity)", tp=sum(g[1] and not g[2] for g in gres)))
    keys = sorted({k for o in out for k in o}, key=lambda k: ["set", "method", "tp", "fn", "fp", "tn", "recall", "specificity"].index(k) if k in ["set", "method", "tp", "fn", "fp", "tn", "recall", "specificity"] else 99)
    with open(f"{HERE}/baseline_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    with open(f"{HERE}/baseline_disagreements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "formula", "sg", "tier", "label", "stoich", "proto", "topo", "strict"]); w.writeheader()
        w.writerows([r for r in rows if r["proto"] != r["topo"]])
    for o in out: print(o)
