"""Stage B: structural/chemical screening of a structure pool, with a per-candidate record of every filter outcome.
Pool format: JSONL.gz of {"id","source","structure"(pymatgen dict)} -> screened CSV + funnel JSON + unique-survivor JSONL.gz."""
import gzip, json, sys, os
import numpy as np, pandas as pd
from pymatgen.core import Structure
sys.path.insert(0, os.path.dirname(__file__))
from crystal import (classify_perovskite, charge_balance, tolerance_factor, geometry_ok, dedup, make_matcher, ANIONS)

T_RANGE = (0.78, 1.10)                                                                                                           


def read_pool(path):
    out = []
    with gzip.open(path, "rt") as f:
        for l in f:
            d = json.loads(l); out.append(d)
    return out


def screen(pool, ref_structs=None, ref_ids=None):
    """pool: list of dicts (id, source, structure|None). ref_*: curated MP perovskite references for novelty.
    Returns (DataFrame with all filter flags, list of unique surviving Structures aligned with df rows 'survivor')."""
    rows, structs = [], {}
    for d in pool:
        r = dict(id=d["id"], source=d["source"], parsed=False)
        try:
            s = Structure.from_dict(d["structure"]) if d.get("structure") else None
            if s is not None and len(s) and np.isfinite(s.volume) and s.volume > 0:
                r["parsed"] = True
        except Exception:
            s = None
        if r["parsed"]:
            r["formula"] = s.composition.reduced_formula; r["nsites"] = len(s)
            r["geometry_ok"] = geometry_ok(s)
            c = classify_perovskite(s)
            r.update(abx3=c["abx3"], family=c["family"], A=",".join(c["A_sites"] or []), Bsite=",".join(c["B_sites"] or []),
                     X=",".join(sorted({x.symbol for x in s.species} & ANIONS)))
            r["perovskite"] = c["family"] == "3D corner-sharing perovskite"
            r["charge_balanced"] = charge_balance(s.composition.reduced_composition)[0] if c["abx3"] else None
            t = tolerance_factor(s, c) if r["perovskite"] else None
            r["tolerance"] = t
            r["tolerance_ok"] = None if t is None else bool(T_RANGE[0] <= t <= T_RANGE[1])
            structs[d["id"]] = s
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("geometry_ok", "abx3", "perovskite"):
        df[c] = df.get(c, False).fillna(False).astype(bool)
    stage = df.parsed & df.geometry_ok & df.abx3 & df.perovskite
    stage &= df.charge_balanced.fillna(False).astype(bool)
    stage &= df.tolerance_ok.map(lambda v: v is not False if v is not None and v == v else True)
    df["pass_chem"] = stage
                                                    
    ids = df.id[stage].tolist(); ss = [structs[i] for i in ids]
    gid, reps = dedup(ss) if ss else ([], [])
    df["dup_group"] = -1; df["unique"] = False
    df.loc[stage, "dup_group"] = gid
    for k in reps: df.loc[df.id == ids[k], "unique"] = True
                                                  
    df["in_ref_formula"] = False; df["in_ref_structure"] = False
    if ref_structs is not None:
        by = {}
        for s in ref_structs: by.setdefault(s.composition.reduced_formula, []).append(s)
        m = make_matcher()
        for i, r in df[df.unique].iterrows():
            cands = by.get(r.formula, [])
            df.at[i, "in_ref_formula"] = bool(cands)
            df.at[i, "in_ref_structure"] = any(m.fit(structs[r.id], c) for c in cands)
    return df, {i: structs[i] for i in df.id[df.unique]}


def funnel(df):
    n = len(df)
    steps = [("generated", df.id.notna()), ("parsed", df.parsed), ("geometry", df.parsed & df.geometry_ok),
             ("ABX3 stoichiometry", df.parsed & df.geometry_ok & df.abx3),
             ("3D perovskite", df.parsed & df.geometry_ok & df.abx3 & df.perovskite)]
    cum = df.parsed & df.geometry_ok & df.abx3 & df.perovskite
    cb = cum & df.charge_balanced.fillna(False).astype(bool); steps.append(("charge balanced", cb))
    steps.append(("tolerance window / unassessable", df.pass_chem)); steps.append(("unique", df.unique))
    out, prev = [], n
    for name, m in steps:
        k = int(m.sum()); out.append(dict(stage=name, n=k, retained_prev=k / prev if prev else 0, retained_total=k / n if n else 0)); prev = k
    return out


def load_refs():
    """Curated MP perovskite references (all splits) used for benchmark-novelty checks."""
    lab = pd.read_csv("data/curated/labels.csv"); per = set(lab.id[lab.is_perovskite])
    out = []
    with gzip.open("data/curated/structures.jsonl.gz", "rt") as f:
        for l in f:
            d = json.loads(l)
            if d["id"] in per: out.append(Structure.from_dict(d["structure"]))
    return out


if __name__ == "__main__":
    pool_path, out_prefix = sys.argv[1], sys.argv[2]
    df, uniq = screen(read_pool(pool_path), load_refs())
    df.to_csv(out_prefix + "_screen.csv", index=False)
    json.dump(funnel(df), open(out_prefix + "_funnel.json", "w"), indent=1)
    with gzip.open(out_prefix + "_unique.jsonl.gz", "wt") as f:
        for i, s in uniq.items(): f.write(json.dumps({"id": i, "structure": s.as_dict()}) + "\n")
    print(pd.DataFrame(funnel(df)))
