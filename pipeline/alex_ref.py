"""Extract ABX3 / perovskite subset of the Alex-MP-20 generator training set (MatterGen release) for novelty tests."""
import gzip, json, sys, os
import pandas as pd
from pymatgen.core import Structure, Composition
sys.path.insert(0, os.path.dirname(__file__))
from crystal import classify_perovskite, abx3_stoichiometry

base = "data/external/alex/alex_mp_20"; rows = []; allf = set()
out = gzip.open("data/curated/alex_perovskites.jsonl.gz", "wt")
for part in ("train", "val"):
    for ch in pd.read_csv(f"{base}/{part}.csv", chunksize=20000, usecols=["material_id", "reduced_formula", "space_group", "num_sites", "cif", "energy_above_hull", "dft_band_gap"]):
        for r in ch.itertuples():
            allf.add(r.reduced_formula)
            try:
                comp = Composition(r.reduced_formula)
            except Exception:
                continue
            if len(comp) < 3 or r.num_sites > 40: continue
            try:
                s = Structure.from_str(r.cif, fmt="cif")
            except Exception:
                continue
            if not abx3_stoichiometry(s)[0]: continue
            c = classify_perovskite(s)
            rows.append(dict(id=r.material_id, formula=r.reduced_formula, split=part, sg=r.space_group, family=c["family"],
                             A=",".join(c["A_sites"] or []), Bsite=",".join(c["B_sites"] or []), gap=r.dft_band_gap, ehull=r.energy_above_hull))
            if c["family"] == "3D corner-sharing perovskite":
                out.write(json.dumps({"id": r.material_id, "structure": s.as_dict()}) + "\n")
        print(part, len(rows), flush=True)
out.close()
pd.DataFrame(rows).to_csv("data/curated/alex_abx3_labels.csv", index=False)
json.dump(sorted(x for x in allf if isinstance(x, str)), open("data/curated/alex_all_formulas.json", "w"))
print("done", len(rows), len(allf))
