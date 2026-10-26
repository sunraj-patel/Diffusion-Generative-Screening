"""Download raw Materials Project summary docs (with structures) to data/raw. Needs MP_API_KEY in env."""
import os, sys, json, datetime
from mp_api.client import MPRester

FIELDS = ["material_id", "formula_pretty", "formula_anonymous", "nsites", "structure", "band_gap", "is_gap_direct",
          "energy_above_hull", "formation_energy_per_atom", "symmetry", "theoretical", "is_stable",
          "database_IDs", "volume", "density"]
OUT = sys.argv[1]
kw = dict(fields=FIELDS)
if sys.argv[2] == "abc3":                                          
    kw.update(formula="ABC3")
else:                                                                         
    kw.update(num_sites=(1, 20), energy_above_hull=(0, 0.15))
with MPRester(os.environ["MP_API_KEY"]) as m:
    docs = m.materials.summary.search(**kw)
    meta = {"retrieved": datetime.date.today().isoformat(), "db_version": m.get_database_version(), "n": len(docs), "query": sys.argv[2]}
with open(OUT, "w") as f:
    f.write(json.dumps(meta) + "\n")
    for d in docs:
        r = d.model_dump(); r["structure"] = d.structure.as_dict()
        f.write(json.dumps(r, default=str) + "\n")
print(meta)
