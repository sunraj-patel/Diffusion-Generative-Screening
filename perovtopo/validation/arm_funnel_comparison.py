"""Formula-only vs topology-aware yield for each sampling arm, from the pipeline funnel files (work/<arm>_funnel.json).
Usage: python arm_funnel_comparison.py <work_dir> <out.csv>"""
import json, sys, csv
work, out = sys.argv[1], sys.argv[2]
ARMS = [("base", "Unconditional"), ("gap14", "Conditioned, gamma=2"), ("gap14g4", "Conditioned, gamma=4"), ("enum", "Prototype substitution")]
rows = []
for a, lab in ARMS:
    f = {x["stage"]: x["n"] for x in json.load(open(f"{work}/{a}_funnel.json"))}
    n, s, t, u = f["generated"], f["ABX3 stoichiometry"], f["3D perovskite"], f["unique"]
    rows.append(dict(arm=lab, n_generated=n, stoichiometry=s, topology=t, unique=u,
                     stoich_per_1000=round(1000*s/n, 1), topology_per_1000=round(1000*t/n, 1), unique_per_1000=round(1000*u/n, 1),
                     stoich_over_topology=round(s/t, 2)))
base = {r["arm"]: r for r in rows}
ref = base["Conditioned, gamma=2"]
for r in rows:
    r["rel_to_gamma2_stoich"] = round(r["stoich_per_1000"]/ref["stoich_per_1000"], 2)
    r["rel_to_gamma2_topology"] = round(r["topology_per_1000"]/ref["topology_per_1000"], 2)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
