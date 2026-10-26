"""Reads finished vc-relax outputs in dft_ext/results/<ID>/relax/vcrelax.out, writes dft_ext/structures/<ID>_relaxed.{json,cif}, runs the Stage C analysis
and appends to reports/cms_dft_extension/structural_retention.csv. Usage: collect_relaxed.py ID [ID...]  (run from repo root)"""
import sys, json, numpy as np, pandas as pd, os
sys.path.insert(0, "dft/scripts"); sys.path.insert(0, "reports/cms_dft_extension/scripts")
from qe_parse import parse
from analyze_structure import analyze
from pymatgen.core import Structure, Lattice
rows = []
for cid in sys.argv[1:]:
    r = parse(f"dft_ext/results/{cid}/relax/vcrelax.out")
    if not (r["job_done"] and r["bfgs_converged"]): print(cid, "NOT CONVERGED: job_done", r["job_done"], "bfgs", r["bfgs_converged"]); continue
    s1 = Structure(Lattice(np.array(r["final_lattice_A"])), [x[0] for x in r["final_positions_crystal"]], [[float(v) for v in x[1:4]] for x in r["final_positions_crystal"]])
    json.dump(s1.as_dict(), open(f"dft_ext/structures/{cid}_relaxed.json", "w")); s1.to(filename=f"dft_ext/structures/{cid}_relaxed.cif")
    s0 = Structure.from_dict(json.load(open(f"dft_ext/structures/{cid}_standardized.json")))
    a = analyze(cid, s0, s1); a.update(E_Ry=r["total_energy_Ry"], force_Ry_bohr=r["total_force_Ry_bohr"], pressure_kbar=r["pressure_kbar"], wall_s=r["wall_time_s"], n_bfgs=r["n_bfgs_steps"], sym_ops_qe=r["sym_ops"])
    rows.append(a); print(cid, a["retention"], "dV=%.2f%%" % a["dV_percent"], a["sg_1_0p01"], a["sg_1_0p1"])
if rows:
    p = "reports/cms_dft_extension/structural_retention.csv"
    d = pd.DataFrame(rows).astype({"lattice_abc_0": str, "lattice_abc_1": str, "angles_1": str})
    if os.path.exists(p): d = pd.concat([pd.read_csv(p), d]).drop_duplicates("id", keep="last")
    d.to_csv(p, index=False)
