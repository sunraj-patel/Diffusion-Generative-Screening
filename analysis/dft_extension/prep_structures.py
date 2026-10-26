"""Stage A: validation + symmetry standardisation of the selected candidates. Run from the repo root.
Original structures are never modified: originals are copied to dft_ext/structures/<ID>_original.cif; the standardised primitive cell (spglib, symprec 0.05 A,
Cartesian-aligned standard setting so QE can detect the full point group) is written to <ID>_standardized.{cif,json}. Displacement from the original is recorded.
Writes reports/cms_dft_extension/stageA_structure_validation.csv"""
import json, gzip, os, numpy as np, pandas as pd, spglib, warnings
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
import sys; sys.path.insert(0, "pipeline")
from crystal import classify_perovskite
warnings.filterwarnings("ignore")
SEL = {"PRV-11": "candidates/structures/PRV-11.cif", "PRV-09": "candidates/structures/PRV-09.cif", "PRV-07": "candidates/structures/PRV-07.cif",
       "PRV-06": "candidates/structures/PRV-06.cif", "PRV-14": "candidates/structures/PRV-14.cif", "CMB-15": None}
def pool_struct(i):
    run = i.rsplit("_gen_", 1)[0]
    for l in gzip.open(f"work/gen/{run}/pool.jsonl.gz", "rt"):
        r = json.loads(l)
        if r["id"] == i: return Structure.from_dict(r["structure"])
out = "dft_ext/structures"; rows = []
for cid, path in SEL.items():
    s = pool_struct("base_s1_gen_662") if cid == "CMB-15" else Structure.from_file(path)
    s.to(filename=f"{out}/{cid}_original.cif")
    sym = SpacegroupAnalyzer(s, symprec=0.05)
    cell = (s.lattice.matrix, s.frac_coords, [site.specie.Z for site in s])
    lat, pos, nums = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=0.05)
    p = Structure(Lattice(lat), nums, pos)
    ds = StructureMatcher(ltol=0.1, stol=0.1, angle_tol=2, primitive_cell=False, scale=False).get_rms_dist(s, p)
    norm = (s.volume / len(s)) ** (1 / 3)
    p.to(filename=f"{out}/{cid}_standardized.cif"); json.dump(p.as_dict(), open(f"{out}/{cid}_standardized.json", "w"))
    dm = p.distance_matrix + np.eye(len(p)) * 99
    c0, c1 = classify_perovskite(s), classify_perovskite(p)
    rows.append(dict(id=cid, formula=s.composition.reduced_formula, natoms_original=len(s), natoms_standardized=len(p), spacegroup_symprec0p05=SpacegroupAnalyzer(p, 0.05).get_space_group_symbol(),
        volume_original=s.volume, volume_standardized=p.volume, rms_disp_A=ds[0] * norm if ds else None, max_disp_A=ds[1] * norm if ds else None,
        min_dist_A=dm.min(), family_original=c0["family"], family_standardized=c1["family"], B_sites=",".join(c1["B_sites"] or []), A_sites=",".join(c1["A_sites"] or [])))
d = pd.DataFrame(rows); d.to_csv("reports/cms_dft_extension/stageA_structure_validation.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30); print(d)
