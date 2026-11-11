"""Decomposition-energy (competing binary phases) QE inputs. Run from the project root.
Same PBE/PAW settings as dft_ext (ecut 80/640 Ry, k-spacing 0.25 1/A, gaussian 0.002 Ry, spin-unpolarised); vc-relax for every phase
(perovskite re-relaxed too, so all energies come from one code version and one protocol). Binaries: lowest-E_hull MP polymorph from the local snapshot,
plus both TiO2 polymorphs (anatase is MP's lowest, rutile is the experimental ground state)."""
import sys, json, os
sys.path.insert(0, "reports/cms_dft_extension/scripts"); sys.path.insert(0, "dft/scripts")
import qe_inputs as q
import make_inputs as mi
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
OUT = "dft_ext2/run"; PD = "./pseudo"
want = {"SrO": None, "TiO2": "rutile", "CsBr": None, "NaBr": None, "RhBr3": None, "MgBr2": None, "CdBr2": None}
best = {}
for fn in ("data/raw/mp_broad.jsonl", "data/raw/mp_abc3.jsonl"):
    with open(fn) as f:
        f.readline()
        for l in f:
            d = json.loads(l); fm = d["formula_pretty"]
            if fm in want and d.get("structure"):
                sg = d["symmetry"]["symbol"] if d.get("symmetry") else ""
                key = fm + ("_rutile" if fm == "TiO2" and sg == "P4_2/mnm" else "")
                if fm == "TiO2" and key == "TiO2" and sg != "I4_1/amd": continue
                if key not in best or d["energy_above_hull"] < best[key][0]: best[key] = (d["energy_above_hull"], d["material_id"], d["structure"], sg)
meta = {}
def emit(name, s, tag):
    s = SpacegroupAnalyzer(s, 0.05).get_primitive_standard_structure()
    k = q.kmesh(s, 0.25); t = q.txt(s, "vc-relax", name, (80, 640), k).replace(q.PSEUDO_DIR, PD)
    mi.write(f"{OUT}/{name}/vcrelax.in", t); meta[name] = dict(formula=s.composition.reduced_formula, natoms=len(s), source=tag, kmesh=k)
for k, (eh, mid, sd, sg) in best.items(): emit(k, Structure.from_dict(sd), f"MP {mid} {sg} ehull={eh:.4f}")
for cid in ("PRV-11", "PRV-07", "CMB-15"): emit(cid, q.load(cid), "dft_ext standardized (generated)" if cid != "PRV-11" else "dft_ext standardized (generated, control)")
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1); print(json.dumps(meta, indent=1))
