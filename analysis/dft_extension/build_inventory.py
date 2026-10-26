"""Machine-readable inventory of every MatterGen-generated perovskite-class candidate in the existing pools.
Run from the repo root: python3 reports/cms_dft_extension/scripts/build_inventory.py
Reads work/cand/{gap14,gap14g4,base}_pool.csv (screening + ALIGNN predictions) and work/gen/<run>/pool.jsonl.gz (structures).
Writes reports/cms_dft_extension/candidate_inventory.csv. No DFT result is read here."""
import gzip, json, glob, os, warnings
import pandas as pd, numpy as np
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
warnings.filterwarnings("ignore")
OUT = "reports/cms_dft_extension"
pool = pd.concat([pd.read_csv(f"work/cand/{n}_pool.csv").assign(pool=n) for n in ("gap14", "gap14g4", "base")])
pool = pool[(pool.perovskite == True) & pool.source.str.contains("mattergen")].drop_duplicates("id")
rank = pd.read_csv("candidates/ranked_table.csv").set_index("id")
structs = {}
for f in glob.glob("work/gen/*/pool.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        if r["id"] in set(pool.id): structs[r["id"]] = r["structure"]
TOXIC = {"Tl", "As", "Pb", "Cd", "Hg", "Cr", "Se", "Te", "Sb", "Be"}
rows = []
for _, p in pool.iterrows():
    s = Structure.from_dict(structs[p.id]); els = sorted({e.symbol for e in s.composition})
    sga = lambda sp: SpacegroupAnalyzer(s, symprec=sp).get_space_group_symbol()
    prim = SpacegroupAnalyzer(s, symprec=0.1).find_primitive() or s
    dm = s.distance_matrix + np.eye(len(s)) * 99
    ox = {}
    try: ox = s.composition.oxi_state_guesses(max_sites=-1)[0]
    except Exception: pass
                                                                                     
    def dn(e, q):
        el = Element(e); 
        if el.block == "f": return "f"
        if el.block != "d": return 0
        g = el.group; return max(0, g - q) if g <= 12 else 0
    open_shell = None
    if ox:
        v = {e: dn(e, q) for e, q in ox.items() if Element(e).block in ("d", "f")}
        open_shell = any(x == "f" or (isinstance(x, int) and 0 < x < 10) for x in v.values())
    f_el = any(Element(e).block == "f" for e in els)
    rows.append(dict(id=p.id, ranked_id=rank.loc[p.id, "candidate"] if p.id in rank.index else "", formula=p.formula, natoms=len(s), primitive_natoms=len(prim),
        spacegroup_0p01=sga(0.01), spacegroup_0p1=sga(0.1), a=s.lattice.a, b=s.lattice.b, c=s.lattice.c, alpha=s.lattice.alpha, beta=s.lattice.beta, gamma=s.lattice.gamma,
        volume=s.volume, cell_type="primitive" if len(prim) == len(s) else "reducible (supercell/conventional)", min_dist_A=dm.min(),
        alignn_gap_eV=p.gap, alignn_gap_std=p.gap_std, alignn_ehull_eV_atom=p.ehull, alignn_ehull_std=p.ehull_std, tolerance=p.tolerance, charge_balanced=p.charge_balanced,
        family=p.family, novelty=p.novelty, struct_in_AlexMP20=p.struct_in_AlexMP20, formula_in_MP_perovskites=p.formula_in_MP_perovskites,
        oxidation_guess=json.dumps({k: float(v) for k, v in ox.items()}), open_shell_proxy=open_shell, has_4f=f_el, toxic_elements=",".join(sorted(set(els) & TOXIC)),
        electrons_proxy_cost=len(s) ** 3, pool=p.pool))
d = pd.DataFrame(rows).sort_values(["natoms", "id"], ascending=[False, True])
d.to_csv(f"{OUT}/candidate_inventory.csv", index=False)
print(d.shape); print(d.natoms.value_counts().sort_index().to_dict())
pd.set_option("display.width", 250, "display.max_columns", 30)
print(d[d.natoms >= 15][["id", "ranked_id", "formula", "natoms", "primitive_natoms", "spacegroup_0p01", "spacegroup_0p1", "alignn_gap_eV", "open_shell_proxy", "has_4f", "toxic_elements"]])
