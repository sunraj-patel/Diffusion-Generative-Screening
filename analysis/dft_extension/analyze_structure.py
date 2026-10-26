"""Stage C: compares an initial (standardised) and a relaxed structure. Usage: analyze_structure.py <ID> <initial.json> <relaxed.json> -> prints/returns a dict.
Retention classes (fixed in selection_rationale.md): retained / distorted-retained / transformed, from pipeline/crystal.classify_perovskite + B-X-B angles + octahedral bond spread."""
import sys, json, numpy as np
sys.path.insert(0, "pipeline")
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from crystal import classify_perovskite, ANIONS

def octa(s):
    """B sites = the nX/3 cations with 6 anion neighbours (within 1.25 x shortest) and shortest cation-anion distance (same rule as pipeline/crystal.py), so a species
    occupying both A and B sites is handled correctly. Returns classification, B-X distances, per-octahedron max/min bond ratio, B-X-B angles, A-site (shortest A-X, CN within 1.3x)."""
    c = classify_perovskite(s); nb = s.get_all_neighbors(4.6)
    xm = np.array([x.specie.symbol in ANIONS for x in s]); nX = xm.sum()
    cn, mind = {}, {}
    for i in np.where(~xm)[0]:
        d = sorted(n.nn_distance for n in nb[i] if xm[n.index])
        cn[i] = sum(1 for x in d if x <= 1.25 * d[0]) if d else 0; mind[i] = d[0] if d else 9
    B = sorted([i for i in cn if cn[i] == 6], key=lambda i: mind[i])[: nX // 3]
    spread, dists, bonds = [], [], {}
    for i in B:
        xs = sorted([(n.nn_distance, n.index, n.coords) for n in nb[i] if xm[n.index]], key=lambda x: x[0])[:6]
        d = [x[0] for x in xs]; dists += d; spread.append(max(d) / min(d))
        for dd, j, xc in xs: bonds.setdefault(j, []).append(s[i].coords - xc)
    ang = []
    for j, v in bonds.items():
        if len(v) == 2: ang.append(np.degrees(np.arccos(np.clip(np.dot(v[0], v[1]) / np.linalg.norm(v[0]) / np.linalg.norm(v[1]), -1, 1))))
    A = [i for i in cn if i not in B]; ad = []
    for i in A:
        d = sorted(n.nn_distance for n in nb[i] if xm[n.index]); ad.append((d[0], sum(1 for x in d if x <= 1.3 * d[0])))
    c["A_shortest_AX"] = float(min(a[0] for a in ad)) if ad else None; c["A_CN_1p3"] = float(np.mean([a[1] for a in ad])) if ad else None
    return c, dists, spread, ang

def analyze(cid, s0, s1):
    c0, d0, sp0, a0 = octa(s0); c1, d1, sp1, a1 = octa(s1)
    sm = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=10, primitive_cell=False, scale=True)
    rms = sm.get_rms_dist(s0, s1)
    r = dict(id=cid, formula=s1.composition.reduced_formula, natoms=len(s1), dV_percent=100 * (s1.volume / s0.volume - 1), V0=s0.volume, V1=s1.volume,
             lattice_abc_0=[round(x, 4) for x in s0.lattice.abc], lattice_abc_1=[round(x, 4) for x in s1.lattice.abc], angles_1=[round(x, 3) for x in s1.lattice.angles],
             sg_0=SpacegroupAnalyzer(s0, 0.05).get_space_group_symbol(), sg_1_0p01=SpacegroupAnalyzer(s1, 0.01).get_space_group_symbol(), sg_1_0p1=SpacegroupAnalyzer(s1, 0.1).get_space_group_symbol(),
             family_0=c0["family"], family_1=c1["family"], x_bridging_1=c1["x_bridging"],
             BX_mean_0=float(np.mean(d0)), BX_mean_1=float(np.mean(d1)), BX_min_1=float(min(d1)), BX_max_1=float(max(d1)),
             octa_spread_max_1=float(max(sp1)), A_shortest_AX_0=c0['A_shortest_AX'], A_shortest_AX_1=c1['A_shortest_AX'], A_CN_0=c0['A_CN_1p3'], A_CN_1=c1['A_CN_1p3'], BXB_min_deg=float(min(a1)) if a1 else None, BXB_mean_deg=float(np.mean(a1)) if a1 else None,
             BXB_dev_from_180=float(180 - min(a1)) if a1 else None, rms_disp_A=(rms[0] * (s0.volume / len(s0)) ** (1 / 3)) if rms and rms[0] is not None else None)
    ok = r["family_1"] == "3D corner-sharing perovskite"
    r["retention"] = "transformed" if not ok else ("distorted-retained" if (r["BXB_dev_from_180"] or 0) > 10 or r["octa_spread_max_1"] > 1.10 else "retained")
    return r

if __name__ == "__main__":
    s0 = Structure.from_dict(json.load(open(sys.argv[2]))); s1 = Structure.from_dict(json.load(open(sys.argv[3])))
    print(json.dumps(analyze(sys.argv[1], s0, s1), indent=1))
