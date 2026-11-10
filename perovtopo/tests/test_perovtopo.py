import json, os
import numpy as np
from pymatgen.core import Structure, Lattice
from perovtopo import classify, Params, PEROVSKITE, funnel, b_site_disjoint_split, vs_constant_baseline
from perovtopo.classifier import FACE_EDGE

HERE = os.path.dirname(__file__)

def cubic(a, A="Sr", B="Ti", X="O"):
    return Structure(Lattice.cubic(a), [A, B, X, X, X], [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])

def double_perovskite():
    s = cubic(3.9, "Ba", "Fe", "O"); s.make_supercell([2, 2, 2])
    fe = [i for i, x in enumerate(s) if x.specie.symbol == "Fe"]
    for i in fe:
        f = s[i].frac_coords * 2
        if int(round(f[0] - .5) + round(f[1] - .5) + round(f[2] - .5)) % 2:
            s.replace(i, "Mo")
    return s

def test_defaults_match_manuscript():
    p = Params()
    assert (p.b_tol, p.b_cn, p.a_tol, p.a_min_cn, p.bx_tol, p.strict_corner) == (1.25, 6, 1.3, 8, 1.25, False)

def test_cubic_perovskite():
    r = classify(cubic(3.905))
    assert r["family"] == PEROVSKITE and r["B_sites"] == ["Ti"] and r["A_sites"] == ["Sr"] and r["dimensionality"] == 3

def test_double_perovskite_has_two_b_elements():
    r = classify(double_perovskite())
    assert r["family"] == PEROVSKITE and r["B_sites"] == ["Fe", "Mo"]

def test_not_abx3():
    nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
    assert classify(nacl)["family"] == "not ABX3"

def test_small_a_site_rejected():
    s = cubic(3.905)
    # an A cation displaced to a low-coordination position cannot satisfy the A-site rule
    s.translate_sites([0], [0.2, 0.2, 0.2], frac_coords=True)
    assert classify(s)["family"] != PEROVSKITE

def test_threshold_is_parameter():
    s = cubic(3.905)
    assert classify(s, Params(a_min_cn=13))["family"] != PEROVSKITE  # no site has 13 neighbours
    assert classify(s, Params(a_min_cn=12))["family"] == PEROVSKITE

def test_face_sharing_polytype_flagged_only_when_strict():
    d = json.load(open(os.path.join(HERE, "data", "BaMnO3_polytype.json")))
    s = Structure.from_dict(d["structure"])
    assert classify(s)["family"] == PEROVSKITE                         # manuscript classifier accepts it
    assert classify(s, Params(strict_corner=True))["family"] == FACE_EDGE
    assert classify(cubic(3.905), Params(strict_corner=True))["family"] == PEROVSKITE

def test_funnel_counts():
    nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
    summ, _ = funnel([cubic(3.9), double_perovskite(), nacl])
    assert (summ["n"], summ["stoichiometry"], summ["topology"]) == (3, 2, 2) and summ["mixed_b_fraction"] == 0.5

def test_b_site_disjoint_split_has_no_overlap():
    rng = np.random.default_rng(1); els = ["Ti", "Zr", "Nb", "Fe", "Mn", "Sn", "Hf", "Ta", "V", "Cr"]
    b = [{els[i]} if rng.random() < .8 else {els[i], els[(i + 1) % 10]} for i in rng.integers(0, 10, 300)]
    tr, te, dr = b_site_disjoint_split(b, 0.2, seed=3)
    assert set().union(*[b[i] for i in tr]).isdisjoint(set().union(*[b[i] for i in te]))
    assert len(tr) + len(te) + len(dr) == 300 and len(te) > 0

def test_constant_baseline_flags_uninformative_model():
    rng = np.random.default_rng(0); y = rng.normal(2, 1, 200); r = vs_constant_baseline(y[:100], y[100:], rng.normal(2, 1, 100))
    assert not r["beats_constant"]
    assert vs_constant_baseline(y[:100], y[100:], y[100:] + .01)["beats_constant"]
