"""Build the labelled validation set for the topology classifier from a local Materials Project dump.

Labels are assigned from crystal-structure PROTOTYPES (space group + Wyckoff-site pattern, determined with spglib at
symprec 0.1) and from literature structure types of named compounds. The classifier under test is NOT used to label.

Tier 1: named compounds with a literature structure type (positives: ABX3 / A2BB'X6 perovskites; negatives: hexagonal
        polytypes, post-perovskite, ilmenite, LiNbO3-type, hexagonal manganites, NH4CdCl3-type).
Tier 2: prototype-defined, chemistry unscreened. Positives: perovskite prototypes (Pm-3m, GdFeO3/Pnma, LaAlO3/R-3c,
        ordered double Fm-3m). Negatives: post-perovskite, ilmenite, NH4CdCl3-type and compounds with an oxyanion former
        (P, As, B, C, N, S, Cl, Br, I) which cannot form a BX6 corner-sharing network.
Usage: python build_validation_set.py DATA_DIR OUT_DIR   (DATA_DIR contains mp_abc3.jsonl and mp_broad.jsonl)
"""
import sys, json, gzip, csv, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sys.path.insert(0, __file__.rsplit("/validation", 1)[0])
from perovtopo import abx3_stoichiometry
from perovtopo.classifier import ANIONS

DATA, OUT = sys.argv[1], sys.argv[2]
OXY = {"P", "As", "B", "C", "N", "S", "Cl", "Br", "I"}
POS_SG = {221, 62, 167, 225, 99, 38, 140, 127, 123, 160, 161, 87, 139, 12, 14}

TIER1_POS = """SrTiO3 CaTiO3 BaTiO3 PbTiO3 SrZrO3 CaZrO3 BaZrO3 LaAlO3 LaFeO3 LaMnO3 LaCoO3 GdFeO3 NdGaO3 SrRuO3 BaSnO3
KMgF3 KZnF3 KNiF3 NaMgF3 CsCaCl3 CsPbCl3 CsPbBr3 CsPbI3 CsSnI3 CsSnBr3 CsGeI3 KNbO3 BiFeO3 CaMnO3 SrHfO3 BaHfO3 YAlO3
Cs2AgBiBr6 Cs2AgInCl6 Cs2NaBiCl6 Cs2LiYCl6 Cs2NaYCl6 Ba2FeMoO6 Sr2FeMoO6 Sr2MgWO6 Sr2CrReO6 Ba2YNbO6 Ba2NaOsO6 Sr2FeReO6
CsMgCl3 RbCaF3 KCaF3 SrSnO3 CaSnO3 NaNbO3 NaTaO3 KTaO3 EuTiO3 SrVO3 CaVO3""".split()
# (formula, space group, why negative)
TIER1_NEG = [("BaNiO3", 194, "hexagonal 2H, face-sharing octahedral chains"), ("BaMnO3", 194, "hexagonal 2H"),
             ("CsNiCl3", 194, "hexagonal, face-sharing chains"), ("CsMnCl3", 194, "hexagonal, face-sharing chains"),
             ("CsCoCl3", 194, "hexagonal"), ("CsFeCl3", 194, "hexagonal"), ("RbNiCl3", 194, "hexagonal"),
             ("CaIrO3", 63, "post-perovskite"), ("CaPtO3", 63, "post-perovskite"), ("MgSiO3", 63, "post-perovskite"),
             ("FeTiO3", 148, "ilmenite"), ("MnTiO3", 148, "ilmenite"), ("ZnTiO3", 148, "ilmenite"), ("MgSiO3", 148, "ilmenite (akimotoite)"),
             ("CoTiO3", 148, "ilmenite"), ("NiTiO3", 148, "ilmenite"),
             ("LiNbO3", 161, "LiNbO3-type, small A cation in octahedral site"), ("LiTaO3", 161, "LiNbO3-type"),
             ("YMnO3", 185, "hexagonal manganite, 5-coordinated Mn"), ("LuMnO3", 185, "hexagonal manganite"),
             ("HoMnO3", 185, "hexagonal manganite"), ("ErMnO3", 185, "hexagonal manganite"), ("LuFeO3", 185, "hexagonal ferrite")]
TIER1_NEG_BY_FORMULA_PATTERN = ["CsPbI3", "CsPbBr3", "CsSnI3", "CsCdBr3", "CsCdCl3", "CsCdI3"]  # NH4CdCl3-type / delta phases

def pattern(s, ss):
    items = []
    for grp in ss.equivalent_sites:
        sym = grp[0].specie.symbol
        items.append((ss.wyckoff_symbols[ss.equivalent_sites.index(grp)], "A" if sym in ANIONS else "C"))
    return tuple(sorted(items))

def proto(sg, n, pat):
    p = sorted(pat)
    if sg == 221 and n == 5 and p in ([("1a", "C"), ("1b", "C"), ("3c", "A")], [("1a", "C"), ("1b", "C"), ("3d", "A")]): return "+", "Pm-3m aristotype"
    if sg == 62 and n == 20:
        inv = [x for x in p if x[0] in ("4a", "4b")]
        rest = sorted(x for x in p if x[0] not in ("4a", "4b"))
        if len(inv) == 1 and inv[0][1] == "C" and rest == [("4c", "A"), ("4c", "C"), ("8d", "A")]: return "+", "GdFeO3-type (Pnma)"
        if p == [("4c", "A")] * 3 + [("4c", "C")] * 2: return "-", "NH4CdCl3-type (Pnma, 1D chains)"
    if sg == 167 and n == 10 and sorted(p) == [("2a", "C"), ("2b", "C"), ("6e", "A")]: return "+", "LaAlO3-type (R-3c)"
    if sg == 225 and n == 10 and sorted(p) == [("1a", "C"), ("1b", "C"), ("2c", "C"), ("6e", "A")]: return "+", "ordered double perovskite (Fm-3m)"
    if sg == 63 and n == 10 and sorted(p) == [("2a", "C"), ("2c", "A"), ("2c", "C"), ("4f", "A")]: return "-", "post-perovskite (Cmcm)"
    if sg == 148 and n == 10 and sorted(p) == [("2c", "C"), ("2c", "C"), ("6f", "A")]: return "-", "ilmenite-type (R-3)"
    return None, None

rows, seen = [], set()
for fn in ("mp_abc3.jsonl", "mp_broad.jsonl"):
    for k, line in enumerate(open(f"{DATA}/{fn}")):
        if k == 0: continue
        r = json.loads(line)
        if r["material_id"] in seen: continue
        seen.add(r["material_id"])
        s = Structure.from_dict(r["structure"])
        if not abx3_stoichiometry(s)[0]: continue
        try:
            ss = SpacegroupAnalyzer(s, symprec=0.1).get_symmetrized_structure()
        except Exception:
            continue
        pat = pattern(s, ss)
        lab, basis = proto(r["symmetry"]["number"], len(s), pat)
        cations = {sp.symbol for sp in s.species if sp.symbol not in ANIONS}
        anions = {sp.symbol for sp in s.species if sp.symbol in ANIONS}
        rows.append(dict(mp_id=r["material_id"], formula=r["formula_pretty"], sg=r["symmetry"]["number"], nsites=len(s),
                         ehull=r.get("energy_above_hull"), theoretical=r.get("theoretical"), proto_label=lab, proto_basis=basis, cations=cations, anions=anions,
                         structure=r["structure"], pattern=pat))
by_formula = defaultdict(list)
for x in rows: by_formula[x["formula"]].append(x)

sel = {}  # mp_id -> record
def add(x, tier, label, basis):
    if x["mp_id"] in sel and sel[x["mp_id"]]["tier"] == 1: return
    sel[x["mp_id"]] = dict(mp_id=x["mp_id"], formula=x["formula"], spacegroup=x["sg"], nsites=x["nsites"], ehull=x["ehull"], tier=tier,
                           label=label, basis=basis, structure=x["structure"])
# tier 1
for f in TIER1_POS:
    c = [x for x in by_formula.get(f, []) if x["sg"] in POS_SG and x["ehull"] is not None
         and (x["sg"] not in (62, 221, 167, 225) or x["proto_label"] == "+")]  # prototype must match where it is defined
    if c:
        x = min(c, key=lambda y: y["ehull"]); add(x, 1, "perovskite", f"literature perovskite {f}, sg {x['sg']}, lowest E_hull polymorph in perovskite space groups")
for f, sg, why in TIER1_NEG:
    c = [x for x in by_formula.get(f, []) if x["sg"] == sg and x["ehull"] is not None and (sg != 194 or x["nsites"] == 10)]
    if c:
        x = min(c, key=lambda y: y["ehull"]); add(x, 1, "non-perovskite", f"{why}")
for f in TIER1_NEG_BY_FORMULA_PATTERN:
    for x in by_formula.get(f, []):
        if x["proto_label"] == "-" and x["proto_basis"].startswith("NH4CdCl3"):
            add(x, 1, "non-perovskite", "NH4CdCl3-type / delta phase: 1D face/edge-sharing chains")
# tier 2
for x in rows:
    if x["sg"] == 194 and x["nsites"] >= 10 and x["anions"] <= {"O", "F", "Cl", "Br", "I"} and x["ehull"] is not None and x["ehull"] <= 0.1:
        add(x, 2, "non-perovskite", "P6_3/mmc ABX3: hexagonal-stacked polytype (face-sharing octahedra) or non-perovskite type")
        continue
    if x["proto_label"] == "+": add(x, 2, "perovskite", x["proto_basis"] + " (chemistry unscreened)")
    elif x["proto_label"] == "-": add(x, 2, "non-perovskite", x["proto_basis"])
    elif x["cations"] & OXY and x["anions"] == {"O"} and x["nsites"] <= 40 and x["theoretical"] is False and (x["ehull"] if x["ehull"] is not None else 9) <= 0.1:
        add(x, 2, "non-perovskite", "oxyanion former (" + ",".join(sorted(x["cations"] & OXY)) + ") cannot form BX6 corner-sharing network")
import os; os.makedirs(OUT, exist_ok=True)
with gzip.open(f"{OUT}/structures.jsonl.gz", "wt") as fh:
    for v in sel.values(): fh.write(json.dumps(dict(mp_id=v["mp_id"], structure=v["structure"])) + "\n")
with open(f"{OUT}/curated_set.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["mp_id", "formula", "spacegroup", "nsites", "ehull", "tier", "label", "basis"])
    for v in sorted(sel.values(), key=lambda v: (v["tier"], v["label"], v["formula"])):
        w.writerow([v[k] for k in ("mp_id", "formula", "spacegroup", "nsites", "ehull", "tier", "label", "basis")])
from collections import Counter
print(Counter((v["tier"], v["label"]) for v in sel.values()))
