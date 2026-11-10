"""Default vs strict_corner classification of a list of structures (e.g. the 171 unique perovskites). Usage: check_unique.py UNIQUE.jsonl.gz"""
import sys, json, gzip, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/validation", 1)[0])
from pymatgen.core import Structure
from perovtopo import Params, classify, PEROVSKITE
n = d = s_ = 0; flagged = []
for l in gzip.open(sys.argv[1], "rt"):
    r = json.loads(l); st = Structure.from_dict(r["structure"]); n += 1
    a = classify(st)["family"] == PEROVSKITE; b = classify(st, Params(strict_corner=True))
    d += a; s_ += b["family"] == PEROVSKITE
    if a and b["family"] != PEROVSKITE: flagged.append((r["id"], st.composition.reduced_formula))
print(f"n={n} default_perovskite={d} strict_perovskite={s_} flagged={flagged}")
