"""Conventional baseline: prototype substitution into Pm-3m and Pnma perovskite prototypes with charge-balance and
tolerance-factor pre-filtering and a Shannon-radius lattice-scaling guess. No learned model, no relaxation.
usage: python enumerate_baseline.py N seed out.jsonl.gz [compositions.csv]
  With a CSV of (a,b,x) compositions (e.g. decoded output of the legacy 45-D descriptor diffusion model) the same
  prototype/lattice construction is applied to those compositions instead of random draws."""
import gzip, json, sys, os
import numpy as np, pandas as pd
from pymatgen.core import Structure, Species, Composition
sys.path.insert(0, os.path.dirname(__file__))
from crystal import tolerance_factor, classify_perovskite, ANIONS

N, SEED, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
GIVEN = pd.read_csv(sys.argv[4])[["a", "b", "x"]].drop_duplicates().sample(frac=1, random_state=SEED).values.tolist() if len(sys.argv) > 4 else None
lab = pd.read_csv("data/curated/labels.csv"); per = lab[lab.is_perovskite & (lab.nsites <= 20)]
                                                                                                        
dom = {k: sorted({e for v in per[c] for e in v.split(",") if e}) for k, c in (("A", "A"), ("B", "Bsite"), ("X", "X"))}
protos = {}
for name, f, sg in (("Pm-3m", "SrTiO3", 221), ("Pnma", "CaTiO3", 62)):
    r = lab[(lab.formula == f) & (lab.sg_number == sg)].sort_values("ehull").iloc[0]
    for l in gzip.open("data/curated/structures.jsonl.gz", "rt"):
        d = json.loads(l)
        if d["id"] == r.id: protos[name] = (Structure.from_dict(d["structure"]), f); break
rng = np.random.RandomState(SEED); out = []; tries = 0
def shannon(el, cn):
    from crystal import charge_balance
    return None
while len(out) < N and tries < 200 * N:
    tries += 1
    if GIVEN is not None:
        if not GIVEN: break
        a, b, x = GIVEN.pop()
    else:
        a, b, x = rng.choice(dom["A"]), rng.choice(dom["B"]), rng.choice(dom["X"])
    if a == b: continue
    comp = Composition({a: 1, b: 1, x: 3})
    g = comp.oxi_state_guesses()
    if not g: continue
    ox = g[0]
    try:
        rA = Species(a, round(ox[a])).get_shannon_radius("XII"); rB = Species(b, round(ox[b])).get_shannon_radius("VI")
        rX = Species(x, round(ox[x])).get_shannon_radius("VI")
    except Exception:
        continue
    t = (rA + rX) / (np.sqrt(2) * (rB + rX))
    if not 0.78 <= t <= 1.10: continue
    pn = rng.choice(list(protos)); p, f = protos[pn]
    s = p.copy(); mp = {"Sr" if pn == "Pm-3m" else "Ca": a, "Ti": b, "O": x}
    s.replace_species(mp)
                                                                   
    bx = [min(n.nn_distance for n in nb if n.specie.symbol == x) for i, nb in enumerate(s.get_all_neighbors(4.5)) if s[i].specie.symbol == b]
    s.scale_lattice(s.volume * ((rB + rX) / np.mean(bx)) ** 3)
    out.append(dict(id=f"enum_{SEED}_{len(out):06d}", source=f"enumeration:{pn}", structure=s.as_dict()))
with gzip.open(OUT, "wt") as fh:
    for d in out: fh.write(json.dumps(d) + "\n")
print(len(out), "structures after", tries, "draws; domain sizes", {k: len(v) for k, v in dom.items()})
