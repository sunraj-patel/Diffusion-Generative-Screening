"""Apply the classifier (and its threshold sweep) to the generated pools used in the manuscript.
Usage: python apply_to_generated.py POOL.jsonl.gz [POOL2 ...]   (each line: {"id", "structure"} pymatgen dict)
Writes generated_funnel.csv (default vs strict) and generated_sensitivity.csv (one-at-a-time threshold sweep)."""
import sys, json, gzip, csv, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pymatgen.core import Structure
sys.path.insert(0, __file__.rsplit("/validation", 1)[0])
from perovtopo import Params, classify, PEROVSKITE
from perovtopo.metrics import b_site_multiplicity
HERE = __file__.rsplit("/", 1)[0]

def work(a):
    d, p = a
    s = Structure.from_dict(d)
    r = classify(s, p); return (r["abx3"], r["family"] == PEROVSKITE, b_site_multiplicity(r) if r["family"] == PEROVSKITE else 0)

def run(structs, p):
    with Pool() as pool: out = pool.map(work, [(d, p) for d in structs], chunksize=20)
    out = [o for o in out if o]
    n3 = sum(o[1] for o in out)
    return dict(n=len(out), stoichiometry=sum(o[0] for o in out), topology=n3,
                mixed_b=sum(1 for o in out if o[2] > 1), mixed_b_fraction=(sum(1 for o in out if o[2] > 1) / n3 if n3 else float("nan")))

if __name__ == "__main__":
    funnel, sens = [], []
    for path in sys.argv[1:]:
        name = path.split("/")[-1].replace("pool_", "").replace(".jsonl.gz", "")
        structs = [json.loads(l)["structure"] for l in gzip.open(path, "rt")]
        for label, p in (("default", Params()), ("strict_corner", Params(strict_corner=True))):
            r = run(structs, p); funnel.append(dict(pool=name, classifier=label, **r)); print(name, label, r, flush=True)
        for par, vals in dict(b_tol=[1.15, 1.20, 1.25, 1.30, 1.40], a_tol=[1.2, 1.3, 1.4, 1.5], a_min_cn=[6, 7, 8, 9, 10], bx_tol=[1.15, 1.20, 1.25, 1.30, 1.40]).items():
            for v in vals:
                r = run(structs, Params().with_(**{par: v})); sens.append(dict(pool=name, param=par, value=v, **r)); print(name, par, v, r["topology"], flush=True)
    for fn, rows in (("generated_funnel.csv", funnel), ("generated_sensitivity.csv", sens)):
        with open(f"{HERE}/{fn}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
