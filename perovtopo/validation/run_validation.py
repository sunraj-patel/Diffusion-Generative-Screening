"""Run the classifier on the curated set: confusion matrices per tier, misclassified list, and one-at-a-time threshold sweeps.
Usage: python run_validation.py [--sweep-sample N]"""
import sys, json, gzip, csv, argparse, random, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pymatgen.core import Structure
sys.path.insert(0, __file__.rsplit("/validation", 1)[0])
from perovtopo import Params, classify, PEROVSKITE
HERE = __file__.rsplit("/", 1)[0]
ap = argparse.ArgumentParser(); ap.add_argument("--sweep-sample", type=int, default=600); a = ap.parse_args()

meta = {r["mp_id"]: r for r in csv.DictReader(open(f"{HERE}/curated_set.csv"))}
structs = {}
for l in gzip.open(f"{HERE}/structures.jsonl.gz", "rt"):
    d = json.loads(l); structs[d["mp_id"]] = Structure.from_dict(d["structure"])
ids = sorted(meta)

def work(args):
    mid, p = args
    return mid, classify(structs[mid], p)["family"]
def run(p, subset):
    with Pool() as pool:
        return dict(pool.map(work, [(m, p) for m in subset], chunksize=20))
def conf(pred, subset):
    tp = fp = tn = fn = 0
    for m in subset:
        t = meta[m]["label"] == "perovskite"; y = pred[m] == PEROVSKITE
        tp += t and y; fn += t and not y; fp += (not t) and y; tn += (not t) and not y
    n = tp + fp + tn + fn
    return dict(n=n, tp=tp, fp=fp, tn=tn, fn=fn, recall=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                precision=tp / max(tp + fp, 1), accuracy=(tp + tn) / max(n, 1))

if __name__ == "__main__":
    res = {}
    for name, par in (("default", Params()), ("strict_corner", Params(strict_corner=True))):
        base = run(par, ids)
        for tier in ("1", "2"):
            sub = [m for m in ids if meta[m]["tier"] == tier]
            res[f"{name}/tier{tier}"] = conf(base, sub)
        sub = [m for m in ids if meta[m]["tier"] == "2" and meta[m]["ehull"] not in ("", "None") and float(meta[m]["ehull"]) <= 0.1]
        res[f"{name}/tier2_ehull<=0.1"] = conf(base, sub)
        with open(f"{HERE}/misclassified_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["mp_id", "formula", "spacegroup", "tier", "label", "predicted_family", "basis"])
            for m in ids:
                if (meta[m]["label"] == "perovskite") != (base[m] == PEROVSKITE):
                    w.writerow([m, meta[m]["formula"], meta[m]["spacegroup"], meta[m]["tier"], meta[m]["label"], base[m], meta[m]["basis"]])
    json.dump(res, open(f"{HERE}/results_default.json", "w"), indent=1)
    for k, v in res.items(): print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()})
    # sweep on all tier-1 plus a fixed random sample of tier 2 (seeded)
    random.seed(0)
    t2p = [m for m in ids if meta[m]["tier"] == "2" and meta[m]["label"] == "perovskite"]
    t2n = [m for m in ids if meta[m]["tier"] == "2" and meta[m]["label"] != "perovskite"]
    sub = [m for m in ids if meta[m]["tier"] == "1"] + random.sample(t2p, min(a.sweep_sample, len(t2p))) + t2n
    grid = dict(b_tol=[1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50], a_tol=[1.10, 1.20, 1.30, 1.40, 1.50, 1.70],
                a_min_cn=[6, 7, 8, 9, 10, 12], bx_tol=[1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50])
    rows = []
    for par, vals in grid.items():
        for v in vals:
            p = Params().with_(**{par: v}); pred = run(p, sub)
            for tier in ("1", "2"):
                c = conf(pred, [m for m in sub if meta[m]["tier"] == tier]); rows.append(dict(param=par, value=v, tier=tier, **c))
            print(par, v, "tier1 acc %.3f  tier2 recall %.3f spec %.3f" % (rows[-2]["accuracy"], rows[-1]["recall"], rows[-1]["specificity"]), flush=True)
    with open(f"{HERE}/sensitivity_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
