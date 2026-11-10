"""python -m perovtopo classify FILE_OR_GLOB ... [--csv out.csv] [--b-tol 1.25 --a-tol 1.3 --a-min-cn 8 --bx-tol 1.25]"""
import argparse, csv, glob, sys
from pymatgen.core import Structure
from .classifier import Params, classify
from .metrics import funnel

def main():
    ap = argparse.ArgumentParser(prog="perovtopo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("paths", nargs="+"); c.add_argument("--csv")
    for k, v in Params().__dict__.items():
        c.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    p = Params(**{k: getattr(a, k) for k in Params().__dict__})
    files = sorted({f for x in a.paths for f in (glob.glob(x) or [x])})
    structs = [Structure.from_file(f) for f in files]
    summ, res = funnel(structs, p)
    rows = [dict(file=f, **{k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()}) for f, r in zip(files, res)]
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(r["file"], "->", r["family"])
    print({k: summ[k] for k in ("n", "stoichiometry", "topology", "mixed_b_fraction")}, file=sys.stderr)

if __name__ == "__main__":
    main()
