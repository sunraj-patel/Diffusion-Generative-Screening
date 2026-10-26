"""Independent recomputation of the headline manuscript numbers from the SAVED PER-SAMPLE FILES (not from the result tables the
macros come from), compared with paper/generated/numbers.tex. Writes results/audit_numbers.csv; exits non-zero on any mismatch."""
import gzip, json, os, re, sys
import numpy as np, pandas as pd
from pymatgen.core import Structure
sys.path.insert(0, os.path.dirname(__file__))
from scipy.stats import spearmanr

M = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open("paper/generated/numbers.tex").read())}
val = lambda k: float(M[k].replace("{,}", "").replace("\\%", "")) if re.fullmatch(r"[\d{},.\-]+", M[k]) else M[k]
rows = []
def chk(name, macro, truth, tol=0.051, source=""):
    got = val(macro)
    ok = abs(got - truth) <= tol
    rows.append(dict(claim=name, macro=macro, manuscript=got, recomputed=round(float(truth), 4), ok=bool(ok), source=source))

n = lambda p: sum(1 for _ in gzip.open(p, "rt"))
                                                                  
chk("gap-conditioned structures", "nGap", n("work/pool_gap14.jsonl.gz") if os.path.exists("work/pool_gap14.jsonl.gz") else 0, 0, "work/pool_gap14.jsonl.gz (line count)")
chk("unconditional structures", "nBase", n("work/pool_base.jsonl.gz"), 0, "work/pool_base.jsonl.gz")
S = {a: pd.read_csv(f"work/{a}_screen.csv") for a in ("gap14", "base", "gap14g4", "enum", "legacy")}
for a, t in (("gap14", "Gap"), ("base", "Base"), ("gap14g4", "GapG")):
    s = S[a]
    chk(f"{a}: ABX3", f"abx{t}", s.abx3.sum(), 0, f"work/{a}_screen.csv")
    chk(f"{a}: 3D perovskite", f"perov{t}", s.perovskite.sum(), 0, f"work/{a}_screen.csv")
    chk(f"{a}: charge balanced", f"cb{t}", (s.perovskite & s.charge_balanced.fillna(False).astype(bool)).sum(), 0, f"work/{a}_screen.csv")
    chk(f"{a}: unique perovskites", f"u{t}", s.unique.sum(), 0, f"work/{a}_screen.csv")
    chk(f"{a}: unique per 1000", f"y{t}", 1000 * s.unique.sum() / len(s), 0.051, f"work/{a}_screen.csv")
                                                       
for split, tgt, pre in (("random", "bandgap", "RGP"), ("Bdisjoint", "bandgap", "BG"), ("random", "ehull", "RHP"), ("Bdisjoint", "ehull", "BH")):
    d = pd.read_csv(f"results/pred_{split}_{tgt}.csv"); d = d[d.is_perovskite]
    chk(f"{split} {tgt} perovskite n", f"{pre}n", len(d), 0, f"results/pred_{split}_{tgt}.csv")
    chk(f"{split} {tgt} MAE", f"{pre}mae", (d.y - d['mean']).abs().mean(), 0.0051 if tgt == "ehull" else 0.0051 + 0.005, "same")
    chk(f"{split} {tgt} R2", f"{pre}r", 1 - ((d.y - d['mean']) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum(), 0.0051 + 0.005, "same")
    chk(f"{split} {tgt} Spearman(spread,|err|)", f"{pre}rho", spearmanr(d["std"], (d.y - d["mean"]).abs()).correlation, 0.0051 + 0.005, "same")
d = pd.read_csv("results/pred_Bdisjoint_bandgap.csv"); chk("B-disjoint MAE of constant zero", "bdZero", d.y.abs().mean(), 0.0051 + 0.005, "results/pred_Bdisjoint_bandgap.csv")
                                                  
P = {a: pd.read_csv(f"work/cand/{a}_pool.csv") for a in S}
p = P["gap14"]; chk("window k", "winK", ((p.gap - 1.4).abs() <= .3).sum(), 0, "work/cand/gap14_pool.csv")
chk("window fraction (%)", "winGap", 100 * ((p.gap - 1.4).abs() <= .3).mean(), 0.51, "same")
lab = pd.read_csv("data/curated/labels.csv"); mp = lab[lab.is_perovskite]
chk("MP base rate (%)", "winMP", 100 * ((mp.band_gap - 1.4).abs() <= .3).mean(), 0.051, "data/curated/labels.csv")
chk("in MP (%)", "inMPGap", 100 * p.struct_in_MP.mean(), 0.51, "work/cand/gap14_pool.csv"); chk("in Alex-MP-20 (%)", "inAlexGap", 100 * p.struct_in_AlexMP20.mean(), 0.51, "same")
chk("new vs both (%)", "newfracGap", 100 * (~(p.struct_in_MP | p.struct_in_AlexMP20)).mean(), 0.51, "same")
for a, t in (("gap14", "Gap"), ("base", "Base"), ("gap14g4", "GapG"), ("enum", "Enum"), ("legacy", "Legacy")):
    q = P[a]; v = ((q.gap - 1.4).abs() <= .3) & (q.ehull <= .1) & (q.gap_std <= .3); new = ~(q.struct_in_MP | q.struct_in_AlexMP20)
    N = json.load(open("work/arms.json"))[a]["n_generated"]
    chk(f"{a}: viable", f"v{t}", v.sum(), 0, f"work/cand/{a}_pool.csv"); chk(f"{a}: viable per 1000", f"vy{t}", 1000 * v.sum() / N, 0.051, "same")
    chk(f"{a}: viable and new", f"vnew{t}", (v & new).sum(), 0, "same")
               
g = P["gap14"].fillna("")
chk("median predicted gap, generated", "medGap", g.gap.median(), 0.0051 + 0.005, "work/cand/gap14_pool.csv"); chk("median label gap, MP", "medMP", mp.band_gap.median(), 0.0051 + 0.005, "data/curated/labels.csv")
ns = spearmanr(g.novelty, g.gap_std).correlation; rows.append(dict(claim="Spearman(novelty, gap spread) generated (text 0.37)", macro="(text)", manuscript=0.37, recomputed=round(ns, 4), ok=abs(ns - 0.37) < 0.006, source="work/cand/gap14_pool.csv"))
               
sl = pd.read_csv("results/shortlist.csv")
rows.append(dict(claim="shortlist gap range (text 1.09-1.88)", macro="(text)", manuscript="1.09-1.88", recomputed=f"{sl.gap.min():.2f}-{sl.gap.max():.2f}", ok=(round(sl.gap.min(), 2) == 1.09 and round(sl.gap.max(), 2) == 1.88), source="results/shortlist.csv"))
rows.append(dict(claim="shortlist max predicted E_hull (text <=0.04)", macro="(text)", manuscript=0.04, recomputed=round(sl.ehull.max(), 4), ok=sl.ehull.max() <= 0.04, source="results/shortlist.csv"))
rows.append(dict(claim="shortlist within 0.3 eV of 1.4 (text: 1 of 5, PRV-04)", macro="(text)", manuscript=1, recomputed=int(((sl.gap - 1.4).abs() <= .3).sum()), ok=int(((sl.gap - 1.4).abs() <= .3).sum()) == 1, source="results/shortlist.csv"))
R = pd.DataFrame(rows); R.to_csv("results/audit_numbers.csv", index=False)
print(R.to_string(max_colwidth=48)); print("\nMISMATCHES:", int((~R.ok).sum()), "of", len(R)); sys.exit(int((~R.ok).sum() > 0))
