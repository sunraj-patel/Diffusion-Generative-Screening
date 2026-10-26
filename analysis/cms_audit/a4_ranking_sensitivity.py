"""Audit G: reproduce the shortlist from work/cand/gap14_pool.csv with pipeline/rank.py, then vary gates/weights/kappa/diversity and bootstrap the pool.
Also compare with the simple selectors on the same eligible pool. Nothing here changes the published shortlist."""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "pipeline"); import rank
O = "reports/cms_audit/out/"
d = pd.read_csv("work/cand/gap14_pool.csv", keep_default_na=False, na_values=[""])
d["unique"] = d["unique"].astype(bool)
base = dict(hull_max=.10, gap_gate=.60, kappa=1.0, W=(.5, .3, .2), min_dist=.5, std_max=.3)
def short(pool, hull_max, gap_gate, kappa, W, min_dist, std_max, k=5):
    rank.W = dict(gap=W[0], hull=W[1], novelty=W[2])
    e = pool[(pool.gap_std <= std_max) & ~pool.struct_in_MP & pool.unique & pool.perovskite]
    try: r = rank.proposed_rank(e, 1.4, kappa, hull_max, gap_gate, 0)
    except ValueError: return []
    return list(rank.diverse_select(r, k, min_dist=min_dist).id) if len(r) else []
ref = short(d, **base); print("reproduced shortlist:", d.set_index("id").loc[ref].formula.tolist())
cur = pd.read_csv("results/shortlist.csv").sample_id.tolist() if "sample_id" in pd.read_csv("results/shortlist.csv").columns else []
print("matches published shortlist ids:", sorted(ref) == sorted(cur), "| published:", cur)
rows = []
grid = dict(hull_max=[.05, .1, .15], gap_gate=[.4, .6, .8], kappa=[0, 1, 2], W=[(.5, .3, .2), (.4, .4, .2), (.6, .2, .2), (.34, .33, .33), (.7, .3, 0), (.5, .5, 0)], min_dist=[0, .3, .5, .7])
for hm, gg, ka, W, md in itertools.product(*grid.values()):
    s = short(d, hm, gg, ka, W, md, .3); rows.append(dict(hull_max=hm, gap_gate=gg, kappa=ka, W=str(W), min_dist=md, shortlist=";".join(sorted(s)), overlap=len(set(s) & set(ref)), n=len(s)))
S = pd.DataFrame(rows); S.to_csv(O + "ranking_sensitivity_grid.csv", index=False)
print("grid size", len(S), "| mean overlap with published (of 5): %.2f | fraction identical: %.2f | fraction >=3 shared: %.2f" % (S.overlap.mean(), (S.overlap == 5).mean(), (S.overlap >= 3).mean()))
from collections import Counter
c = Counter(i for s in S.shortlist for i in s.split(";") if i); fm = d.set_index("id").formula
print("selection frequency across grid:"); [print("  ", fm[i], i in ref, round(v/len(S), 2)) for i, v in c.most_common(10)]
                             
rng = np.random.default_rng(0); cnt = Counter(); B = 300
for b in range(B):
    bs = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).drop_duplicates("id")
    cnt.update(short(bs, **base))
print("bootstrap (300 resamples of the generated pool) selection frequency of published members:", {fm[i]: round(cnt[i]/B, 2) for i in ref})
                                                                                                 
e = d[(d.gap_std <= .3) & ~d.struct_in_MP & d.unique & d.perovskite]
for m in ("gap_only", "stability_aware", "uncertainty_aware"):
    s = rank.select(e, m, 5); print(m, s.formula.tolist(), "n within 0.3 eV:", int(((s.gap - 1.4).abs() <= .3).sum()))
