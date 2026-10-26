"""Aggregate arms (generated / enumeration / legacy) into comparison tables and select the final candidates.
usage: python make_results.py   (reads work/cand/<arm>_pool.csv, work/gen/*/..._funnel.json; writes results/*.csv|json)"""
import json, os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import rank
from crystal import tolerance_factor

CFG = dict(target=1.4, win=0.3, hull=0.10, std=0.3)
ARMS = json.load(open("work/arms.json"))                                                           
rows, pools = [], {}
for arm, meta in ARMS.items():
    d = pd.read_csv(f"work/cand/{arm}_pool.csv"); d["arm"] = arm; pools[arm] = d
    viable = (d.gap - CFG["target"]).abs().le(CFG["win"]) & d.ehull.le(CFG["hull"]) & d.gap_std.le(CFG["std"])
    pair = [rank.site_distance(d.iloc[i], d.iloc[j]) for i, j in np.random.RandomState(0).randint(0, len(d), (2000, 2)) if i != j] if len(d) > 1 else [np.nan]
    rows.append(dict(arm=arm, label=meta["label"], n_generated=meta["n_generated"], unique_perovskites=len(d),
        yield_unique_per_1000=1000 * len(d) / meta["n_generated"],
        frac_gap_in_window=(d.gap - CFG["target"]).abs().le(CFG["win"]).mean(), frac_ehull_le_0p1=d.ehull.le(CFG["hull"]).mean(),
        viable=int(viable.sum()), viable_per_1000_generated=1000 * viable.sum() / meta["n_generated"],
        median_gap_err=(d.gap - CFG["target"]).abs().median(), median_gap_std=d.gap_std.median(), frac_high_uncertainty=(d.gap_std > CFG["std"]).mean(),
        frac_formula_in_MP=d.formula_in_MP_perovskites.mean(), frac_formula_in_AlexMP20=d.formula_in_AlexMP20_train.mean(),
        frac_struct_in_MP=d.struct_in_MP.mean(), frac_struct_in_AlexMP20=d.struct_in_AlexMP20.mean(),
        frac_new_vs_both=(~(d.struct_in_MP | d.struct_in_AlexMP20)).mean(),
        n_B_elements=len({e for v in d.Bsite.fillna("") for e in v.split(",") if e}), n_A_elements=len({e for v in d.A.fillna("") for e in v.split(",") if e}),
        n_anions=len({e for v in d.X.fillna("") for e in v.split(",") if e}), mean_family_distance=float(np.nanmean(pair)), cost=meta["cost"]))
T = pd.DataFrame(rows); T.to_csv("results/arm_comparison.csv", index=False); print(T.round(3).T.to_string())

                                                                       
sel_rows = []
for arm, d in pools.items():
    dd = d.copy(); dd["unique"] = True
    for m in ("random", "gap_only", "stability_aware", "uncertainty_aware", "proposed"):
        reps = range(200) if m == "random" else [0]
        v = []
        for s in reps:
            sel = rank.select(dd, m, k=5, rng=s)
            dist = [rank.site_distance(sel.iloc[i], sel.iloc[j]) for i in range(len(sel)) for j in range(i)]
            v.append(dict(pred_gap_err=(sel.gap - CFG["target"]).abs().mean(), pred_ehull=sel.ehull.mean(), gap_std=sel.gap_std.mean(),
                          novelty=sel.novelty.mean(), diversity=np.mean(dist), distinct_B=sel.Bsite.nunique(), frac_new_struct=(~(sel.struct_in_MP | sel.struct_in_AlexMP20)).mean()))
        sel_rows.append(dict(arm=arm, method=m, **pd.DataFrame(v).mean().to_dict()))
pd.DataFrame(sel_rows).to_csv("results/selection_rules.csv", index=False); print(pd.DataFrame(sel_rows).round(3).to_string())
