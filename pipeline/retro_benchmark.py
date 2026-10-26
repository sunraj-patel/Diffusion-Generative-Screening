"""Retrospective test of the selection rules on held-out MP perovskites with DFT labels (label-bearing pools only).
For each pool, each rule picks k structures using ONLY predictions; hits are scored with the true MP labels.
usage: python retro_benchmark.py MODELS_RESULTS_DIR OUT_CSV"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import rank
from candidates import vec

res, out = sys.argv[1:3]
lab = pd.read_csv("data/curated/labels.csv").set_index("id")
cfg = dict(target=1.4, win=0.3, hull_hit=0.10)
rows = []
for split in ("random", "Bdisjoint"):
    g = pd.read_csv(f"{res}/pred_{split}_bandgap.csv"); h = pd.read_csv(f"{res}/pred_{split}_ehull.csv")
                                                                                                   
    ids = pd.read_csv(sorted(__import__("glob").glob(f"{res}/models/{split}_bandgap_s0/prediction_results_test_set.csv"))[0]).id.str.replace(".cif", "", regex=False).str.split("/").str[-1]
    d = pd.DataFrame(dict(id=ids, gap=g["mean"], gap_std=g["std"], ehull=h["mean"].clip(lower=0), ehull_std=h["std"]))
    d = d[lab.loc[d.id, "is_perovskite"].values].reset_index(drop=True)
    L = lab.loc[d.id]; d["formula"] = L.formula.values; d["A"] = L.A.values; d["Bsite"] = L.Bsite.values; d["X"] = L.X.values
    d["true_gap"] = L.band_gap.values; d["true_ehull"] = L.ehull.values; d["unique"] = True
    train = lab[lab[f"split_{split}"] == "train"]; R = np.array([vec(f) for f in train[train.is_perovskite].formula.unique()])
    nn = np.array([0.5 * np.abs(R - vec(f)).sum(1).min() for f in d.formula]); d["novelty"] = pd.Series(nn).rank(pct=True).values
    d["hit"] = ((d.true_gap - cfg["target"]).abs() <= cfg["win"]) & (d.true_ehull <= cfg["hull_hit"])
    for k in (5, 10):
        for m in ("random", "gap_only", "stability_aware", "uncertainty_aware", "proposed"):
            reps = range(200) if m == "random" else [0]
            vals = []
            for s in reps:
                sel = rank.select(d, m, k=k, rng=s)
                dist = [rank.site_distance(sel.iloc[i], sel.iloc[j]) for i in range(len(sel)) for j in range(i)]
                vals.append(dict(hit_rate=sel.hit.mean(), true_gap_err=(sel.true_gap - cfg["target"]).abs().mean(), true_ehull=sel.true_ehull.mean(),
                                 pred_gap_err=(sel.gap - cfg["target"]).abs().mean(), diversity=np.mean(dist) if dist else np.nan, distinct_B=sel.Bsite.nunique()))
            rows.append(dict(split=split, k=k, method=m, pool=len(d), pool_hits=int(d.hit.sum()), **pd.DataFrame(vals).mean().to_dict()))
pd.DataFrame(rows).to_csv(out, index=False); print(pd.DataFrame(rows).round(3).to_string())
