"""Post-hoc audit and EXPLORATORY analyses from saved outputs only (no new generation or training):
 (1) rates with Wilson intervals and per-seed variation, (2) tolerance-window sensitivity, (3) cost per viable candidate,
 (4) viable-and-new counts, (5) B-site-disjoint baselines, (6) chemical-concern table for the ranked candidates.
Writes results/audit_extra.json and results/*.csv.  Thresholds in (2) are fixed a priori from the literature range and NOT tuned."""
import json, os, re, sys, time, gzip
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
out = {}

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))

def ratio_ci(k1, n1, k2, n2, z=1.96):                                                       
    r = (k1 / n1) / (k2 / n2); se = np.sqrt(1 / k1 - 1 / n1 + 1 / k2 - 1 / n2); return r, r * np.exp(-z * se), r * np.exp(z * se)

S = {a: pd.read_csv(f"work/{a}_screen.csv") for a in ("gap14", "base", "gap14g4", "enum", "legacy")}
P = {a: pd.read_csv(f"work/cand/{a}_pool.csv") for a in S}
N = {a: json.load(open("work/arms.json"))[a]["n_generated"] for a in S}

                                 
rows = []
for a in ("gap14", "base", "gap14g4"):
    s = S[a]; n = len(s)
    for nm, k in (("ABX3", s.abx3.sum()), ("3D perovskite", s.perovskite.sum()), ("unique perovskite", s.unique.sum())):
        lo, hi = wilson(k, n); rows.append(dict(arm=a, stage=nm, k=int(k), n=n, rate=k / n, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv("results/rates_wilson.csv", index=False)
out["rate_ratio_unique_gap_vs_base"] = ratio_ci(S["gap14"].unique.sum(), len(S["gap14"]), S["base"].unique.sum(), len(S["base"]))
out["rate_ratio_perov_gap_vs_base"] = ratio_ci(S["gap14"].perovskite.sum(), len(S["gap14"]), S["base"].perovskite.sum(), len(S["base"]))
g = S["gap14"].copy(); g["seed"] = g.id.str.extract(r"_s(\d)_")[0]
b = S["base"].copy(); b["seed"] = b.id.str.extract(r"_s(\d)_")[0]
per = []
for nm, d in (("gap14", g), ("base", b)):
    for sd, x in d.groupby("seed"): per.append(dict(arm=nm, seed=int(sd), n=len(x), abx3=int(x.abx3.sum()), perovskite=int(x.perovskite.sum()), perov_rate=x.perovskite.mean(), pass_chem=int(x.pass_chem.sum())))
per = pd.DataFrame(per); per.to_csv("results/per_seed_yield.csv", index=False)
out["per_seed_perov_rate_gap14"] = [float(v) for v in per[per.arm == "gap14"].perov_rate]
out["per_seed_perov_rate_base"] = [float(v) for v in per[per.arm == "base"].perov_rate]
                                                                                                             
p = P["gap14"]; k = int(((p.gap - 1.4).abs() <= .3).sum()); out["window_gap14"] = dict(k=k, n=len(p), frac=k / len(p), ci=wilson(k, len(p)), per_1000_generated=1000 * k / N["gap14"])
p = P["gap14g4"]; k = int(((p.gap - 1.4).abs() <= .3).sum()); out["window_gap14g4"] = dict(k=k, n=len(p), frac=k / len(p), ci=wilson(k, len(p)))
mp = pd.read_csv("data/curated/labels.csv"); mp = mp[mp.is_perovskite]; k = int(((mp.band_gap - 1.4).abs() <= .3).sum()); out["window_MP"] = dict(k=k, n=len(mp), frac=k / len(mp), ci=wilson(k, len(mp)))

                                                                        
def passing(s, lo, hi, need_t):
    ok = s.parsed & s.geometry_ok & s.abx3 & s.perovskite & s.charge_balanced.fillna(False).astype(bool)
    t = s.tolerance
    inside = t.between(lo, hi)
    ok &= inside if need_t else (inside | t.isna())
    return ok
CFG = [("used (0.78-1.10, unassessable kept)", 0.78, 1.10, False), ("0.80-1.00, unassessable kept", 0.80, 1.00, False), ("0.80-1.00, assessable only", 0.80, 1.00, True), ("0.825-1.05, assessable only", 0.825, 1.05, True)]
rows = []
for a in S:
    s = S[a]; p = P[a].set_index("id")
    for nm, lo, hi, need in CFG:
        m = passing(s, lo, hi, need); ids = s.id[m]
        grp = s.loc[m & (s.dup_group >= 0), "dup_group"].nunique()
        pp = p[p.index.isin(ids)]
        via = ((pp.gap - 1.4).abs() <= .3) & (pp.ehull <= .1) & (pp.gap_std <= .3)
        rows.append(dict(arm=a, tolerance=nm, passing=int(m.sum()), unique=len(pp), unique_per_1000=1000 * len(pp) / N[a], viable=int(via.sum()), viable_per_1000=1000 * via.sum() / N[a]))
T = pd.DataFrame(rows); T.to_csv("results/tolerance_sensitivity.csv", index=False)

                                                   
costs = json.load(open("work/gen_costs.json")); cost_rows = []
for a in S:
    p = P[a]; via = ((p.gap - 1.4).abs() <= .3) & (p.ehull <= .1) & (p.gap_std <= .3); new = ~(p.struct_in_MP | p.struct_in_AlexMP20)
    gpu = costs.get(a, {}).get("gpu_hours", np.nan)
    cost_rows.append(dict(arm=a, N=N[a], viable=int(via.sum()), viable_new=int((via & new).sum()), viable_per_1000=1000 * via.sum() / N[a], viable_new_per_1000=1000 * (via & new).sum() / N[a],
                          gpu_h=gpu, gpu_h_per_viable=gpu / via.sum() if via.sum() else np.nan, gpu_h_per_viable_new=gpu / (via & new).sum() if (via & new).sum() else np.nan,
                          viable_ci_lo=wilson(int(via.sum()), N[a])[0] * 1000, viable_ci_hi=wilson(int(via.sum()), N[a])[1] * 1000))
C = pd.DataFrame(cost_rows); C.to_csv("results/cost_per_viable.csv", index=False)

                                                          
d = pd.read_csv("results/pred_Bdisjoint_bandgap.csv")
out["bdisjoint_gap"] = dict(n=len(d), true_metal_frac=float((d.y < 0.1).mean()), pred_metal_frac=float((d["mean"] < 0.1).mean()), mae_ensemble=float((d.y - d["mean"]).abs().mean()),
    mae_predict_zero=float(d.y.abs().mean()), mae_train_mean=float((d.y - 1.332).abs().mean()), mae_median_of_test=float((d.y - d.y.median()).abs().mean()),
    wide_gap_true=int((d.y > 1).sum()), wide_gap_recovered=int(((d.y > 1) & (d["mean"] > 1)).sum()))
r = pd.read_csv("results/pred_random_bandgap.csv"); r = r[r.is_perovskite]
out["random_gap_perov"] = dict(n=len(r), mae_ensemble=float((r.y - r["mean"]).abs().mean()), mae_predict_zero=float(r.y.abs().mean()))
json.dump(out, open("results/audit_extra.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)); print(T.round(2).to_string()); print(C.round(2).to_string()); print(per.round(3).to_string())
