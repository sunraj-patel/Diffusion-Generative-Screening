"""Held-out evaluation of the ALIGNN ensembles from saved prediction_results_test_set.csv files.
usage: python evaluate_predictors.py MODELS_DIR OUT_JSON   (MODELS_DIR has {split}_{target}_s{seed}/ directories)"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from metrics import regression_metrics, ensemble_stats, uncertainty_report

mdir, out = sys.argv[1:3]
lab = pd.read_csv("data/curated/labels.csv").set_index("id")
res = {}
for split in ("random", "Bdisjoint"):
    for tgt, col in (("bandgap", "band_gap"), ("ehull", "ehull")):
        files = sorted(glob.glob(f"{mdir}/{split}_{tgt}_s*/prediction_results_test_set.csv"))
        if not files: continue
        dfs = [pd.read_csv(f).assign(id=lambda d: d.id.str.replace(".cif", "", regex=False).str.split("/").str[-1]).set_index("id") for f in files]
        ids = dfs[0].index
        P = np.stack([d.loc[ids, "prediction"].values for d in dfs])
        if tgt == "ehull": P = np.clip(P, 0, None)
        mean, std = ensemble_stats(P)
        y = lab.loc[ids, col].values.copy()
        if tgt == "ehull": y = np.clip(y, None, 1.0)                                                
        trainmean = lab[lab[f"split_{split}"] == "train"][col].clip(upper=1.0 if tgt == "ehull" else None).mean()
        sub = {"all_test": np.ones(len(ids), bool), "perovskite_test": lab.loc[ids, "is_perovskite"].values}
        for name, m in sub.items():
            if m.sum() < 5: continue
            r = dict(members=len(files), **{k: float(v) for k, v in regression_metrics(y[m], mean[m], trainmean).items()})
            if len(files) > 1: r["uncertainty"] = uncertainty_report(y[m], mean[m], std[m])
            r["single_member_mae"] = [float(np.mean(np.abs(y[m] - P[k][m]))) for k in range(len(files))]
            res[f"{split}/{tgt}/{name}"] = r
        pd.DataFrame(dict(id=ids, y=y, mean=mean, std=std, is_perovskite=lab.loc[ids, "is_perovskite"].values)).to_csv(
            f"{os.path.dirname(out)}/pred_{split}_{tgt}.csv", index=False)
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items(): print(k, {a: round(b, 3) for a, b in v.items() if isinstance(b, float)})
