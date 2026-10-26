"""Verified DFT summary + prediction-vs-DFT table. Reads raw QE outputs (dft_ext/results/<ID>/final/{scf,nscf}.out, relax/vcrelax.out), ALIGNN predictions
(dft_ext/alignn/pred_*.csv, made with pipeline/predict.py) and the existing PRV-01/03 table (results/alignn_vs_qe.csv). Writes reports/cms_dft_extension/dft_vs_alignn.csv."""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, "dft/scripts"); from qe_parse import parse
A = {t: pd.read_csv(f"dft_ext/alignn/pred_{t}.csv").set_index("id") for t in ("random_bandgap", "random_ehull", "Bdisjoint_bandgap")}
ret = pd.read_csv("reports/cms_dft_extension/structural_retention.csv").set_index("id")
rows = []
for cid in ["PRV-11", "PRV-09", "PRV-07", "PRV-06", "PRV-14", "CMB-15"]:
    f = f"dft_ext/results/{cid}/final/nscf.out"
    if not os.path.exists(f): continue
    n, s = parse(f), parse(f"dft_ext/results/{cid}/final/scf.out"); r = parse(f"dft_ext/results/{cid}/relax/vcrelax.out")
    ok = n["job_done"] and s["job_done"] and s["scf_converged"] and r["bfgs_converged"]
    rows.append(dict(id=cid, formula=ret.loc[cid, "formula"], natoms=ret.loc[cid, "natoms"], retention=ret.loc[cid, "retention"], dft_converged=ok,
        pbe_gap_nscf_eV=n.get("gap_eV"), pbe_direct_min_nscf_eV=n.get("direct_gap_min_eV"), pbe_gap_scf_eV=s.get("gap_eV"), nscf_nkpts=n["nkpts"], nscf_class=n.get("classification"),
        vbm_k=n.get("vbm_kindex"), cbm_k=n.get("cbm_kindex"),
        alignn_gap_orig=A["random_bandgap"].loc[cid + "_original", "mean"], alignn_gap_orig_std=A["random_bandgap"].loc[cid + "_original", "std"],
        alignn_gap_relaxed=A["random_bandgap"].loc[cid + "_relaxed", "mean"], alignn_gap_relaxed_std=A["random_bandgap"].loc[cid + "_relaxed", "std"],
        alignnBd_gap_orig=A["Bdisjoint_bandgap"].loc[cid + "_original", "mean"], alignn_ehull_orig=A["random_ehull"].loc[cid + "_original", "mean"],
        qe_version=n["qe_version"], wall_nscf_s=n["wall_time_s"]))
d = pd.DataFrame(rows)
old = pd.read_csv("results/alignn_vs_qe.csv")
for _, o in old.iterrows():
    cid = o.candidate; sub = lambda t, k, m: A[t].loc[f"{cid}_{k}", m]
    d.loc[len(d)] = dict(id=cid, formula=o.formula, natoms=10, retention="retained", dft_converged=True, pbe_gap_nscf_eV=o.qe_sr_pbe_gap_eV, pbe_direct_min_nscf_eV=o.qe_sr_pbe_direct_min_eV,
        alignn_gap_orig=sub("random_bandgap", "original", "mean"), alignn_gap_orig_std=sub("random_bandgap", "original", "std"), alignn_gap_relaxed=sub("random_bandgap", "relaxed", "mean"),
        alignn_gap_relaxed_std=sub("random_bandgap", "relaxed", "std"), alignnBd_gap_orig=sub("Bdisjoint_bandgap", "original", "mean"), alignn_ehull_orig=sub("random_ehull", "original", "mean"), qe_version="7.3.1 (earlier run)")
d["alignn_minus_dft_orig_eV"] = d.alignn_gap_orig - d.pbe_gap_nscf_eV
d["alignn_minus_dft_relaxedpred_eV"] = d.alignn_gap_relaxed - d.pbe_gap_nscf_eV
d["within_ensemble_std"] = d.alignn_minus_dft_orig_eV.abs() <= d.alignn_gap_orig_std
d.to_csv("reports/cms_dft_extension/dft_vs_alignn.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40); print(d[["id", "formula", "retention", "dft_converged", "pbe_gap_nscf_eV", "pbe_direct_min_nscf_eV", "alignn_gap_orig", "alignn_gap_orig_std", "alignn_gap_relaxed", "alignn_minus_dft_orig_eV", "alignnBd_gap_orig"]].round(3))
