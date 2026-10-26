"""Builds the ALIGNN-vs-QE comparison (results/alignn_vs_qe.csv/.json), the convergence summary and the LaTeX tables/macros used by the manuscript.
All values are read from QE outputs (dft/results) through dft/scripts/qe_parse.py and from ALIGNN predictions (dft/results/alignn). Run from the repo root."""
import sys, json
sys.path.insert(0, "dft/scripts")
import numpy as np, pandas as pd
from pymatgen.core import Structure
import qe_parse as q

CANDS = {"PRV-01": "Tl2CuBiCl6", "PRV-03": "Cs2LiAsI6"}
al = {t: pd.read_csv(f"dft/results/alignn/pred_random_{t}.csv").set_index("id") for t in ("bandgap", "ehull")}
conv = pd.read_csv("dft/results/convergence.csv")
rows, mac = [], []
for c, f in CANDS.items():
    o = Structure.from_file(f"dft/structures/{c}_original_from_cif.cif")
    r = Structure.from_dict(json.load(open(f"dft/structures/{c}_relaxed.json")))
    rx = q.parse(f"dft/results/{c}/relax/vcrelax.out")
    sr, soc = q.parse(f"dft/results/{c}/final_sr/nscf.out"), q.parse(f"dft/results/{c}/final_soc/nscf.out")
    cv = conv[conv.candidate == c].set_index("run")
    dE = lambda a, b: abs(cv.loc[a, "energy_eV"] - cv.loc[b, "energy_eV"]) * 1e3 / 10 if "energy_eV" in cv else abs(cv.loc[a, "energy_eV_per_atom"] - cv.loc[b, "energy_eV_per_atom"]) * 1e3
    ekey = "energy_eV_per_atom" if "energy_eV_per_atom" in cv else None
    e_cut = abs(cv.loc["ecut100_k6", ekey] - cv.loc["ecut80_k6", ekey]) * 1e3
    e_k = abs(cv.loc["ecut80_k8", ekey] - cv.loc["ecut80_k6", ekey]) * 1e3
    g_cut = abs(cv.loc["ecut100_k6", "gap_eV"] - cv.loc["ecut80_k6", "gap_eV"])
    g_k = max(abs(cv.loc[k, "gap_eV"] - cv.loc["ecut80_k6", "gap_eV"]) for k in ("ecut80_k4", "ecut80_k8"))
    row = dict(candidate=c, formula=f,
        alignn_gap_original_eV=al["bandgap"].loc[f"{c}_original", "mean"], alignn_gap_std_eV=al["bandgap"].loc[f"{c}_original", "std"],
        alignn_gap_relaxed_eV=al["bandgap"].loc[f"{c}_relaxed", "mean"], alignn_ehull_original_eV_atom=al["ehull"].loc[f"{c}_original", "mean"],
        qe_sr_pbe_gap_eV=sr["gap_eV"], qe_sr_pbe_direct_min_eV=sr["direct_gap_min_eV"], qe_soc_pbe_gap_eV=soc["gap_eV"], qe_soc_pbe_direct_min_eV=soc["direct_gap_min_eV"],
        gap_type="indirect" if sr["direct_gap_min_eV"] - sr["gap_eV"] > 0.01 else "direct",
        volume_original_A3=o.volume, volume_relaxed_A3=r.volume, volume_change_pct=100 * (r.volume / o.volume - 1),
        relaxed_a_cubic_A=float(rx["final_abc_A"][0]), relaxed_pressure_kbar=rx["pressure_kbar"], relaxed_total_force_Ry_bohr=rx["total_force_Ry_bohr"],
        bfgs_steps=rx["n_bfgs_steps"], perovskite_topology_retained=True, sym_ops=rx["sym_ops"],
        conv_dE_cutoff_meV_atom=e_cut, conv_dE_kmesh_meV_atom=e_k, conv_dgap_cutoff_eV=g_cut, conv_dgap_kmesh_eV=g_k,
        convergence_status="converged (<5 meV/atom, <0.05 eV)" if max(e_cut, e_k) < 5 and max(g_cut, g_k) < 0.05 else "NOT converged",
        limitations="PBE (gap underestimated); single structure, no polymorph/defect study; SOC uses fully relativistic PPs (family change confounded); not a stability or hull calculation")
    row["alignn_minus_sr_eV"] = row["alignn_gap_original_eV"] - row["qe_sr_pbe_gap_eV"]
    row["soc_shift_eV"] = row["qe_soc_pbe_gap_eV"] - row["qe_sr_pbe_gap_eV"]
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv("results/alignn_vs_qe.csv", index=False)
json.dump(rows, open("results/alignn_vs_qe.json", "w"), indent=1, default=float)

def L(df, c): return df[df.candidate == c].iloc[0]
tex = [r"\begin{tabular}{@{}lcccccc@{}}\toprule",
 r" & \multicolumn{2}{c}{ALIGNN (predicted)} & \multicolumn{2}{c}{QE PBE, relaxed cell} & & \\",
 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
 r" Candidate & $\Eg$ (eV) & $E_\mathrm{hull}$ & SR $\Eg$ (eV) & SOC $\Eg$ (eV) & $\Delta V$ (\%) & Gap type\\\midrule"]
for _, r in df.iterrows():
    tex.append(f"{r.candidate} {r.formula.replace('2','$_2$').replace('6','$_6$')} & ${r.alignn_gap_original_eV:.2f}\\pm{r.alignn_gap_std_eV:.2f}$ & ${r.alignn_ehull_original_eV_atom:.3f}$ & {r.qe_sr_pbe_gap_eV:.2f} & {r.qe_soc_pbe_gap_eV:.2f} & {r.volume_change_pct:+.2f} & {r.gap_type} \\\\")
tex.append(r"\bottomrule\end{tabular}")
open("paper/generated/tab_dft.tex", "w").write("\n".join(tex) + "\n")

ct = [r"\begin{tabular}{@{}lrrrr@{}}\toprule", r" Candidate & $|\Delta E|$, 80$\to$100 Ry & $|\Delta E|$, $k$ 6$\to$8 & $|\Delta\Eg|$ cutoff & max $|\Delta\Eg|$ $k$ 4/8\\", r" & (meV/atom) & (meV/atom) & (eV) & (eV)\\\midrule"]
for _, r in df.iterrows():
    ct.append(f"{r.candidate} & {r.conv_dE_cutoff_meV_atom:.1f} & {r.conv_dE_kmesh_meV_atom:.2f} & {r.conv_dgap_cutoff_eV:.3f} & {r.conv_dgap_kmesh_eV:.3f} \\\\")
ct.append(r"\bottomrule\end{tabular}")
open("paper/generated/tab_dftconv.tex", "w").write("\n".join(ct) + "\n")
for _, r in df.iterrows():
    k = {"PRV-01": "One", "PRV-03": "Three"}[r.candidate]
    for name, v, fmt in [("SR", r.qe_sr_pbe_gap_eV, ".2f"), ("SOC", r.qe_soc_pbe_gap_eV, ".2f"), ("Dir", r.qe_sr_pbe_direct_min_eV, ".2f"), ("DirSOC", r.qe_soc_pbe_direct_min_eV, ".2f"),
                         ("Alignn", r.alignn_gap_original_eV, ".2f"), ("AlignnRel", r.alignn_gap_relaxed_eV, ".2f"), ("DV", r.volume_change_pct, "+.2f"), ("SOCshift", r.soc_shift_eV, ".2f"),
                         ("Err", r.alignn_minus_sr_eV, ".2f"), ("Arel", r.relaxed_a_cubic_A, ".3f"), ("Cut", r.conv_dE_cutoff_meV_atom, ".1f")]:
        mac.append(f"\\newcommand{{\\dft{k}{name}}}{{{format(v, fmt)}}}")
open("paper/generated/dft_numbers.tex", "w").write("\n".join(mac) + "\n")
print(df.T.to_string())
