"""Generate LaTeX tables/macros for paper_cms from the audit outputs (no hand-typed numbers)."""
import pandas as pd, numpy as np
G = "paper_cms/generated/"; O = "reports/cms_audit/out/"
                         
B = pd.read_csv(O + "predictor_baselines.csv"); C = pd.read_csv(O + "predictor_classification.csv")
order = ["ALIGNN ensemble", "composition RF (Magpie-like, 50 feat)", "composition kNN (element fractions)", "train median", "constant 0"]
nm = {"ALIGNN ensemble": "ALIGNN ensemble", "composition RF (Magpie-like, 50 feat)": "Composition RF", "composition kNN (element fractions)": "Composition kNN", "train median": "Training median", "constant 0": "Constant zero"}
rows = []
for p in order:
    r = {}
    for sp in ("random", "Bdisjoint"):
        a = B[(B.split == sp) & (B.subset == "all perovskite test") & (B.predictor == p)].iloc[0]
        s = B[(B.split == sp) & (B.subset == "label 0.5-3 eV") & (B.predictor == p)].iloc[0]
        r[sp] = (f"{a.MAE:.2f} [{a.MAE_lo:.2f}, {a.MAE_hi:.2f}]", f"{a.R2:.2f}", f"{s.MAE:.2f}")
    rows.append(f"{nm[p]} & {r['random'][0]} & {r['random'][1]} & {r['random'][2]} & {r['Bdisjoint'][0]} & {r['Bdisjoint'][1]} & {r['Bdisjoint'][2]} \\\\")
open(G + "cms_tab_baselines.tex", "w").write("\\begin{tabular}{@{}lcccccc@{}}\\toprule\n & \\multicolumn{3}{c}{Random split ($n=576$)} & \\multicolumn{3}{c}{B-site disjoint ($n=290$)}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nPredictor & MAE [95\\% CI] & $R^2$ & MAE, 0.5--3~eV & MAE [95\\% CI] & $R^2$ & MAE, 0.5--3~eV\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
                    
cr = []
for _, r in C.iterrows():
    cr.append(f"{r.split.replace('Bdisjoint','B-site disjoint')} & {r.task.replace('>1 eV gap','gap $>1$~eV')} & {int(r.TP+r.FN)} & {r.precision:.2f} & {r.recall:.2f} & {r.F1:.2f}\\\\")
open(G + "cms_tab_classification.tex", "w").write("\\begin{tabular}{@{}llrccc@{}}\\toprule\nSplit & Class & Positives & Precision & Recall & F1\\\\\\midrule\n" + "\n".join(cr) + "\n\\bottomrule\\end{tabular}\n")
                            
A = pd.read_csv(O + "arm_denominators.csv").set_index("arm")
lab = {"base": "MatterGen, unconditional", "gap14": "MatterGen, gap-conditioned, $\\gamma=2$", "gap14g4": "MatterGen, gap-conditioned, $\\gamma=4$", "enum": "Prototype substitution (Pm$\\bar3$m, Pnma)", "legacy": "Legacy descriptor model"}
yr = []
for a in ("base", "gap14", "gap14g4", "enum"):
    r = A.loc[a]
    yr.append(f"{lab[a]} & {int(r.n_attempted_or_built):,} & {int(r.unique_perovskite_structures):,} & {int(r.distinct_formulas):,} & {r.per1000_structures:.1f} & {r.per1000_formulas:.1f} & {100*r.frac_multiB_double:.0f} & {r.in_window_ci.split(' = ')[1].replace('%','\\%').replace(' (',' (').replace(')',')')} \\\\")
open(G + "cms_tab_yield.tex", "w").write("\\begin{tabular}{@{}lrrrrrrl@{}}\\toprule\nArm & $N$ & Unique structures & Distinct formulas & Struct./1000 & Formulas/1000 & Mixed $B$ (\\%) & Predicted gap in window (95\\% CI)\\\\\\midrule\n" + "\n".join(yr) + "\n\\bottomrule\\end{tabular}\n")
                            
R = pd.read_csv(O + "generated_vs_reference.csv"); m = R[R.in_MP & R.mp_label.notna()]; e = (m.pred_gap - m.mp_label)
M = dict(nMatch=len(R), nInMP=int(R.in_MP.sum()), nInAlex=int(R.in_Alex.sum()), nAlexOnly=int((R.in_Alex & ~R.in_MP).sum()), nEither=int((R.in_MP | R.in_Alex).sum()), nNeither=int((~(R.in_MP | R.in_Alex)).sum()),
         gmpN=len(m), gmpMAE=f"{e.abs().mean():.2f}", gmpMed=f"{e.abs().median():.2f}", gmpRMSE=f"{np.sqrt((e**2).mean()):.2f}")
w = m[(m.mp_label >= .5) & (m.mp_label <= 3)]; M.update(gmpWN=len(w), gmpWMAE=f"{(w.pred_gap-w.mp_label).abs().mean():.2f}")
open(G + "cms_numbers.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print(M)
                                                                              
D = pd.read_csv("reports/cms_dft_extension/dft_vs_alignn.csv"); S = pd.read_csv("reports/cms_dft_extension/structural_retention.csv").set_index("id")
old = pd.read_csv("results/alignn_vs_qe.csv").set_index("candidate")
fm = lambda f: __import__("re").sub(r"(\d+)", r"$_{\1}$", f)
rr = []
ids = ["PRV-11", "PRV-09", "PRV-01", "PRV-03", "PRV-07", "PRV-06", "CMB-15", "PRV-14"]
for i in ids:
    if i in S.index:
        s = S.loc[i]; rr.append(f"{i} & {fm(s.formula)} & {int(s.natoms)} & {s.sg_0} & {s.sg_1_0p01} & {s.dV_percent:+.2f} & {s.BX_mean_1:.3f} & {s.BXB_min_deg:.0f} & {int(s.A_CN_0)}$\\to${int(s.A_CN_1)} & {s.retention.replace('-', '--')} \\\\")
    else:
        o = old.loc[i]; rr.append(f"{i} & {fm(o.formula)} & 10 & Fm-3m & Fm-3m & {o.volume_change_pct:+.2f} & -- & -- & -- & retained \\\\")
open(G + "cms_tab_retention.tex", "w").write("\\begin{tabular}{@{}llrllrcccl@{}}\\toprule\nID & Formula & Atoms & SG start & SG relaxed & $\\Delta V$ (\\%) & $\\langle d_{BX}\\rangle$ (\\AA) & min $B$--$X$--$B$ ($^\\circ$) & $A$ CN & Class\\\\\\midrule\n" + "\n".join(rr) + "\n\\bottomrule\\end{tabular}\n")
gr = []
for _, r in D.iterrows():
    gr.append(f"{r.id} & {fm(r.formula)} & {r.alignn_gap_orig:.2f}$\\pm${r.alignn_gap_orig_std:.2f} & {r.alignn_gap_relaxed:.2f} & {r.pbe_gap_nscf_eV:.2f} & {r.pbe_direct_min_nscf_eV:.2f} & {r.alignn_minus_dft_orig_eV:+.2f} & {r.alignn_minus_dft_relaxedpred_eV:+.2f} \\\\")
open(G + "cms_tab_dftgap.tex", "w").write("\\begin{tabular}{@{}llccccrr@{}}\\toprule\nID & Formula & ALIGNN, start & ALIGNN, relaxed & PBE gap & PBE min. direct & $\\Delta_{\\rm start}$ & $\\Delta_{\\rm relaxed}$\\\\\\midrule\n" + "\n".join(gr) + "\n\\bottomrule\\end{tabular}\n")
print(len(gr), "gap rows")
