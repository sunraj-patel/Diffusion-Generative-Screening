"""Emit LaTeX number macros and tables from result files -> paper/generated/. Every number in the manuscript text comes from here."""
import json, os, re
import numpy as np, pandas as pd

OUT = "paper/generated"; os.makedirs(OUT, exist_ok=True)
M = {}                                                           
def put(name, val, fmt=None):
    if fmt: val = format(val, fmt)
    M[name] = str(val)
def num(x, d=0): return f"{x:,.{d}f}".replace(",", "{,}")

cr = json.load(open("data/curated/curation_report.json")); P = json.load(open("results/predictor_metrics.json"))
A = pd.read_csv("results/arm_comparison.csv").set_index("arm"); SEL = pd.read_csv("results/selection_rules.csv"); RB = pd.read_csv("results/retro_benchmark.csv")
RT = pd.read_csv("results/ranked_table.csv"); SL = RT[RT.shortlist]
put("NTotal", num(cr["counts"]["total"])); put("NABX", num(cr["counts"]["abx3"])); put("NPerov", num(cr["counts"]["perovskite"]))
for k, v in cr["counts"]["families"].items(): pass
for arm in A.index:
    r = A.loc[arm]; t = {"gap14": "Gap", "base": "Base", "enum": "Enum", "legacy": "Legacy", "gap14g4": "GapG"}[arm]
    put(f"n{t}", num(r.n_generated)); put(f"u{t}", num(r.unique_perovskites)); put(f"y{t}", r.yield_unique_per_1000, ".1f")
    put(f"v{t}", num(r.viable)); put(f"vy{t}", r.viable_per_1000_generated, ".1f"); put(f"win{t}", 100 * r.frac_gap_in_window, ".0f")
    put(f"newfrac{t}", 100 * r.frac_new_vs_both, ".0f"); put(f"inMP{t}", 100 * r.frac_struct_in_MP, ".0f"); put(f"inAlex{t}", 100 * r.frac_struct_in_AlexMP20, ".0f")
    put(f"fMP{t}", 100 * r.frac_formula_in_MP, ".0f"); put(f"fAlex{t}", 100 * r.frac_formula_in_AlexMP20, ".0f")
    put(f"hu{t}", 100 * r.frac_high_uncertainty, ".0f"); put(f"nB{t}", int(r.n_B_elements)); put(f"nA{t}", int(r.n_A_elements)); put(f"hull{t}", 100 * r.frac_ehull_le_0p1, ".0f")
    put(f"cost{t}", r.cost)
for arm, t in (("gap14", "Gap"), ("base", "Base"), ("enum", "Enum"), ("legacy", "Legacy"), ("gap14g4", "GapG")):
    p = f"work/{arm}_funnel.json"
    if not os.path.exists(p): continue
    f = pd.DataFrame(json.load(open(p))).set_index("stage")
    put(f"abx{t}", num(f.loc["ABX3 stoichiometry", "n"])); put(f"perov{t}", num(f.loc["3D perovskite", "n"])); put(f"cb{t}", num(f.loc["charge balanced", "n"])); put(f"tol{t}", num(f.loc["tolerance window / unassessable", "n"]))
    put(f"pabx{t}", 100 * f.loc["ABX3 stoichiometry", "retained_total"], ".1f"); put(f"pperov{t}", 100 * f.loc["3D perovskite", "retained_total"], ".1f")
for key, t in (("random/bandgap/all_test", "RG"), ("random/bandgap/perovskite_test", "RGP"), ("random/ehull/all_test", "RH"), ("random/ehull/perovskite_test", "RHP"),
               ("Bdisjoint/bandgap/perovskite_test", "BG"), ("Bdisjoint/ehull/perovskite_test", "BH")):
    m = P[key]
    for k in ("n", "mae", "rmse", "r2", "baseline_mae"): put(f"{t}{k}", m[k], ".0f" if k == "n" else ".3f" if "ehull" in key and k != "n" else ".2f")
    if "uncertainty" in m: put(f"{t}rho", m["uncertainty"]["spearman_std_abs_err"], ".2f"); [put(f"{t}cov{n}", 100 * m["uncertainty"]["gaussian_coverage"][c], ".0f") for c, n in (("50%", "Fifty"), ("80%", "Eighty"), ("95%", "NinetyFive"))]
for t, key in (("RGP", "random/bandgap/perovskite_test"), ("BG", "Bdisjoint/bandgap/perovskite_test")):
    b = P[key]["uncertainty"]["bins"]; put(f"{t}binlo", b[0]["mae"], ".2f"); put(f"{t}binhi", b[-1]["mae"], ".2f")
MN = {"random": "Rnd", "gap_only": "GapOnly", "stability_aware": "Stab", "uncertainty_aware": "Unc", "proposed": "Prop"}
for _, r in SEL[SEL.arm == "gap14"].iterrows():
    t = MN[r.method]; put(f"sel{t}err", r.pred_gap_err, ".2f"); put(f"sel{t}hull", r.pred_ehull, ".3f"); put(f"sel{t}std", r.gap_std, ".2f"); put(f"sel{t}nov", r.novelty, ".2f"); put(f"sel{t}div", r.diversity, ".2f"); put(f"sel{t}new", 100 * r.frac_new_struct, ".0f")
for sp, sn in (("random", "R"), ("Bdisjoint", "B")):
    for k in (5, 10):
        for m, t in MN.items():
            r = RB[(RB.split == sp) & (RB.k == k) & (RB.method == m)].iloc[0]; put(f"ret{sn}{k}{t}hit", r.hit_rate, ".2f"); put(f"ret{sn}{k}{t}err", r.true_gap_err, ".2f")
    put(f"ret{sn}pool", int(r.pool)); put(f"ret{sn}hits", int(r.pool_hits))
G = json.load(open("results/gap_distribution.json"))
for arm, t in (("gap14", "Gap"), ("base", "Base"), ("gap14g4", "GapG"), ("enum", "Enum"), ("legacy", "Legacy")):
    put(f"med{t}", G[arm]["median_pred_gap"], ".2f"); put(f"metal{t}", 100 * G[arm]["frac_metal_pred"], ".0f")
put("medMP", G["MP_labels"]["median_gap"], ".2f"); put("metalMP", 100 * G["MP_labels"]["frac_metal"], ".0f"); put("winMP", 100 * G["MP_labels"]["frac_in_window"], ".1f")
C = json.load(open("results/chemistry.json"))
for nm, t in (("MatterGen gap-conditioned", "Gen"), ("MP perovskites", "MP"), ("Prototype substitution", "EnumC")):
    for c, ct in (("oxide", "Ox"), ("fluoride", "F"), ("chloride/bromide/iodide", "Hal"), ("chalcogenide (S,Se,Te)", "Chal")): put(f"an{t}{ct}", 100 * C[nm].get(c, 0), ".0f")
put("dblGen", 100 * C["double_or_mixed_fraction_gen"], ".0f"); put("dblMP", 100 * C["double_or_mixed_fraction_MP"], ".0f")
put("nTierOne", int((RT.tier == 1).sum()))

             
names = {"gap14": "MatterGen, $\\Eg$-conditioned", "base": "MatterGen, unconditional", "gap14g4": "MatterGen, $\\Eg$-cond., $\\gamma{=}4$", "enum": "Prototype substitution", "legacy": "Legacy 45-D + prototypes"}
rows = []
for arm in A.index:
    r = A.loc[arm]
    rows.append(f"{names[arm]} & {num(r.n_generated)} & {num(r.unique_perovskites)} & {r.yield_unique_per_1000:.0f} & {100*r.frac_gap_in_window:.0f} & {100*r.frac_ehull_le_0p1:.0f} & {num(r.viable)} & {r.viable_per_1000_generated:.1f} & {100*r.frac_new_vs_both:.0f} & {100*r.frac_high_uncertainty:.0f} \\\\")
open(f"{OUT}/tab_arms.tex", "w").write("\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\n Arm & $N$ & Unique & per 1000 & Gap in & Hull & Qualifying & per 1000 & New & $\\sigma_g$\\\\\n & & perov. & & window (\\%) & $\\le0.1$ (\\%) & & & (\\%) & $>0.3$ (\\%)\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
rows = []
for key, nm in (("random/bandgap/all_test", "Gap, random, all"), ("random/bandgap/perovskite_test", "Gap, random, perovskites"), ("Bdisjoint/bandgap/perovskite_test", "Gap, B-site disjoint"),
                ("random/ehull/all_test", "$\\Ehull$, random, all"), ("random/ehull/perovskite_test", "$\\Ehull$, random, perovskites"), ("Bdisjoint/ehull/perovskite_test", "$\\Ehull$, B-site disjoint")):
    m = P[key]; d = 3 if "ehull" in key else 2
    rows.append(f"{nm} & {m['n']:.0f} & {m['mae']:.{d}f} & {m['rmse']:.{d}f} & {m['r2']:.2f} & {m['baseline_mae']:.{d}f} & {m['y_min']:.1f}--{m['y_max']:.1f} & {m['uncertainty']['spearman_std_abs_err']:.2f} \\\\")
open(f"{OUT}/tab_pred.tex", "w").write("\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n Target and test set & $n$ & MAE & RMSE & $R^2$ & Mean-baseline MAE & Label range & $\\rho(\\sigma,|e|)$\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
mn = {"random": "random", "gap_only": "gap only", "stability_aware": "gap + hull gate", "uncertainty_aware": "risk-adjusted gap + hull gate", "proposed": "proposed"}
rows = []
for sp, spn in (("random", "random split"), ("Bdisjoint", "B-site disjoint")):
    for m in mn:
        r = RB[(RB.split == sp) & (RB.k == 5) & (RB.method == m)].iloc[0]
        rows.append(f"{spn if m=='random' else ''} & {mn[m]} & {r.hit_rate:.2f} & {r.true_gap_err:.2f} & {r.true_ehull:.3f} & {r.diversity:.2f} \\\\")
    rows.append("\\midrule" if sp == "random" else "")
open(f"{OUT}/tab_retro.tex", "w").write("\\begin{tabular}{@{}llrrrr@{}}\\toprule\n Pool & Rule & Hit rate & True $|\\Eg-1.4|$ (eV) & True $\\Ehull$ & Diversity\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
rows = []
for _, r in RT.iterrows():
    tf = f"{r.tolerance_factor:.2f}" if r.tolerance_factor == r.tolerance_factor else "n/a"
    star = "\\textbf{" + r.candidate + "}" if r.shortlist else r.candidate
    rows.append(f"{star} & {r.formula} & {r.spacegroup.replace('_','\\_')} & {r.gap:.2f}$\\pm${r.gap_std:.2f} & {r.ehull:.3f}$\\pm${r.ehull_std:.3f} & {tf} & {'yes' if r.struct_in_AlexMP20 else 'no'} & {'yes' if r.formula_in_MP_perovskites else 'no'} & {r.novelty:.2f} \\\\")
open(f"{OUT}/tab_ranked.tex", "w").write("\\begin{tabular}{@{}llllllllr@{}}\\toprule\n ID & Formula & Space group & $\\hat E_{\\rm g}\\pm\\sigma$ (eV) & $\\hat E_{\\rm hull}\\pm\\sigma$ (eV/atom) & $t$ & In Alex-MP-20 & Formula in MP & Novelty\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
                                                                           
X = json.load(open("results/audit_extra.json")); CP = pd.read_csv("results/cost_per_viable.csv").set_index("arm"); TS = pd.read_csv("results/tolerance_sensitivity.csv"); TM = json.load(open("results/timing.json"))
CC = pd.read_csv("results/candidate_concerns.csv").fillna("")
r_, lo_, hi_ = X["rate_ratio_unique_gap_vs_base"]; put("ratioUnique", r_, ".1f"); put("ratioUniqueLo", lo_, ".1f"); put("ratioUniqueHi", hi_, ".1f")
r_, lo_, hi_ = X["rate_ratio_perov_gap_vs_base"]; put("ratioPerov", r_, ".1f"); put("ratioPerovLo", lo_, ".1f"); put("ratioPerovHi", hi_, ".1f")
pg = X["per_seed_perov_rate_gap14"]; pb = X["per_seed_perov_rate_base"]
put("seedGapLo", 100 * min(pg) + 1e-9, ".1f"); put("seedGapHi", 100 * max(pg) + 1e-9, ".1f"); put("seedBaseLo", 100 * min(pb) + 1e-9, ".1f"); put("seedBaseHi", 100 * max(pb) + 1e-9, ".1f")
w = X["window_gap14"]; put("winK", w["k"]); put("winN", w["n"]); put("winLo", 100 * w["ci"][0], ".0f"); put("winHi", 100 * w["ci"][1], ".0f"); put("winPerK", w["per_1000_generated"], ".1f")
w = X["window_gap14g4"]; put("winGK", w["k"]); put("winGN", w["n"]); put("winGLo", 100 * w["ci"][0], ".0f"); put("winGHi", 100 * w["ci"][1], ".0f")
w = X["window_MP"]; put("winMPLo", 100 * w["ci"][0], ".1f"); put("winMPHi", 100 * w["ci"][1], ".1f")
b_ = X["bdisjoint_gap"]
put("bdZero", b_["mae_predict_zero"], ".2f"); put("bdEns", b_["mae_ensemble"], ".2f"); put("bdMetalTrue", 100 * b_["true_metal_frac"], ".0f"); put("bdMetalPred", 100 * b_["pred_metal_frac"], ".0f")
put("bdWide", b_["wide_gap_true"]); put("bdWideRec", b_["wide_gap_recovered"])
for arm, t in (("gap14", "Gap"), ("base", "Base"), ("gap14g4", "GapG"), ("enum", "Enum"), ("legacy", "Legacy")):
    c = CP.loc[arm]; put(f"vnew{t}", int(c.viable_new)); put(f"vnewy{t}", c.viable_new_per_1000, ".1f"); put(f"vciLo{t}", c.viable_ci_lo, ".1f"); put(f"vciHi{t}", c.viable_ci_hi, ".1f")
    if c.gpu_h == c.gpu_h: put(f"gpuper{t}", c.gpu_h_per_viable, ".2f"); put(f"gpuh{t}", c.gpu_h, ".1f")
    t_ = TS[(TS.arm == arm)]
    put(f"tolStrictU{t}", int(t_.iloc[2]["unique"])); put(f"tolStrictV{t}", int(t_.iloc[2].viable)); put(f"tolMidU{t}", int(t_.iloc[1]["unique"])); put(f"tolMidV{t}", int(t_.iloc[1].viable))
put("enumDraws", num(TM["enum_random_draws"])); put("enumCPUsec", f"{TM['enum_total_cpu_wall_s']:.0f}"); put("enumCPUper", TM["enum_total_cpu_wall_s"] / int(A.loc["enum", "viable"]), ".1f")
put("enumAccept", 100 * TM["enum_structures"] / TM["enum_random_draws"], ".1f")
put("concNoneTierOne", int(((CC.tier == 1) & (CC.n_concern_flags == 0)).sum()))
rows = []
for _, r in CC.iterrows():
    fl = [x for x in (("toxic: " + r.toxic) if r.toxic else "", ("lone pair on B: " + r.lone_pair_on_B) if r.lone_pair_on_B else "", ("large monovalent on B: " + r.large_monovalent_on_B) if r.large_monovalent_on_B else "",
          "oxidation states not unique" if r.oxidation_not_unique is True or str(r.oxidation_not_unique) == "True" else "", "$t$ n/a" if str(r.tolerance_not_assessable) == "True" else "", "$\\sigma_g>0.2$" if str(r.spread_gt_0p2) == "True" else "") if x]
    rows.append(f"{'\\textbf{'+r.candidate+'}' if r.shortlist else r.candidate} & {r.formula} & {r.tier} & {r.gap:.2f} & {'; '.join(fl) if fl else '--'} & {int(r.n_concern_flags)} \\\\")
open(f"{OUT}/tab_concerns.tex", "w").write("\\begin{tabular}{@{}llrrp{7.2cm}r@{}}\\toprule\n ID & Formula & Tier & $\\hat E_{\\rm g}$ & Flags (heuristic) & $n$ \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
names2 = {"gap14": "MatterGen, $\\Eg$-conditioned", "base": "MatterGen, unconditional", "gap14g4": "MatterGen, $\\gamma{=}4$", "enum": "Prototype substitution", "legacy": "Legacy 45-D + prototypes"}
rows = []
for arm in CP.index:
    c = CP.loc[arm]; ts = TS[TS.arm == arm]
    cost = f"{c.gpu_h:.1f} V100-h" if c.gpu_h == c.gpu_h else ("%.0f s CPU$^{\\dagger}$" % TM["enum_total_cpu_wall_s"] if arm == "enum" else "CPU, not timed")
    per = f"{c.gpu_h_per_viable:.2f}" if c.gpu_h == c.gpu_h and c.viable else ("%.1f s" % (TM["enum_total_cpu_wall_s"] / c.viable) if arm == "enum" else "--")
    rows.append(f"{names2[arm]} & {num(c.N)} & {int(c.viable)} ({c.viable_ci_lo:.1f}--{c.viable_ci_hi:.1f}) & {int(c.viable_new)} & {c.viable_new_per_1000:.1f} & {cost} & {per} & {int(ts.iloc[2]["viable"])} \\\\")
open(f"{OUT}/tab_cost.tex", "w").write("\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n Arm & $N$ & Qualifying & Qualifying and & per 1000 & Cost of & Cost per & Qualifying, strict\\\\\n & & $n$ (95\\% CI per 1000) & new & (new) & generation & viable & $t$ window$^\\ddagger$\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
rows = []
for arm in ("gap14", "base", "gap14g4", "enum", "legacy"):
    for _, r in TS[TS.arm == arm].iterrows():
        rows.append(f"{names2[arm] if r.tolerance.startswith('used') else ''} & {r.tolerance.replace('-', '--')} & {int(r.passing)} & {int(r["unique"])} & {r["unique_per_1000"]:.0f} & {int(r.viable)} & {r.viable_per_1000:.1f} \\\\")
    rows.append("\\midrule")
open(f"{OUT}/tab_tolerance.tex", "w").write("\\begin{tabular}{@{}llrrrrr@{}}\\toprule\n Arm & Tolerance criterion & Pass & Unique & per 1000 & Qualifying & per 1000\\\\\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n")

with open(f"{OUT}/numbers.tex", "w") as f:
    for k, v in M.items(): f.write("\\newcommand{\\%s}{%s}\n" % (re.sub(r"[^A-Za-z]", "", k.replace("1", "One").replace("4", "Four")), v))
SD = pd.read_csv("results/per_seed_yield.csv"); rows = []
for _, r in SD.iterrows():
    rows.append(f"{'gap-conditioned' if r.arm == 'gap14' else 'unconditional'} & {int(r.seed)}{' (pilot)' if r.seed == 0 else ''} & {int(r.n)} & {int(r.abx3)} & {int(r.perovskite)} & {100 * r.perov_rate + 1e-9:.1f} \\\\")
open(f"{OUT}/tab_seed.tex", "w").write("\\begin{tabular}{@{}lrrrrr@{}}\\toprule\n Mode & Seed & Structures & \\ABX{} & 3D perovskites & (\\%) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print(len(M), "macros")
