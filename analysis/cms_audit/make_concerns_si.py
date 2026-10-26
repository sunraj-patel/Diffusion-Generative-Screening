"""SI concerns table with the lone-pair flag split into informational (established B-site lone-pair cations: Pb2+, Sn2+, Bi3+, Sb3+, Ge2+) and
concern (As3+, Tl+, Se4+, Te4+). Toxicity is a post-screening warning; it does not enter the ranking."""
import pandas as pd, re
D = pd.read_csv("results/candidate_concerns.csv", keep_default_na=False)
INFO = {"Pb", "Sn", "Bi", "Sb", "Ge"}
fm = lambda f: re.sub(r"(\d+)", r"$_{\1}$", f)
rows = []
for _, r in D.iterrows():
    lp = [x for x in r.lone_pair_on_B.split(",") if x]
    info = [x for x in lp if re.match(r"[A-Z][a-z]?", x).group(0) in INFO]; conc = [x for x in lp if x not in info]
    flags = []
    if r.toxic: flags.append("toxic: " + r.toxic.replace("/", ", "))
    if conc: flags.append("lone pair on B: " + ", ".join(conc))
    if r.large_monovalent_on_B: flags.append("monovalent " + r.large_monovalent_on_B + " on B")
    if r.oxidation_not_unique in (True, "True"): flags.append("ox. states ambiguous")
    if r.tolerance_not_assessable in (True, "True"): flags.append("$t$ n/a")
    if r.spread_gt_0p2 in (True, "True"): flags.append("$\\sigma_g>0.2$")
    n = len(flags) - (1 if r.toxic else 0)
    note = ("; B-site " + ", ".join(info) + " established in halide perovskites") if info else ""
    rows.append(f"{r.candidate} & {fm(r.formula)} & {r.gap:.2f} & {'; '.join(flags) if flags else '--'}{note} & {n} \\\\")
open("paper_cms/generated/cms_tab_concerns.tex", "w").write("\\begin{tabular}{@{}llrp{8cm}r@{}}\\toprule\nID & Formula & $\\hat E_{\\rm g}$ & Flags & $n$ (excl. toxicity)\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
