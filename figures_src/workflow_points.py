"""Writes workflow_points.tex: the real risk-adjusted costs of the 171 unique generated perovskites (for the 'Rank' panel of workflow.tex)."""
import pandas as pd, numpy as np
ROOT = "../.."
d = pd.read_csv(f"{ROOT}/work/cand/gap14_pool.csv"); d = d[d.unique.astype(bool)]
sl = pd.read_csv(f"{ROOT}/results/shortlist.csv").fillna(""); sl = sl[sl.candidate != ""]
cg = (d.gap - 1.4).abs() + d.gap_std; ch = d.ehull.clip(lower=0) + d.ehull_std
XM, YM = 2.0, 0.16
def pt(a, b): return f"{min(a, XM) / XM:.3f}/{min(b, YM) / YM:.3f}"
gate = (d.ehull <= 0.10) & (cg <= 0.6)
sid = set(sl.id)
grey = [pt(a, b) for a, b, g, i in zip(cg, ch, gate, d.id) if not g and i not in sid]
plum = [pt(a, b) for a, b, g, i in zip(cg, ch, gate, d.id) if g and i not in sid]
star = [pt(a, b) for a, b, i in zip(cg, ch, d.id) if i in sid]
assert len(star) == 5 and len(grey) + len(plum) + 5 == 171
open("workflow_points.tex", "w").write("\\newcommand{\\ptsGrey}{%s}\n\\newcommand{\\ptsPlum}{%s}\n\\newcommand{\\ptsStar}{%s}\n" % (",".join(grey), ",".join(plum), ",".join(star)))
print(len(grey), len(plum), len(star))
