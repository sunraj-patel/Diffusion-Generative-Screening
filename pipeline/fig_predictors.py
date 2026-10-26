"""Figure: predictor reliability on held-out MP structures (random and B-site-disjoint)."""
import json, sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
import style as S
S.apply()
M = json.load(open("results/predictor_metrics.json"))
fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.9), constrained_layout=True)
def parity(a, split, tgt, title, lim, unit, sub):
    d = pd.read_csv(f"results/pred_{split}_{tgt}.csv"); d = d[d.is_perovskite]
    m = M[f"{split}/{tgt}/perovskite_test"]
    sc = a.scatter(d.y, d["mean"], c=np.log10(d["std"].clip(lower=1e-3)), s=7, cmap="viridis", vmin=-2.5, vmax=0, lw=0, alpha=.85)
    a.plot(lim, lim, color=S.INK, lw=.8, ls="--"); a.set_xlim(lim); a.set_ylim(lim)
    a.text(.04, .95, f"n={m['n']:.0f}\nMAE {m['mae']:.2f} {unit}\nR$^2$ {m['r2']:.2f}\nbaseline MAE {m['baseline_mae']:.2f}", transform=a.transAxes, va="top", fontsize=7.5)
    a.set_title(f"({sub}) {title}"); a.set_xlabel(f"MP label ({unit})"); a.set_ylabel(f"ALIGNN ensemble ({unit})"); S.finish(a, "both"); return sc
sc = parity(ax[0, 0], "random", "bandgap", "Gap, random split", (-0.3, 8), "eV", "a")
parity(ax[0, 1], "Bdisjoint", "bandgap", "Gap, B-site disjoint", (-0.3, 8), "eV", "b")
parity(ax[0, 2], "random", "ehull", r"$E_{\rm hull}$, random split", (-0.03, 1.0), "eV/atom", "c")
parity(ax[1, 0], "Bdisjoint", "ehull", r"$E_{\rm hull}$, B-site disjoint", (-0.03, 1.0), "eV/atom", "d")
cb = fig.colorbar(sc, ax=ax[0, :], shrink=.6, location="right", pad=0.01); cb.set_label(r"$\log_{10}\sigma$ (ensemble std)")
                                   
a = ax[1, 1]
for (split, tgt, col, lab) in (("random", "bandgap", S.BLUE, "gap, random"), ("Bdisjoint", "bandgap", S.CLAY, "gap, B-disjoint")):
    u = M[f"{split}/{tgt}/perovskite_test"]["uncertainty"]; b = u["bins"]
    a.plot([x["mean_std"] for x in b], [x["mae"] for x in b], "o-", color=col, label=f"{lab} ($\\rho$={u['spearman_std_abs_err']:.2f})")
a.set_xscale("log"); a.set_xlabel("mean ensemble std in quartile (eV)"); a.set_ylabel("MAE in quartile (eV)"); a.set_title("(e) Uncertainty vs error"); a.legend(fontsize=7); S.finish(a, "both")
              
a = ax[1, 2]; nom = [.5, .8, .95]
for (split, col, lab) in (("random", S.BLUE, "random"), ("Bdisjoint", S.CLAY, "B-disjoint")):
    cov = M[f"{split}/bandgap/perovskite_test"]["uncertainty"]["gaussian_coverage"]
    a.plot(nom, [cov[k] for k in ("50%", "80%", "95%")], "o-", color=col, label=lab)
a.plot([0.4, 1], [0.4, 1], color=S.INK, ls="--", lw=.8); a.set_xlabel("nominal Gaussian coverage"); a.set_ylabel("observed coverage"); a.set_title("(f) Not calibrated"); a.legend(fontsize=7); S.finish(a, "both")
fig.savefig("paper/figures/fig_predictors.pdf"); fig.savefig("paper/figures/fig_predictors.png", dpi=300); print("ok")
