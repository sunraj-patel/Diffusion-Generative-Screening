"""Audit A/B/F: generation yields, denominators, window membership, prototype coverage, shortlist arithmetic. Reads existing saved data only."""
import json, numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint as ci
R_ = lambda x, n: round(float(x), n)
O = "reports/cms_audit/out/"
arms = json.load(open("work/arms.json"))
P = {a: pd.read_csv(f"work/cand/{a}_pool.csv") for a in ("gap14", "base", "gap14g4", "enum", "legacy")}
S = {a: pd.read_csv(f"work/{a}_screen.csv") for a in P}
def w(k, n): lo, hi = ci(k, n, method="wilson"); return f"{k}/{n} = {100*k/max(n,1):.1f}% ({100*lo:.1f}-{100*hi:.1f})"
rows = []
for a, p in P.items():
    N = arms[a]["n_generated"]; s = S[a]
    uniq = p[p.unique & p.perovskite]                                                   
    inwin = ((uniq.gap - 1.4).abs() <= 0.3 + 1e-9)
    multiB = uniq.Bsite.fillna("").str.contains(",")
    rows.append(dict(arm=a, n_attempted_or_built=N, n_screen_rows=len(s), unique_perovskite_structures=len(uniq),
        distinct_formulas=uniq.formula.nunique(), per1000_structures=1000*len(uniq)/N, per1000_formulas=1000*uniq.formula.nunique()/N,
        in_window=int(inwin.sum()), in_window_frac=inwin.mean(), in_window_ci=w(int(inwin.sum()), len(uniq)),
        frac_multiB_double=multiB.mean(), n_multiB=int(multiB.sum()), frac_chalcogenide=uniq.X.isin(["S","Se","Te"]).mean(),
        frac_in_MP=uniq.struct_in_MP.mean(), frac_in_either=(uniq.struct_in_MP | uniq.struct_in_AlexMP20).mean()))
R = pd.DataFrame(rows); R.to_csv(O + "arm_denominators.csv", index=False); print(R.T.to_string())
                                    
b, g, g4 = [P[x][P[x].unique & P[x].perovskite] for x in ("base", "gap14", "gap14g4")]
k = lambda d: int(((d.gap - 1.4).abs() <= .3 + 1e-9).sum())
print("\nWINDOW  base", w(k(b), len(b)), "| gap14", w(k(g), len(g)), "| gap14g4", w(k(g4), len(g4)))
print("Fisher base vs gap14:", fisher_exact([[k(b), len(b)-k(b)], [k(g), len(g)-k(g)]]))
print("gap pred mean/median: base", R_(b.gap.mean(),2), R_(b.gap.median(),2), "gap14", R_(g.gap.mean(),2), R_(g.gap.median(),2), "g4", R_(g4.gap.mean(),2), R_(g4.gap.median(),2))
print("frac predicted gap<0.1: base", (b.gap<.1).mean(), "gap14", (g.gap<.1).mean(), "g4", (g4.gap<.1).mean())
                                                                          
from scipy.stats import beta
k1, n1, k0, n0 = 171, 3500, 12, 1200
lo = beta.ppf(.025, k1, k0+1); hi = beta.ppf(.975, k1+1, k0)
f = lambda x: x/(1-x); r = lambda x: f(x)*n0/n1
print("yield ratio %.2f, exact 95%% CI %.1f-%.1f (base denominator is only %d perovskites)" % ((k1/n1)/(k0/n0), r(lo), r(hi), k0))
                                    
e = P["enum"]; eu = e[e.unique & e.perovskite]
print("\nENUM: unique perovskite 'hits' =", len(eu), "distinct formulas =", eu.formula.nunique(), "prototypes:", eu.source.value_counts().to_dict())
print("ENUM formulas appearing under both prototypes:", int((eu.groupby("formula").source.nunique() == 2).sum()))
print("ENUM multi-B (double) fraction:", eu.Bsite.fillna("").str.contains(",").mean(), "| gen gap14:", g.Bsite.fillna("").str.contains(",").mean())
V = lambda q: q[((q.gap-1.4).abs() <= .3+1e-9) & (q.ehull <= .1) & (q.gap_std <= .3)]
ve = V(e); print("ENUM viable rows", len(ve), "distinct formulas", ve.formula.nunique(), "chalcogenide frac", ve.X.isin(["S","Se","Te"]).mean(), "halide", ve.X.isin(["Cl","Br","I","F"]).mean(), "oxide", (ve.X=="O").mean())
vg = V(P["gap14"]); print("GEN gap14 viable", len(vg), vg.formula.tolist())
print("ENUM acceptance: 4000 built from", 84077, "draws (4.8%)")
                                                                                                                                                          
print("charge_balanced fraction among unique perovskites: gap14", R_(g.charge_balanced.mean(),3), "enum", R_(eu.charge_balanced.mean(),3))
print("tolerance_ok fraction: gap14", R_(g.tolerance_ok.mean(),3), "enum", R_(eu.tolerance_ok.mean(),3))
                                            
sl = pd.read_csv("results/shortlist.csv"); sl["signed_err"] = sl.gap - 1.4; sl["abs_err"] = sl.signed_err.abs()
sl["in_window_unrounded"] = sl.abs_err <= 0.3
print("\nSHORTLIST\n", sl[["candidate","formula","gap","gap_std","signed_err","abs_err","in_window_unrounded","ehull","tier"]].to_string())
print("mean signed err %.3f, MAE %.3f, max ehull %.4f" % (sl.signed_err.mean(), sl.abs_err.mean(), sl.ehull.max()))
sl.to_csv(O + "shortlist_arithmetic.csv", index=False)
       
ps = pd.read_csv("results/per_seed_yield.csv"); print("\n", ps.groupby("arm")[["n","abx3","perovskite"]].sum())
