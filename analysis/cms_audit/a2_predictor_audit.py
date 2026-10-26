"""Audit C: bandgap-predictor reliability. Baselines (zero / train median / composition-RF / composition-kNN), bootstrap CIs,
stratified errors, metal/non-metal and window classification, test-set composition, and coverage of the generated set by held-out B elements."""
import json, warnings, numpy as np, pandas as pd
from collections import Counter
from pymatgen.core import Composition, Element
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings("ignore")
O = "reports/cms_audit/out/"; rng = np.random.default_rng(0)
lab = pd.read_csv("data/curated/labels.csv", keep_default_na=False, na_values=[""])
HELD = ["Be","Cd","Er","Ho","La","Os","Pa","Pu","Re","Rh","Tc","U","Yb","Zn"]
PROPS = ["Z","X","atomic_radius","row","group","mendeleev_no","atomic_mass","ionization_energy","electron_affinity","molar_volume"]
def prop(e, k):
    try:
        v = getattr(Element(e), k); return float(v) if v is not None else np.nan
    except Exception: return np.nan
EL = {e: [prop(e, k) for k in PROPS] for e in {x for f in lab.formula for x in Composition(f).chemical_system.split("-")}}
def feat(f):
    c = Composition(f).fractional_composition; els = [str(e) for e in c]; w = np.array([c[e] for e in c])
    M = np.array([EL[e] for e in els], float); out = []
    for j in range(M.shape[1]):
        v = M[:, j]; ok = ~np.isnan(v)
        if not ok.any(): out += [np.nan]*5; continue
        vv, ww = v[ok], w[ok]/w[ok].sum(); m = (vv*ww).sum()
        out += [m, vv.min(), vv.max(), vv.max()-vv.min(), np.sqrt(((vv-m)**2*ww).sum())]
    return out
ALL = sorted(EL)
def frac(f):
    c = Composition(f).fractional_composition; v = np.zeros(len(ALL))
    for e in c: v[ALL.index(str(e))] = c[e]
    return v
lab["fx"] = [feat(f) for f in lab.formula]; X = np.array(lab.fx.tolist(), float); X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
F = np.array([frac(f) for f in lab.formula])
def boot(y, p, B=2000):
    n = len(y); idx = rng.integers(0, n, (B, n)); e = (y[idx]-p[idx])
    mae = np.abs(e).mean(1); rmse = np.sqrt((e**2).mean(1)); r2 = 1-(e**2).sum(1)/((y[idx]-y[idx].mean(1, keepdims=True))**2).sum(1)
    q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5)); return q(mae), q(rmse), q(r2)
def metr(y, p):
    return dict(MAE=np.abs(y-p).mean(), RMSE=np.sqrt(((y-p)**2).mean()), R2=1-((y-p)**2).sum()/((y-y.mean())**2).sum())
rows = []; cls = []
for split, col, predf in (("random", "split_random", "results/pred_random_bandgap.csv"), ("Bdisjoint", "split_Bdisjoint", "results/pred_Bdisjoint_bandgap.csv")):
    tr = (lab[col] == "train").values; te = lab[lab.id.isin(pd.read_csv(predf).id) & lab.is_perovskite & (lab[col] == "test")].index.values
    pr = pd.read_csv(predf).set_index("id"); pr = pr.loc[lab.loc[te, "id"]]
    y = lab.loc[te, "band_gap"].values; yt = lab.loc[tr, "band_gap"].values
    rf = RandomForestRegressor(300, n_jobs=-1, random_state=0, min_samples_leaf=2).fit(X[tr], yt)
    kn = KNeighborsRegressor(5, metric="cosine", weights="distance").fit(F[tr], yt)
    preds = {"ALIGNN ensemble": pr["mean"].values.clip(0), "ALIGNN ensemble (unclipped)": pr["mean"].values, "constant 0": np.zeros(len(y)),
             "train median": np.full(len(y), np.median(yt)), "train mean": np.full(len(y), yt.mean()),
             "composition RF (Magpie-like, 50 feat)": rf.predict(X[te]), "composition kNN (element fractions)": kn.predict(F[te])}
    for nm, p in preds.items():
        m = metr(y, p); (a, b), (c, d), (e, f) = boot(y, p)
        rows.append(dict(split=split, subset="all perovskite test", n=len(y), predictor=nm, **m, MAE_lo=a, MAE_hi=b, RMSE_lo=c, RMSE_hi=d, R2_lo=e, R2_hi=f))
        for sn, msk in (("label 0.5-3 eV", (y >= .5) & (y <= 3)), ("label > 0 (nonzero)", y > 0), ("label = 0 (metal)", y == 0)):
            if msk.sum() > 3: rows.append(dict(split=split, subset=sn, n=int(msk.sum()), predictor=nm, **metr(y[msk], p[msk])))
                    
    p = preds["ALIGNN ensemble"]; ymet = y < 0.1
    for thr_name, ytrue, ppred in (("metal (<0.1 eV)", ymet, p < .1), ("window 1.1-1.7 eV", (y >= 1.1) & (y <= 1.7), (p >= 1.1) & (p <= 1.7)), (">1 eV gap", y > 1, p > 1)):
        tp, fp, fn, tn = (ytrue & ppred).sum(), (~ytrue & ppred).sum(), (ytrue & ~ppred).sum(), (~ytrue & ~ppred).sum()
        pr_, rc = tp/max(tp+fp, 1), tp/max(tp+fn, 1); cls.append(dict(split=split, task=thr_name, n=len(y), TP=tp, FP=fp, FN=fn, TN=tn, precision=pr_, recall=rc, F1=2*pr_*rc/max(pr_+rc, 1e-9), base_rate=ytrue.mean()))
                          
    t = lab.loc[te]; felem = {e for f in t.formula for e in Composition(f).chemical_system.split("-") if Element(e).is_lanthanoid or Element(e).is_actinoid}
    print(f"\n[{split}] n={len(y)} metal frac(label<0.1)={ymet.mean():.3f}  exact zero={np.mean(y==0):.3f}  f-element-containing={np.mean([any(Element(e).is_lanthanoid or Element(e).is_actinoid for e in Composition(f).chemical_system.split('-')) for f in t.formula]):.3f}")
    err = y - p; print(f"  ALIGNN err: RMSE/MAE={np.sqrt((err**2).mean())/np.abs(err).mean():.2f}; |err|>1 eV: {(np.abs(err)>1).sum()} of {len(y)} ({np.mean(np.abs(err)>1):.3f}); these contribute {(err[np.abs(err)>1]**2).sum()/(err**2).sum():.2f} of squared error; median |err| {np.median(np.abs(err)):.3f}")
R = pd.DataFrame(rows); R.to_csv(O + "predictor_baselines.csv", index=False); C = pd.DataFrame(cls); C.to_csv(O + "predictor_classification.csv", index=False)
pd.set_option("display.width", 250); print(R[R.subset == "all perovskite test"].round(3)[["split","predictor","MAE","MAE_lo","MAE_hi","RMSE","R2","R2_lo","R2_hi"]].to_string()); print(R[R.subset != "all perovskite test"].round(3)[["split","subset","n","predictor","MAE","RMSE"]].query("predictor in ['ALIGNN ensemble','constant 0','train median','composition RF (Magpie-like, 50 feat)','composition kNN (element fractions)']").to_string()); print(C.round(3).to_string())
                                                                                                                                     
g = pd.read_csv("work/cand/gap14_pool.csv"); g = g[g.unique & g.perovskite]
trB = Counter(); 
for s in lab.loc[(lab.split_Bdisjoint == "train")].Bsite.dropna(): trB.update(s.split(","))
allB = Counter()
for s in lab.Bsite.dropna(): allB.update(s.split(","))
genB = Counter()
for s in g.Bsite.dropna(): genB.update(s.split(","))
gen_heldout = [i for i, s in zip(g.formula, g.Bsite.fillna("")) if set(s.split(",")) & set(HELD)]
print("\nGenerated unique perovskites:", len(g), "| with a held-out-14 B element:", len(gen_heldout), gen_heldout[:10])
print("generated B elements:", len(genB), "| absent from the whole curated MP set:", [e for e in genB if e not in allB], "| absent from B-disjoint TRAIN:", [e for e in genB if trB[e] == 0])
print("generated B elements absent from MP perovskite labels entirely:", [e for e in genB if allB[e] == 0])
pd.Series({"n_generated_unique": len(g), "n_with_heldout_B": len(gen_heldout)}).to_json(O + "gen_Bcoverage.json")
