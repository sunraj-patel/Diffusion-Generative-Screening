"""Stage D/E: uncertainty-aware, diversity-aware candidate ranking and simple comparison selectors.
Input DataFrame columns: id, formula, A, Bsite, X, gap, gap_std, ehull, ehull_std, novelty (0..1), unique (bool)."""
import numpy as np, pandas as pd

TARGET, KAPPA = 1.4, 1.0


def risk_adjusted(df, target=TARGET, kappa=KAPPA):
    """Upper-bound style costs: gap error and E_hull penalised by ensemble spread (lower is better)."""
    d = df.copy()
    d["gap_err"] = (d.gap - target).abs()
    d["gap_cost"] = d.gap_err + kappa * d.gap_std
    d["hull_cost"] = d.ehull.clip(lower=0) + kappa * d.ehull_std
    return d


def pareto_rank(costs):
    """Non-dominated sorting on an (n, k) minimisation matrix. Returns front index (0 = Pareto front)."""
    c = np.asarray(costs, float); n = len(c); front = np.full(n, -1); rem = np.arange(n); f = 0
    while len(rem):
        sub = c[rem]
        dom = np.array([np.any(np.all(sub <= sub[i], 1) & np.any(sub < sub[i], 1)) for i in range(len(rem))])
        front[rem[~dom]] = f; rem = rem[dom]; f += 1
    return front


def site_distance(a, b):
    """Chemical-family distance in [0,1]: Jaccard distance averaged over A, B, X element sets (+ anion identity)."""
    def j(x, y):
        x, y = set(x.split(",")) - {""}, set(y.split(",")) - {""}
        return 1 - len(x & y) / len(x | y) if x | y else 0
    return (j(a.A, b.A) + j(a.Bsite, b.Bsite) + j(a.X, b.X)) / 3


def diverse_select(df, k=5, order_col="front_score", min_dist=0.5):
    """Greedy selection down the ranking: skip candidates within min_dist (chemical-family distance) of any chosen one;
    fall back to max-min-distance if fewer than k pass."""
    d = df.sort_values(order_col).reset_index(drop=True); chosen = []
    for i in range(len(d)):
        if all(site_distance(d.iloc[i], d.iloc[j]) >= min_dist for j in chosen): chosen.append(i)
        if len(chosen) == k: break
    if len(chosen) < k:
        rest = [i for i in range(len(d)) if i not in chosen]
        while len(chosen) < k and rest:
            best = max(rest, key=lambda i: min(site_distance(d.iloc[i], d.iloc[j]) for j in chosen) if chosen else 1)
            chosen.append(best); rest.remove(best)
    return d.iloc[chosen]


W = dict(gap=0.5, hull=0.3, novelty=0.2)                                                                             


def proposed_rank(df, target=TARGET, kappa=KAPPA, hull_max=0.10, gap_gate=0.5, min_pool=15):
    """Uncertainty-aware multi-objective ranking.
    1. Gates: unique, predicted E_hull <= hull_max (eV/atom) and risk-adjusted gap error |mu-target| + kappa*sigma <= gap_gate
       (gate doubled until at least min_pool candidates remain, so the rule never returns an empty list).
    2. Objectives (minimise): risk-adjusted gap error, risk-adjusted E_hull (mu + kappa*sigma), -novelty.
    3. Non-dominated sorting; inside a front, candidates are ordered by the weighted mean W of min-max normalised objectives."""
    d = risk_adjusted(df, target, kappa)
    d = d[(d.ehull <= hull_max) & (d["unique"] if "unique" in d else True)]
    g = gap_gate
    while (d.gap_cost <= g).sum() < min(min_pool, len(d)) and g < 50: g *= 2
    d = d[d.gap_cost <= g].copy()
    obj = np.c_[d.gap_cost, d.hull_cost, -d.novelty]
    d["front"] = pareto_rank(obj)
    norm = (obj - obj.min(0)) / np.maximum(np.ptp(obj, 0), 1e-9)
    d["front_score"] = d.front * 10 + norm @ np.array([W["gap"], W["hull"], W["novelty"]])
    return d.sort_values("front_score")


def select(df, method, k=5, rng=None, target=TARGET, hull_max=0.10):
    """Comparison selectors used in the retrospective benchmark and on the generated pool."""
    d = risk_adjusted(df, target)
    if method == "random":
        return d.sample(min(k, len(d)), random_state=rng)
    if method == "gap_only":
        return d.sort_values("gap_err").head(k)
    if method == "stability_aware":
        return d[d.ehull <= hull_max].sort_values("gap_err").head(k)
    if method == "uncertainty_aware":
        return d[d.ehull <= hull_max].sort_values("gap_cost").head(k)
    if method == "proposed":
        return diverse_select(proposed_rank(df, target, hull_max=hull_max), k)
    raise ValueError(method)
