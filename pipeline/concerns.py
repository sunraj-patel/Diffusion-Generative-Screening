"""Heuristic chemical-concern flags for the ranked candidates (exploratory; does NOT change the shortlist).
Flags are conventions of inorganic chemistry, not calculations: toxic elements, lone-pair cations on an octahedral B site,
large monovalent cations on a B site, non-unique oxidation-state assignment, unassessable tolerance factor, large ensemble spread."""
import os, sys
import numpy as np, pandas as pd
from pymatgen.core import Composition
T = pd.read_csv("results/ranked_table.csv").fillna("")
TOX = {"Tl", "Pb", "Cd", "Hg", "As"}
LONE = {("As", 3), ("Sb", 3), ("Bi", 3), ("Tl", 1), ("Pb", 2), ("Sn", 2), ("Te", 4), ("Se", 4), ("Ge", 2)}
LARGE_1 = {"K", "Rb", "Cs", "Tl"}
rows = []
for _, r in T.iterrows():
    comp = Composition(r.formula); g = comp.oxi_state_guesses(max_sites=-1)
    ox = g[0] if g else {}
    B = [e for e in r.Bsite.split(",") if e]; A = [e for e in r.A.split(",") if e]
    tox = sorted(set(A + B + [r.X]) & TOX)
    lone = [f"{e}{ox[e]:+.0f}" for e in B if e in ox and (e, round(ox[e])) in LONE]
    big = [e for e in B if e in LARGE_1 and ox.get(e, 0) > 0 and round(ox[e]) == 1]
    nonunique = len({tuple(sorted((k, round(v, 2)) for k, v in x.items())) for x in g}) > 1
    sigma = r.gap_std > 0.2; notol = not (isinstance(r.tolerance_factor, (int, float)) and r.tolerance_factor == r.tolerance_factor and r.tolerance_factor != "")
    flags = dict(toxic="/".join(tox), lone_pair_on_B=",".join(lone), large_monovalent_on_B="/".join(big), oxidation_not_unique=nonunique, tolerance_not_assessable=notol, spread_gt_0p2=bool(sigma))
    n = bool(tox) + bool(lone) + bool(big) + nonunique + notol + sigma
    rows.append(dict(candidate=r.candidate, formula=r.formula, tier=r.tier, shortlist=r.shortlist, gap=r.gap, gap_std=r.gap_std, ehull=r.ehull, oxidation_guess=str({k: round(v, 1) for k, v in ox.items()}), **flags, n_concern_flags=n))
D = pd.DataFrame(rows); D.to_csv("results/candidate_concerns.csv", index=False)
print(D.drop(columns=["oxidation_guess"]).to_string())
