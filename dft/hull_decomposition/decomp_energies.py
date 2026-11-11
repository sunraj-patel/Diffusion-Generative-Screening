"""Decomposition energies from the relaxed PBE total energies in out/energies.csv (phase, E_Ry, natoms, formula_units).
dE = E(compound) - sum(E(products)), per formula unit and per atom, eV. Negative = stable against that route.
Binary phases only; this is not a convex hull."""
import csv, sys
RY = 13.605693123
E = {}
for r in csv.DictReader(open("out/energies.csv")):
    E[r["phase"]] = float(r["E_Ry"]) / int(r["fu"])          # Ry per formula unit
routes = [
 ("SrTiO3 (PRV-11)", "PRV-11", 5, [("SrO",1),("TiO2",1)]),
 ("SrTiO3 (PRV-11), rutile", "PRV-11", 5, [("SrO",1),("TiO2_rutile",1)]),
 ("Cs2NaRhBr6 (PRV-07)", "PRV-07", 10, [("CsBr",2),("NaBr",1),("RhBr3",1)]),
 ("Cs3Mg2CdBr9 (CMB-15)", "CMB-15", 15, [("CsBr",3),("MgBr2",2),("CdBr2",1)]),
]
rows = []
for name, c, n, prods in routes:
    if c not in E or any(p not in E for p,_ in prods): print("skip (missing):", name); continue
    d = (E[c] - sum(k*E[p] for p,k in prods)) * RY
    rows.append((name, " + ".join(f"{k}{p}" for p,k in prods), round(d,4), round(d/n,4)))
w = csv.writer(open("out/decomposition_energies.csv","w")); w.writerow(["compound","products","dE_eV_per_fu","dE_eV_per_atom"]); w.writerows(rows)
for r in rows: print(r)
