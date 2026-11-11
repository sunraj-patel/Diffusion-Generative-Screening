# Decomposition energies against binary phases (PBE, Quantum ESPRESSO)

Variable-cell relaxations (PBE PAW, 80/640 Ry, k-spacing 0.25 1/A, Gaussian smearing 0.002 Ry, spin-unpolarised) of SrTiO3, Cs2NaRhBr6, Cs3Mg2CdBr9 and their binary phases.
`build_hull_inputs.py` writes the inputs, `extract_energies.py` collects the final energies, `decomp_energies.py` writes `decomposition_energies.csv`.
Binary phases only; this is not a convex hull. The CdBr2 energy comes from a relaxation that had not fully converged (layered cell, no dispersion correction), so Cs3Mg2CdBr9 is provisional.
QE 6.8 (sanganak) and QE 7.3.1 gave the same total energies for CsBr, NaBr and Cs2NaRhBr6 (<0.01 meV/atom). Pseudopotentials are not redistributed (PSlibrary 1.0.0, Cd 0.3.1).
