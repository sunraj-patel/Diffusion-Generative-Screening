"""Rattled-supercell symmetry-breaking test. Usage: make_supercell_test.py <ID> <nx> <ny> <nz> <amp_A> <seed> <ecutwfc> <ecutrho> <kspacing>
Builds the nx*ny*nz supercell of the RELAXED primitive cell (dft_ext/structures/<ID>_relaxed.json) and writes (a) an unperturbed SCF reference and (b) a fixed-cell 'relax' from
Gaussian-rattled positions (sigma = amp, fixed seed). The supercell is a probe of instabilities that the primitive cell cannot represent, NOT a larger generated structure."""
import sys, json, numpy as np
sys.path.insert(0, "reports/cms_dft_extension/scripts"); import qe_inputs as q
from pymatgen.core import Structure
cid, nx, ny, nz, amp, seed, ew, er, ksp = sys.argv[1], *map(int, sys.argv[2:5]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9])
s = q.load(cid, "relaxed"); s.make_supercell([nx, ny, nz]); rng = np.random.default_rng(seed)
r = s.copy(); disp = rng.normal(0, amp, (len(s), 3)); r = Structure(s.lattice, s.species, s.cart_coords + disp, coords_are_cartesian=True)
tag = f"{cid}_sc{nx}{ny}{nz}"; k = q.kmesh(s, ksp); print(tag, len(s), "atoms, k =", k, "electrons", q.nel(s))
json.dump(s.as_dict(), open(f"dft_ext/structures/{tag}_perfect.json", "w")); json.dump(r.as_dict(), open(f"dft_ext/structures/{tag}_rattled.json", "w"))
q.mi.write(f"dft_ext/run/{tag}/scf_perfect/scf.in", q.txt(s, "scf", tag + "_p", (ew, er), k))
q.mi.write(f"dft_ext/run/{tag}/relax_rattled/relax.in", q.txt(r, "relax", tag + "_r", (ew, er), k))
