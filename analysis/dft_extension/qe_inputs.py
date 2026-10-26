"""Writes QE 7.3.1 inputs for the extension (reuses dft/scripts/make_inputs.pw_input so conventions match PRV-01/03).
Usage: qe_inputs.py conv|relax|final <ID> <ecutwfc> <ecutrho> <kspacing_1/A> [outroot]
k-mesh n_i = max(3, ceil(|b_i| / kspacing)) with |b_i| in 1/A including 2*pi, the same rule for every cell, Gamma-centred.
Scalar-relativistic PBE PAW (PSlibrary 1.0.0; Cd 0.3.1), spin-unpolarised, gaussian smearing 0.002 Ry, same thresholds as the PRV-01/03 protocol."""
import sys, json, os, math, numpy as np
sys.path.insert(0, "dft/scripts")
from pymatgen.core import Structure
import make_inputs as mi
PP = dict(mi.PSEUDO_SR)
PP.update({"Sr": "Sr.pbe-spn-kjpaw_psl.1.0.0.UPF", "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF", "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF", "Nb": "Nb.pbe-spn-kjpaw_psl.1.0.0.UPF",
           "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF", "Na": "Na.pbe-spn-kjpaw_psl.1.0.0.UPF", "Rh": "Rh.pbe-spn-kjpaw_psl.1.0.0.UPF", "Br": "Br.pbe-n-kjpaw_psl.1.0.0.UPF",
           "Pb": "Pb.pbe-dn-kjpaw_psl.1.0.0.UPF", "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF", "Cd": "Cd.pbe-dn-kjpaw_psl.0.3.1.UPF"})
mi.MASS.update({"Sr": 87.62, "Ti": 47.867, "K": 39.098, "Nb": 92.906, "O": 15.999, "Na": 22.99, "Rh": 102.906, "Br": 79.904, "Pb": 207.2, "Mg": 24.305, "Cd": 112.414})
PSEUDO_DIR = "/scratch/sunraj/diffusion_ext/pseudo"
VAL = {"Li": 3, "Cl": 7, "Cu": 11, "As": 5, "Cs": 9, "I": 7, "Tl": 13, "Bi": 15, "Sr": 10, "Ti": 12, "K": 9, "Nb": 13, "O": 6, "Na": 9, "Rh": 17, "Br": 7, "Pb": 14, "Mg": 10, "Cd": 12}
def kmesh(s, sp): b = 2 * np.pi * np.linalg.inv(s.lattice.matrix).T; return tuple(max(3, math.ceil(np.linalg.norm(v) / sp)) for v in b)
def load(cid, kind="standardized"): return Structure.from_dict(json.load(open(f"dft_ext/structures/{cid}_{kind}.json")))
def txt(s, calc, prefix, ecut, k, **kw):
    t = mi.pw_input(s, calc, prefix, ecut, k, PP, outdir="./tmp", **kw)
    return t.replace("/scratch/sunraj/diffusion_qe/pseudo", PSEUDO_DIR)
def nel(s): return sum(VAL[x.species_string] for x in s)
if __name__ == "__main__":
    mode, cid, ew, er, ksp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]); root = sys.argv[6] if len(sys.argv) > 6 else "dft_ext/run"
    nb = nel(load(cid)) // 2 + 20
    if mode == "conv":
        s = load(cid)
        for tag, e, kk in (("ecut60_k0", (60, 480), ksp), ("ecut80_k0", (80, 640), ksp), ("ecut100_k0", (100, 800), ksp), ("ecut80_kfine", (80, 640), ksp * 0.75), ("ecut80_kcoarse", (80, 640), ksp * 1.5)):
            k = kmesh(s, kk); mi.write(f"{root}/{cid}/conv/{tag}.in", txt(s, "scf", f"{cid}_{tag}", e, k, nbnd=nb))
    elif mode == "relax":
        s = load(cid); k = kmesh(s, ksp); mi.write(f"{root}/{cid}/relax/vcrelax.in", txt(s, "vc-relax", f"{cid}_relax", (ew, er), k))
    elif mode == "final":
        s = load(cid, "relaxed"); k = kmesh(s, ksp); kn = tuple(n + (n % 2) for n in kmesh(s, ksp * 0.75))                                                                         
        mi.write(f"{root}/{cid}/final/scf.in", txt(s, "scf", f"{cid}_fin", (ew, er), k, nbnd=nb))
        mi.write(f"{root}/{cid}/final/nscf.in", txt(s, "nscf", f"{cid}_fin", (ew, er), kn, nbnd=nb))
