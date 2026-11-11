"""Build out/energies.csv from vc-relax outputs. Final '!' energy of each converged run (last SCF in the final basis set).
Sources: out/sanganak_energies.txt (QE 6.8) and mmpd/*.vcrelax.out (QE 7.3.1). Prefers a converged QE 6.8 value unless --prefer7 given."""
import re, sys, glob, os
FU = {"PRV-11":1,"SrO":1,"TiO2":2,"TiO2_rutile":2,"PRV-07":1,"CsBr":1,"NaBr":1,"RhBr3":2,"CMB-15":1,"MgBr2":1,"CdBr2":2}
NAT = {"PRV-11":5,"SrO":2,"TiO2":6,"TiO2_rutile":6,"PRV-07":10,"CsBr":2,"NaBr":2,"RhBr3":8,"CMB-15":15,"MgBr2":3,"CdBr2":6}
def last(path):
    t = open(path).read()
    if "JOB DONE" not in t or "bfgs converged" not in t: return None
    return float(re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)", t, re.M)[-1])
v68 = {}
for l in open("out/sanganak_energies.txt"):
    p = l.split()
    if len(p) >= 4 and p[1] == "1" and p[3] == "1": v68[p[0]] = float(p[2])
v73 = {os.path.basename(f).split(".")[0]: last(f) for f in glob.glob("mmpd/*.vcrelax.out")}
v73 = {k: v for k, v in v73.items() if v is not None}
# CdBr2: layered PBE relaxation keeps expanding the interlayer gap (no dispersion correction); if no converged run exists, use the last SCF energy of the partial run and flag it
if "CdBr2" not in v73 and os.path.exists("mmpd/CdBr2.partial.out"):
    v73["CdBr2"] = float(re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)", open("mmpd/CdBr2.partial.out").read(), re.M)[-1]); print("NOTE: CdBr2 energy is from an unconverged relaxation (partial run)")
print("QE6.8:", v68); print("QE7.3.1:", v73)
prefer7 = "--prefer7" in sys.argv
rows = []
for p in FU:
    e = (v73.get(p) if prefer7 else None) or v68.get(p) or v73.get(p)
    if e is not None: rows.append((p, e, NAT[p], FU[p]))
open("out/energies.csv", "w").write("phase,E_Ry,natoms,fu\n" + "".join(f"{a},{b:.8f},{c},{d}\n" for a,b,c,d in rows))
for p in v68:
    if p in v73: print(f"{p}: 7.3.1 - 6.8 = {(v73[p]-v68[p])*13605.693/NAT[p]:.2f} meV/atom")
