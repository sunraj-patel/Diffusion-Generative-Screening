"""Parses dft_ext/results/<ID>/conv/*.out -> reports/cms_dft_extension/convergence.csv. Run from repo root after syncing results."""
import sys, glob, os, pandas as pd
sys.path.insert(0, "dft/scripts"); from qe_parse import parse
rows = []
for f in sorted(glob.glob("dft_ext/results/*/conv/*.out")):
    r = parse(f); cid = f.split("/")[2]; tag = os.path.basename(f)[:-4]
    rows.append(dict(id=cid, run=tag, ecutwfc=r["ecutwfc_Ry"], ecutrho=r["ecutrho_Ry"], nkpts_irr=r["nkpts"], energy_Ry=r["total_energy_Ry"], gap_eV=r.get("gap_eV"), direct_min_eV=r.get("direct_gap_min_eV"),
        force_Ry_bohr=r["total_force_Ry_bohr"], pressure_kbar=r["pressure_kbar"], scf_converged=r["scf_converged"], job_done=r["job_done"], wall_s=r["wall_time_s"], qe_version=r["qe_version"], nat=r["nat"]))
d = pd.DataFrame(rows)
d["E_meV_per_atom_vs_ref"] = d.groupby("id").apply(lambda g: (g.energy_Ry - g.loc[g.run == "ecut100_k0", "energy_Ry"].values[0]) * 13605.693 / g.nat if (g.run == "ecut100_k0").any() else g.energy_Ry * 0).reset_index(level=0, drop=True)
d.to_csv("reports/cms_dft_extension/convergence.csv", index=False); pd.set_option("display.width", 250); print(d.drop(columns=["qe_version"]))
