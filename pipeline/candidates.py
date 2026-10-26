"""Stage C/D/E driver: merge screening records, ALIGNN ensemble predictions and novelty measures for a screened pool,
then rank (pipeline/rank.py), and export the candidate package.
usage: python candidates.py TAG screen.csv unique.jsonl.gz gap_pred.csv ehull_pred.csv OUTDIR"""
import gzip, json, os, sys
import numpy as np, pandas as pd
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sys.path.insert(0, os.path.dirname(__file__))
import rank
from crystal import make_matcher

ELS = [Element.from_Z(z).symbol for z in range(1, 104)]


def vec(formula):
    c = Composition(formula).fractional_composition; v = np.zeros(len(ELS))
    for e, a in c.items(): v[ELS.index(e.symbol)] = a
    return v


def reference_vectors():
    lab = pd.read_csv("data/curated/labels.csv"); mp = lab[lab.is_perovskite]
    al = pd.read_csv("data/curated/alex_abx3_labels.csv"); al = al[al.family == "3D corner-sharing perovskite"]
    return mp.formula.unique().tolist(), al.formula.unique().tolist(), json.load(open("data/curated/alex_all_formulas.json"))


def novelty_features(df, structs, mp_f, alex_f, alex_all):
    Rmp = np.array([vec(f) for f in mp_f]); Ral = np.array([vec(f) for f in alex_f]); R = np.vstack([Rmp, Ral])
    mpset, alset, allset = set(mp_f), set(alex_f), set(alex_all)
                                                                                                 
    mp_s, al_s = {}, {}
    lab = pd.read_csv("data/curated/labels.csv"); mpid = set(lab.id[lab.is_perovskite])
    need = set(df.formula)
    for l in gzip.open("data/curated/structures.jsonl.gz", "rt"):
        d = json.loads(l)
        if d["id"] in mpid:
            s = Structure.from_dict(d["structure"])
            if s.composition.reduced_formula in need: mp_s.setdefault(s.composition.reduced_formula, []).append(s)
    for l in gzip.open("data/curated/alex_perovskites.jsonl.gz", "rt"):
        d = json.loads(l); f = d["structure"]
        s = Structure.from_dict(f)
        if s.composition.reduced_formula in need: al_s.setdefault(s.composition.reduced_formula, []).append(s)
    m = make_matcher(); out = []
    for _, r in df.iterrows():
        v = vec(r.formula); nn = 0.5 * np.abs(R - v).sum(1).min()
        s = structs[r.id]
        out.append(dict(comp_nn_dist=nn, formula_in_MP_perovskites=r.formula in mpset, formula_in_AlexMP20_train=r.formula in allset,
                        struct_in_MP=any(m.fit(s, c) for c in mp_s.get(r.formula, [])),
                        struct_in_AlexMP20=any(m.fit(s, c) for c in al_s.get(r.formula, []))))
    o = pd.DataFrame(out, index=df.index)
                                                                                                                              
                                                                                                                        
    o["novelty"] = (o.comp_nn_dist.rank(pct=True) + (~o.struct_in_MP) + (~o.struct_in_AlexMP20)) / 3
    return pd.concat([df, o], axis=1)


def export(sel, structs, outdir, tag):
    os.makedirs(f"{outdir}/structures", exist_ok=True); rows = []
    for k, (_, r) in enumerate(sel.iterrows(), 1):
        cid = f"{tag}-{k:02d}"; s = structs[r.id]
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        s.to(filename=f"{outdir}/structures/{cid}.cif"); Poscar(s).write_file(f"{outdir}/structures/{cid}.POSCAR")
        rows.append(dict(candidate=cid, sample_id=r.id, spacegroup=sga.get_space_group_symbol(), sg_number=sga.get_space_group_number(),
                         a=s.lattice.a, b=s.lattice.b, c=s.lattice.c, alpha=s.lattice.alpha, beta=s.lattice.beta, gamma=s.lattice.gamma,
                         volume_per_atom=s.volume / len(s)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tag, scr, uniq, gapcsv, hullcsv, outdir = sys.argv[1:7]
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(scr); df = df[df.unique].copy()
    structs = {}
    for l in gzip.open(uniq, "rt"):
        d = json.loads(l); structs[d["id"]] = Structure.from_dict(d["structure"])
    g = pd.read_csv(gapcsv).set_index("id"); h = pd.read_csv(hullcsv).set_index("id")
    df["gap"] = df.id.map(g["mean"]); df["gap_std"] = df.id.map(g["std"]); df["ehull"] = df.id.map(h["mean"]).clip(lower=0); df["ehull_std"] = df.id.map(h["std"])
    df = df.dropna(subset=["gap", "ehull"])
    df = novelty_features(df, structs, *reference_vectors())
    df.to_csv(f"{outdir}/{tag}_pool.csv", index=False)
    print(len(df), "scored unique perovskites")
