"""ALIGNN (pure-torch) ensemble inference on a structure pool. Runs inside venv_alignn, on a compute node.
usage: python predict.py MEMBER_PREFIX pool.jsonl.gz out.csv [batch]     e.g. models/random_bandgap  ->  models/random_bandgap_s*/"""
import sys, json, gzip, glob, os, shutil, tempfile
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from pymatgen.core import Structure
from jarvis.core.atoms import Atoms
from alignn.pure_lmdb_dataset import get_torch_dataset
from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig


def to_jarvis(s):
    """Same route as training (CIF written by pymatgen, read by jarvis) so that cell setting and graph match what the model saw."""
    with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False) as f:
        f.write(s.to(fmt="cif")); name = f.name
    try:
        return Atoms.from_cif(name)
    finally:
        os.unlink(name)


def build_dataset(records, cfg, tmp):
    return get_torch_dataset(dataset=records, id_tag="jid", target="target", neighbor_strategy="pure_torch",
        atom_features=cfg["atom_features"], use_canonize=cfg["use_canonize"], line_graph=True, cutoff=cfg["cutoff"],
        cutoff_extra=cfg["cutoff_extra"], max_neighbors=cfg["max_neighbors"], three_body_cutoff=cfg.get("three_body_cutoff"), output_dir=tmp, tmp_name=os.path.join(tmp, "pred_lmdb"),
        dtype=cfg["dtype"])


@torch.no_grad()
def predict_member(mdir, data, batch, device):
    cfg = json.load(open(f"{mdir}/config.json"))
    net = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**cfg["model"]))
                                                                                                                                    
    ckpt = f"{mdir}/last_model.pt" if os.path.exists(f"{mdir}/last_model.pt") else f"{mdir}/best_model.pt"
    sd = torch.load(ckpt, map_location=device)
    net.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd); net.to(device).eval()
    dl = DataLoader(data, batch_size=batch, shuffle=False, collate_fn=data.collate_line_graph, num_workers=0)
    out = []
    for g, lg, lat, _ in dl:
        r = net([g.to(device), lg.to(device), lat.to(device)])
        out.append((r["out"] if isinstance(r, dict) else r).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def main():
    prefix, pool, outcsv = sys.argv[1:4]; batch = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = []
    for l in gzip.open(pool, "rt"):
        d = json.loads(l)
        if d.get("structure") is None: continue
        recs.append(dict(jid=d["id"], atoms=to_jarvis(Structure.from_dict(d["structure"])).to_dict(), target=0.0))
    ids = [r["jid"] for r in recs]
    members = [m for m in sorted(glob.glob(prefix + "_s*")) if os.path.exists(f"{m}/last_model.pt") or os.path.exists(f"{m}/best_model.pt")]
    cfg = json.load(open(f"{members[0]}/config.json")); tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
    data = build_dataset(recs, cfg, tmp)
    P = np.stack([predict_member(m, data, batch, device) for m in members])
    df = pd.DataFrame({"id": ids, "mean": P.mean(0), "std": P.std(0, ddof=1) if len(members) > 1 else 0.0})
    for k, m in enumerate(members): df[f"m{k}"] = P[k]
    df.to_csv(outcsv, index=False); shutil.rmtree(tmp, ignore_errors=True); print(len(members), "members;", len(df), "structures")

if __name__ == "__main__":
    main()
