"""Write ALIGNN input folders (shared CIF dir + per-run id_prop.csv ordered train/val/test) under an output dir."""
import gzip, json, os, sys
import pandas as pd
from pymatgen.core import Structure

out = sys.argv[1]; os.makedirs(f"{out}/cifs", exist_ok=True)
df = pd.read_csv("data/curated/labels.csv")
with gzip.open("data/curated/structures.jsonl.gz", "rt") as f:
    for l in f:
        d = json.loads(l); Structure.from_dict(d["structure"]).to(filename=f"{out}/cifs/{d['id']}.cif")
df["ehull_c"] = df.ehull.clip(upper=1.0)
for split in ("random", "Bdisjoint"):
    col = f"split_{split}"
    for tgt, c in (("bandgap", "band_gap"), ("ehull", "ehull_c")):
        d = f"{out}/runs/{split}_{tgt}"; os.makedirs(d, exist_ok=True)
        parts = [df[df[col] == k].sample(frac=1, random_state=0) for k in ("train", "val", "test")]
        pd.concat(parts)[["id", c]].assign(id=lambda x: "../../cifs/" + x.id + ".cif").to_csv(f"{d}/id_prop.csv", index=False, header=False)
        json.dump([len(p) for p in parts], open(f"{d}/sizes.json", "w"))
        print(d, [len(p) for p in parts])
