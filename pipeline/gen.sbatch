#!/bin/bash
# usage: sbatch --export=ALL,TAG=cond14,MODEL=dft_band_gap,PROPS="{'dft_band_gap': 1.4}",GUID=2.0,NB=20,BS=100,SEED=0 pipeline/gen.sbatch
#SBATCH --job-name=mgen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gen_%x_%j.out
set -e
cd ~/perov; source venv_mattergen/bin/activate
module load mldl/cuda/cuda-12.9 2>/dev/null || true
OUT=gen/${TAG}_s${SEED}; mkdir -p $OUT logs
export HF_HUB_OFFLINE=0 PYTHONPATH=$HOME/perov:$PYTHONPATH
P="$PROPS"; [ -z "$P" ] && P="{}"
PJ=$(python -c "import ast,json,sys; print(json.dumps(ast.literal_eval(sys.argv[1])))" "$P")
OJ=${OVR:-[]}
python ~/perov/mg_generate.py $OUT ckpt/$MODEL $BS $NB $SEED ${GUID:-0} "$PJ" "$OJ"
python - <<PY
import zipfile, json, gzip
from pymatgen.core import Structure
z = zipfile.ZipFile("$OUT/generated_crystals_cif.zip"); n = 0
with gzip.open("$OUT/pool.jsonl.gz", "wt") as f:
    for name in sorted(z.namelist()):
        try: s = Structure.from_str(z.read(name).decode(), fmt="cif"); d = s.as_dict()
        except Exception: d = None
        f.write(json.dumps({"id": "${TAG}_s${SEED}_" + name.replace(".cif", ""), "source": "mattergen:$MODEL", "structure": d}) + "\n"); n += 1
print("pool size", n)
PY
