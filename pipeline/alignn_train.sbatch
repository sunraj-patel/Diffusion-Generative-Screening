#!/bin/bash
# usage: sbatch --export=RUN=random_bandgap,SEED=0,EPOCHS=40 pipeline/alignn_train.sbatch   (run on the cluster from ~/perov)
#SBATCH --job-name=alignn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=logs/alignn_%x_%j.out
set -e
cd ~/perov
module load mldl/cuda/cuda-12.9 2>/dev/null || true
OUT=models/${RUN}_s${SEED}; mkdir -p $OUT logs
python3 - <<PY
import json
c = json.load(open("/home/sunrajp23/gnn/config.json"))
n = json.load(open("alignn_in/rundirs/${RUN}/sizes.json"))
c["model"]["name"] = "alignn_atomwise_pure"
c.update(dict(neighbor_strategy="pure_torch", random_seed=${SEED}, epochs=${EPOCHS:-40}, batch_size=64, learning_rate=0.001, keep_data_order=True,
              n_train=n[0], n_val=n[1], n_test=n[2], train_ratio=None, val_ratio=None, test_ratio=None,
              use_lmdb=True, num_workers=4, n_early_stopping=None, write_predictions=True, output_dir="$OUT",
              target="target", filename="${RUN}_s${SEED}"))
json.dump(c, open("$OUT/config.json", "w"), indent=1)
PY
~/perov/venv_alignn/bin/python -m alignn.train_alignn --root_dir alignn_in/rundirs/${RUN} --config_name $OUT/config.json \
   --file_format cif --output_dir $OUT --id_key jid --target_key target
