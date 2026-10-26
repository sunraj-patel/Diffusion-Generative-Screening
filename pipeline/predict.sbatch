#!/bin/bash
# usage: POOL=pool.jsonl.gz OUT=prefix sbatch --export=ALL pipeline/predict.sbatch  (CPU is enough for a few thousand structures)
#SBATCH --job-name=alignn_pred
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pred_%j.out
cd ~/perov
for t in bandgap ehull; do
  ~/perov/venv_alignn/bin/python ~/perov/predict.py models/random_$t $POOL ${OUT}_$t.csv 64
done
