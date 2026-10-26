#!/bin/bash
# usage: predict_all.sh UNIQUE_POOL.jsonl.gz OUT_PREFIX   (local CPU inference with the trained random-split ensembles)
for t in bandgap ehull; do
  .venv_alignn/bin/python pipeline/predict.py results/models/random_$t "$1" "$2_$t.csv" 64
done
