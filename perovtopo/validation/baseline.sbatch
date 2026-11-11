#!/bin/bash
#SBATCH --job-name=perovtopo_base
#SBATCH --partition=small
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
PY=$HOME/perov/venv_alignn/bin/python
cd $HOME/perov/perovtopo_run/perovtopo
$PY validation/baseline_comparison.py $HOME/perov/perovtopo_run/data/pool_gap14.jsonl.gz
