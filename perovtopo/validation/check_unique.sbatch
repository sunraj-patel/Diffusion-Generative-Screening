#!/bin/bash
#SBATCH --job-name=perovtopo_u171
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
cd "${SLURM_SUBMIT_DIR:?}"
$HOME/perov/venv_alignn/bin/python validation/check_unique.py $HOME/perov/perovtopo_run/data/gap14_unique.jsonl.gz > validation/check_unique_gap14.txt 2>&1
