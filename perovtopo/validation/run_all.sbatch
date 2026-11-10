#!/bin/bash
# Rebuilds the validation set, scores the classifier, and applies it to the generated pools. Submit from perovtopo/:  sbatch validation/run_all.sbatch
#SBATCH --job-name=perovtopo_val
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
PY=${PY:-$HOME/perov/venv_alignn/bin/python}
DATA=${DATA:-$HOME/perov/perovtopo_run/data}
cd "${SLURM_SUBMIT_DIR:?submit from perovtopo/ with sbatch validation/run_all.sbatch}"
$PY validation/build_validation_set.py $DATA validation
$PY validation/run_validation.py > validation/run_log.txt 2>&1
$PY validation/apply_to_generated.py $DATA/pool_gap14.jsonl.gz $DATA/pool_base.jsonl.gz > validation/apply_log.txt 2>&1
$PY -m pytest -q tests > validation/pytest_log.txt 2>&1 || true
