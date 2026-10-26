#!/bin/bash
# Local (laptop) part of the pipeline after generation pools have been copied to work/gen/*/pool.jsonl.gz.
# usage: pipeline/run_local.sh      (idempotent; ~10-20 min on an M3 Pro)
set -e
cd "$(dirname "$0")/.."
export MP_API_KEY="${MP_API_KEY:-unused-here}"
arm() {   # arm NAME pool.jsonl.gz
  python3 pipeline/screen.py "$2" work/$1 > work/$1_screen.log 2>&1
  ./pipeline/predict_all.sh work/$1_unique.jsonl.gz work/pred_$1 > work/pred_$1.log 2>&1
  python3 pipeline/candidates.py $1 work/$1_screen.csv work/$1_unique.jsonl.gz work/pred_$1_bandgap.csv work/pred_$1_ehull.csv work/cand
}
cat work/gen/pilot_cond14_s0/pool.jsonl.gz work/gen/cond14_s[0-9]/pool.jsonl.gz > work/pool_gap14.jsonl.gz
cat work/gen/pilot_base_s0/pool.jsonl.gz work/gen/base_s[0-9]/pool.jsonl.gz > work/pool_base.jsonl.gz
arm gap14 work/pool_gap14.jsonl.gz
arm base work/pool_base.jsonl.gz
arm gap14g4 work/gen/gap14g4_s4/pool.jsonl.gz
python3 pipeline/enumerate_baseline.py 4000 0 work/pool_enum.jsonl.gz > work/enum.log 2>&1
python3 pipeline/enumerate_baseline.py 5000 0 work/pool_legacy.jsonl.gz work/legacy_compositions.csv > work/legacy.log 2>&1
arm enum work/pool_enum.jsonl.gz
arm legacy work/pool_legacy.jsonl.gz
python3 pipeline/write_arms.py
python3 pipeline/make_results.py
python3 pipeline/finalize.py gap14 work/gap14_unique.jsonl.gz
