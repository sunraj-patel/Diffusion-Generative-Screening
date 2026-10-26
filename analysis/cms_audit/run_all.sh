#!/usr/bin/env bash
# Rebuild every audit output, generated table, figure and both PDFs from the stored data.
# Run from the repository root (Diffusion/). Needs the project Python env (pymatgen, pandas, scikit-learn, matplotlib, spglib)
# and tectonic. Does not touch the cluster. Raw QE outputs must already be in dft_ext/results/.
set -euo pipefail
S=reports/cms_audit/scripts
for s in a1_generation_stats a2_predictor_audit a3_novelty_geometry a4_ranking_sensitivity; do
  python3 $S/$s.py > reports/cms_audit/out/$s.log 2>&1
done
python3 reports/cms_dft_extension/scripts/summarize_dft.py
python3 reports/cms_dft_extension/scripts/make_figures.py
python3 paper_cms/figures_src/make_main_figures.py
python3 $S/make_cms_tables.py
python3 $S/make_concerns_si.py
cd paper_cms
tectonic -X compile main.tex
tectonic -X compile si.tex
