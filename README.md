# Diffusion-Generative-Screening

Code, tables and DFT inputs for the manuscript on perovskite generation with the pretrained MatterGen bandgap model,
ALIGNN property prediction and Quantum ESPRESSO (PBE) checks (Computational Materials Science submission).

| Directory | Contents |
|---|---|
| `perovtopo/` | Python package: connectivity-based classifier for 3D corner-sharing ABX3 perovskites, site-resolved metrics, transfer splits, tests, and the classifier validation and threshold-sensitivity scripts and outputs |
| `analysis/` | Scripts that regenerate the numbers in the manuscript |
| `tables/` | Generated tables |
| `figures_src/` | Figure source data and scripts |
| `dft/` | Quantum ESPRESSO inputs and outputs |

Install the classifier with `pip install -e perovtopo` and run its tests with `pytest perovtopo/tests`.
Details, validation results and caveats are in `perovtopo/README.md` and `perovtopo/validation/README.md`.

The Materials Project structures in `perovtopo/validation/structures.jsonl.gz` come from the Materials Project database
(CC BY 4.0, snapshot 2026.04.13). Third-party data are not redistributed beyond this subset. Code is released under the MIT licence (`LICENSE`). Derived data tables and generated-structure summaries in this repository are shared under CC BY 4.0, with attribution to this repository and to the Materials Project for the structure subset; the Materials Project data keep their own CC BY 4.0 terms.
