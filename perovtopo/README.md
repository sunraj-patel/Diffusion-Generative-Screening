# perovtopo

Connectivity-based identification of three-dimensional corner-sharing ABX3 perovskites in sets of generated or database
crystal structures, with site-resolved B-site metrics, a face/edge-sharing check and a threshold-sensitivity harness.

This is the classifier used in the manuscript, with the thresholds exposed as parameters. The defaults reproduce the
manuscript's counts exactly. The thresholds are operational definitions, not universal ones; see `validation/README.md`.

```python
from pymatgen.core import Structure
from perovtopo import classify, Params, funnel

s = Structure.from_file("candidate.cif")
r = classify(s)                         # {'family': '3D corner-sharing perovskite', 'B_sites': [...], 'A_sites': [...], ...}
r = classify(s, Params(strict_corner=True))   # also reject face/edge-sharing octahedra
summary, per_structure = funnel(list_of_structures)   # stoichiometry -> topology counts and B-site multiplicity
```

Command line: `python -m perovtopo classify "structs/*.cif" --csv out.csv [--a-tol 1.3 --a-min-cn 8 ...]`

**What it does**
1. Stoichiometry: ordered, at least two cation species, anion:cation = 3:2.
2. B sites: cations with six anions within `b_tol` x their shortest cation-anion distance.
3. A sites: remaining cations need at least `a_min_cn` anions within `a_tol` x their shortest distance.
4. Every anion must bridge exactly two B cations and the B-X network must be periodic in three dimensions.
5. Optional `strict_corner`: no pair of B cations may share more than one anion.

`perovtopo.metrics.funnel` gives the stoichiometry/topology counts and the site-resolved B multiplicity (distinct elements on the
octahedral sublattice, not total element count). `perovtopo.transfer` provides a B-site-disjoint split and a constant-baseline
comparison for property predictors.

**Install and test:** `pip install -e . && pytest tests`

**Limitations.** Strongly off-centred B cations (lone-pair Ge2+, Sn2+) and small, strongly tilted A cations can be rejected;
mixed cubic/hexagonal polytypes are accepted unless `strict_corner` is set. Classification is on the supplied (unrelaxed or
relaxed) geometry; relaxation can change the class.
