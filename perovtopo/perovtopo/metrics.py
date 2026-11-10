"""Site-resolved metrics and the topology funnel for a set of generated structures."""
from __future__ import annotations
from collections import Counter
from .classifier import classify, abx3_stoichiometry, PEROVSKITE, DEFAULT


def b_site_multiplicity(result: dict) -> int:
    """Number of distinct elements on the octahedral B sublattice (site-resolved, not total element count)."""
    return len(result["B_sites"] or [])


def funnel(structures, params=DEFAULT):
    """Counts at each stage: total, stoichiometry-compatible, topology-positive, plus family breakdown and B multiplicity."""
    n = 0; stoich = 0; fam = Counter(); mult = Counter(); results = []
    for s in structures:
        n += 1
        r = classify(s, params); results.append(r)
        if r["abx3"]:
            stoich += 1
        fam[r["family"]] += 1
        if r["family"] == PEROVSKITE:
            mult[b_site_multiplicity(r)] += 1
    topo = fam[PEROVSKITE]
    return dict(n=n, stoichiometry=stoich, topology=topo,
                topology_given_stoichiometry=topo / stoich if stoich else float("nan"),
                families=dict(fam), b_multiplicity=dict(sorted(mult.items())),
                mixed_b_fraction=sum(v for k, v in mult.items() if k > 1) / topo if topo else float("nan")), results
