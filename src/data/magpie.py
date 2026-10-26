"""Magpie-style elemental descriptors (15 features per element).

All features are derived from pymatgen's `Element` data plus electron-configuration
counting, following the spirit of Ward et al. (2016) "A general-purpose machine
learning framework for predicting properties of inorganic materials".
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from pymatgen.core import Element

                                                                
_ORB_RE = re.compile(r"(\d+)([spdf])(\d+)")

FEATURE_NAMES = [
    "atomic_number",
    "mendeleev_number",
    "log_atomic_weight",
    "log_melting_point",
    "row",
    "group",
    "atomic_radius",
    "average_ionic_radius",
    "electronegativity",
    "max_oxidation_state",
    "min_oxidation_state",
    "sp_valence",
    "nd_valence",
    "nf_valence",
    "first_ionization_energy",
]
N_FEATURES = len(FEATURE_NAMES)


def _config_counts(e: Element) -> tuple[int, int, int, int]:
    """Return (ns, np, nd, nf) total electron occupancies from the config string."""
    s = p = d = f = 0
    try:
        cfg = e.electronic_structure
    except Exception:
        cfg = str(e)
    for principal, orbital, count in _ORB_RE.findall(cfg):
        n = int(count)
        if orbital == "s":
            s += n
        elif orbital == "p":
            p += n
        elif orbital == "d":
            d += n
        elif orbital == "f":
            f += n
    return s, p, d, f


@lru_cache(maxsize=None)
def element_features(symbol: str) -> np.ndarray:
    """Return the 15-feature element descriptor (most discriminative properties)."""
    e = Element(symbol)
    ns, np_, nd, nf = _config_counts(e)
    row = e.row

    def _f(v, default=0.0):
        return float(v) if v is not None else float(default)

    feats = [
        e.Z,
        e.mendeleev_no,
        np.log1p(e.atomic_mass),
        np.log1p(_f(e.melting_point)),
        row,
        e.group if e.group is not None else 0,
        _f(e.atomic_radius),
        _f(e.average_ionic_radius),
        _f(e.X),
        _f(e.max_oxidation_state, 0),
        _f(e.min_oxidation_state, 0),
        float(ns + np_),
        float(nd),
        float(nf),
        _f(e.ionization_energy),
    ]
    return np.asarray(feats, dtype=np.float32)


@lru_cache(maxsize=None)
def ionic_radius(symbol: str) -> float:
    r = Element(symbol).average_ionic_radius
    return float(r) if r is not None else 0.0


def site_table(elements: list[str]) -> np.ndarray:
    """Build the descriptor table (n_elements x N_FEATURES) for a list of symbols."""
    return np.stack([element_features(el) for el in elements])
