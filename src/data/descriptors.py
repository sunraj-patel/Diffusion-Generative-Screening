"""Composition -> descriptor vector x0 construction.

x0 = concat[ Magpie(A) | Magpie(B) | Magpie(X) | structural(A, B, X) ]

Structural descriptors (10) are fully composition-derived so the same pipeline
works both for MP entries and for decoded (generated) candidates:
  1.  Goldschmidt tolerance factor t = (rA + rX) / (sqrt(2) * (rB + rX))
  2.  Octahedral factor rB / rX
  3.  Ionic radius sum rA + rB + 3*rX
  4.  Atomic radius sum rA + rB + 3*rX
  5.  |EN(A) - EN(X)|
  6.  |EN(B) - EN(X)|
  7.  |EN(A) - EN(B)|
  8.  |row(A) - row(B)|
  9.  |row(B) - row(X)|
  10. Mendeleev-number gap |M(A) - M(B)|
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Element

from src.data.magpie import N_FEATURES, element_features, ionic_radius
from src.config import DiffusionConfig

N_STRUCTURAL = 10


def tolerance_factor(a: str, b: str, x: str) -> float:
    """Goldschmidt tolerance factor using average ionic radii (heuristic)."""
    ra, rb, rx = ionic_radius(a), ionic_radius(b), ionic_radius(x)
    if rb + rx <= 0:
        return 0.0
    return (ra + rx) / (np.sqrt(2.0) * (rb + rx))


def octahedral_factor(b: str, x: str) -> float:
    rb, rx = ionic_radius(b), ionic_radius(x)
    return rb / rx if rx > 0 else 0.0


def structural_features(a: str, b: str, x: str) -> np.ndarray:
    ea, eb, ex = Element(a), Element(b), Element(x)
    return np.asarray(
        [
            tolerance_factor(a, b, x),
            octahedral_factor(b, x),
            ionic_radius(a) + ionic_radius(b) + 3.0 * ionic_radius(x),
            (ea.atomic_radius or 0.0) + (eb.atomic_radius or 0.0) + 3.0 * (ex.atomic_radius or 0.0),
            abs((ea.X or 0.0) - (ex.X or 0.0)),
            abs((eb.X or 0.0) - (ex.X or 0.0)),
            abs((ea.X or 0.0) - (eb.X or 0.0)),
            float(abs(ea.row - eb.row)),
            float(abs(eb.row - ex.row)),
            abs(ea.mendeleev_no - eb.mendeleev_no),
        ],
        dtype=np.float32,
    )


def composition_to_x0(a: str, b: str, x: str, cfg: DiffusionConfig) -> np.ndarray:
    """Full descriptor vector for an ABX3 composition."""
    if getattr(cfg, "elemental_only", False):
        return np.concatenate(
            [element_features(a), element_features(b), element_features(x)]
        ).astype(np.float32)
    assert N_STRUCTURAL == cfg.n_structural
    return np.concatenate(
        [
            element_features(a),
            element_features(b),
            element_features(x),
            structural_features(a, b, x),
        ]
    ).astype(np.float32)


def split_sites(x0: np.ndarray, cfg: DiffusionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a descriptor vector into (A, B, X, structural) blocks."""
    n = cfg.n_features_site
    if getattr(cfg, "elemental_only", False):
        return x0[0:n], x0[n:2 * n], x0[2 * n:3 * n], np.zeros(0, dtype=np.float32)
    return x0[0:n], x0[n:2 * n], x0[2 * n:3 * n], x0[3 * n:3 * n + cfg.n_structural]
