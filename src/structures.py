"""Build crystal structures for decoded ABX3 perovskite compositions.

Generates a cubic perovskite prototype (space group Pm-3m, #221) with the A-site at
the corner, B-site at the body center and X-site at the face centers. The lattice
constant is estimated from Shannon ionic radii: a = 2 * (r_B + r_X). Oxidation states
are assigned from the first charge-balanced combination of common oxidation states so
the resulting Structure passes pymatgen's charge-balance check.

Outputs CIF files ready for downstream DFT validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymatgen.core import Element, Lattice, Species, Structure

                                                                       
                                                              
                                                                               
_PROTO_SITES = [
    ("A", (0.0, 0.0, 0.0)),
    ("B", (0.5, 0.5, 0.5)),
    ("X", (0.5, 0.5, 0.0)),
]


def charge_balanced_states(a: str, b: str, x: str) -> tuple[int, int, int] | None:
    """Return a charge-balanced (oxA, oxB, oxX) triple from common oxidation states."""
    ea, eb, ex = Element(a), Element(b), Element(x)
    a_ox = [s for s in ea.common_oxidation_states if s > 0] or [0]
    b_ox = [s for s in eb.common_oxidation_states if s > 0] or [0]
    x_ox = [s for s in ex.common_oxidation_states if s < 0] or [-1]
    for ao in a_ox:
        for bo in b_ox:
            for xo in x_ox:
                if abs(ao + bo + 3.0 * xo) < 1e-6:
                    return ao, bo, xo
    return None


def shannon_radius(symbol: str, ox: int) -> float:
    radii = Element(symbol).ionic_radii
    if radii:
                                                                       
        if ox in radii:
            return float(radii[ox])
        closest = min(radii, key=lambda k: abs(k - ox))
        return float(radii[closest])
    return Element(symbol).average_ionic_radius or 1.0


def lattice_parameter(a: str, b: str, x: str, ox: tuple[int, int, int]) -> float:
    """Ideal cubic perovskite lattice constant a = 2 * (r_B + r_X)."""
    rb = shannon_radius(b, ox[1])
    rx = shannon_radius(x, ox[2])
    return 2.0 * (rb + rx)


def build_structure(a: str, b: str, x: str, lattice: float | None = None) -> Structure:
    """Construct the cubic perovskite Structure for composition ABX3."""
    ox = charge_balanced_states(a, b, x)
    if lattice is None:
        if ox is None:
            ra, rb, rx = (Element(s).average_ionic_radius or 1.0 for s in (a, b, x))
            lattice = 2.0 * (rb + rx)
        else:
            lattice = lattice_parameter(a, b, x, ox)

    species = []
    coords = []
    element_map = {"A": a, "B": b, "X": x}
    for site, coord in _PROTO_SITES:
        sym = element_map[site]
        if ox is not None:
            ox_state = {"A": ox[0], "B": ox[1], "X": ox[2]}[site]
        else:
            ox_state = 0
        species.append(Species(sym, ox_state) if ox_state != 0 else Element(sym))
        coords.append(coord)

    structure = Structure.from_spacegroup("Pm-3m", Lattice.cubic(lattice), species, coords, coords_are_cartesian=False)
    structure.properties["lattice_parameter"] = lattice
    return structure


def write_cifs(candidates: list[dict], out_dir: Path) -> list[Path]:
    """Write a CIF for every candidate and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, c in enumerate(candidates, start=1):
        formula = c["formula"]
        structure = build_structure(c["A"], c["B"], c["X"])
        path = out_dir / f"candidate_{i:02d}_{formula}.cif"
        structure.to(filename=str(path))
        paths.append(path)
    return paths


if __name__ == "__main__":
    from src.config import OUTPUT_DIR

    with open(OUTPUT_DIR / "top20_for_dft.json") as f:
        cands = json.load(f)
    cif_paths = write_cifs(cands, OUTPUT_DIR / "structures")
    for p in cif_paths:
        print(p.name)
    print(f"Wrote {len(cif_paths)} structure files.")
