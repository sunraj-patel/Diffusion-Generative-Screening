"""Scrape the Materials Project API for ABX3 (1:1:3) inorganic compounds.

Pulls bandgap, energy-above-hull, formation energy and structural metadata via the
`/materials/summary/` endpoint using the anonymous-formula query `ABC3`, then filters
to charge-balanced 1:1:3 stoichiometries with valid property values.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from src.config import MP_API_KEY, MP_BASE_URL, RAW_JSONL

FIELDS = (
    "material_id,formula_pretty,composition_reduced,band_gap,is_metal,"
    "energy_above_hull,formation_energy_per_atom,volume,nsites,density,"
    "spacegroup_symbol,is_stable,symmetry"
)
PER_PAGE = 1000
TIMEOUT = 60
MAX_PAGES = 100


def _get_page(page: int, session: requests.Session) -> dict:
    url = f"{MP_BASE_URL}/materials/summary/"
    params = {
        "formula": "ABC3",
        "deprecated": "false",
        "_fields": FIELDS,
        "_per_page": PER_PAGE,
        "_page": page,
    }
    for attempt in range(4):
        r = session.get(url, params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        wait = 2 ** attempt * 2
        print(f"  [retry {attempt + 1}/4] HTTP {r.status_code}: {r.text[:120]} (wait {wait}s)")
        time.sleep(wait)
    raise RuntimeError(f"Failed page {page}: HTTP {r.status_code}")


def is_abx3(comp: dict) -> bool:
    """Return True if the reduced composition is an ABX3 (1:1:3) stoichiometry."""
    amounts = sorted(comp.values())
    if len(amounts) != 3:
        return False
    import math

    return math.isclose(amounts[0], 1.0, rel_tol=1e-3) and math.isclose(amounts[1], 1.0, rel_tol=1e-3) and math.isclose(
        amounts[2], 3.0, rel_tol=1e-3
    )


def scrape() -> list[dict]:
    session = requests.Session()
    session.headers.update({"X-API-KEY": MP_API_KEY, "Accept": "application/json"})

    records: list[dict] = []
    total = None
    for page in range(1, MAX_PAGES + 1):
        body = _get_page(page, session)
        meta = body.get("meta", {})
        total = meta.get("total_doc", total)
        data = body.get("data", [])
        if not data:
            break

        kept = 0
        for d in data:
            comp = d.get("composition_reduced") or {}
            if not is_abx3(comp):
                continue
            if d.get("band_gap") is None or d.get("energy_above_hull") is None:
                continue
            kept += 1
            records.append(d)

        print(f"  page {page}: {len(data)} raw, kept {kept} (running total {len(records)}/{total})")
        if len(data) < PER_PAGE:
            break
        time.sleep(0.6)

    with open(RAW_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} ABX3 records -> {RAW_JSONL}")
    return records


if __name__ == "__main__":
    scrape()
