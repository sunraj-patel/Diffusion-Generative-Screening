"""Validate top candidates against the real Materials Project DFT database.

For each candidate formula, query MP's `/materials/summary/` endpoint and report the
actual DFT-computed bandgap and energy-above-hull (ground truth proxy for the pending
DFT validation), or mark it as not-yet-in-MP (a true discovery candidate).
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from src.config import MP_API_KEY, MP_BASE_URL, OUTPUT_DIR


def query_mp_formula(formula: str, session: requests.Session) -> dict | None:
    url = f"{MP_BASE_URL}/materials/summary/"
    params = {
        "formula": formula,
        "_fields": "material_id,formula_pretty,band_gap,energy_above_hull,formation_energy_per_atom,is_stable",
        "_limit": 5,
    }
    for attempt in range(3):
        r = session.get(url, params=params, timeout=60)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if not data:
                return None
                                               
            return min(data, key=lambda d: d.get("energy_above_hull") if d.get("energy_above_hull") is not None else 1e9)
        time.sleep(2 ** attempt)
    return None


def validate():
    with open(OUTPUT_DIR / "top20_for_dft.json") as f:
        top = json.load(f)

    session = requests.Session()
    session.headers.update({"X-API-KEY": MP_API_KEY, "Accept": "application/json"})

    rows = []
    for c in top:
        mp = query_mp_formula(c["formula"], session)
        rows.append(
            {
                "rank": len(rows) + 1,
                "formula": c["formula"],
                "model_pred_bandgap": round(c["pred_bandgap"], 3),
                "model_pred_hull": round(c["pred_ehull"], 3),
                "in_mp": mp is not None,
                "mp_bandgap": round(mp["band_gap"], 3) if mp and mp.get("band_gap") is not None else None,
                "mp_ehull": round(mp["energy_above_hull"], 4) if mp else None,
                "mp_stable": bool(mp["is_stable"]) if mp else None,
                "mp_formula": mp["formula_pretty"] if mp else None,
            }
        )
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "validation_vs_mp.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    validate()
