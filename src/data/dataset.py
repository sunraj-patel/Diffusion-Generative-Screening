"""Dataset assembly: filter MP records to true ABX3 perovskites, build x0 + labels."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import DATA_DIR, DATASET_CSV, RAW_JSONL, STATS_NPZ, DiffusionConfig
from src.data.descriptors import composition_to_x0
from src.data.magpie import N_FEATURES

ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P"}


@dataclass
class ABX3Record:
    material_id: str
    a: str
    b: str
    x: str
    formula: str
    bandgap: float
    ehull: float
    formation_energy: float


def _x_element(comp: dict) -> str | None:
    for el, amt in comp.items():
        if abs(amt - 3.0) < 1e-3 and el in ANIONS:
            return el
    return None


def assign_ab(a: str, b: str) -> tuple[str, str]:
    """Assign A/B by ionic radius: A is the larger cation (typical perovskite)."""
    from src.data.magpie import ionic_radius

    return (a, b) if ionic_radius(a) >= ionic_radius(b) else (b, a)


def load_records() -> list[ABX3Record]:
    """Filter raw scrape to chemically-sensible ABX3 perovskites, dedup by composition (min E_hull)."""
    from src.config import A_SITES, B_SITES

    best: dict[str, ABX3Record] = {}
    with open(RAW_JSONL) as f:
        for line in f:
            d = json.loads(line)
            comp = d.get("composition_reduced") or {}
            x = _x_element(comp)
            if x is None:
                continue
            a_elem = [el for el, amt in comp.items() if abs(amt - 1.0) < 1e-3 and el != x]
            if len(a_elem) != 2:
                continue
            a, b = assign_ab(*a_elem)
            if a not in A_SITES or b not in B_SITES:
                continue
            key = "-".join(sorted(comp.keys()))
            rec = ABX3Record(
                material_id=d["material_id"],
                a=a,
                b=b,
                x=x,
                formula=d.get("formula_pretty", ""),
                bandgap=float(d["band_gap"]),
                ehull=float(d["energy_above_hull"]),
                formation_energy=float(d.get("formation_energy_per_atom") or 0.0),
            )
            if key not in best or rec.ehull < best[key].ehull:
                best[key] = rec
    return list(best.values())


def build_dataset_csv() -> pd.DataFrame:
    records = load_records()
    df = pd.DataFrame([r.__dict__ for r in records])
    df.to_csv(DATASET_CSV, index=False)
    print(f"{len(df)} ABX3 perovskites -> {DATASET_CSV}")
    return df


def get_records_df() -> pd.DataFrame:
    if DATASET_CSV.exists():
        return pd.read_csv(DATASET_CSV)
    return build_dataset_csv()


def compute_stats(x0: np.ndarray, props: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-8
    return {
        "x_mean": x0.mean(0),
        "x_std": x0.std(0) + eps,
        "prop_mean": props.mean(0),
        "prop_std": props.std(0) + eps,
    }


class PerovskiteDataset(Dataset):
    def __init__(self, cfg: DiffusionConfig, split: str = "train"):
        df = get_records_df()
        self.cfg = cfg
        self.df = df.reset_index(drop=True)

        x0 = np.stack(
            [composition_to_x0(r.a, r.b, r.x, cfg) for r in [ABX3Record(**row) for row in df.to_dict("records")]]
        )
        props = df[["bandgap", "ehull"]].to_numpy(dtype=np.float32)
        assert x0.shape[1] == cfg.d, f"descriptor dim {x0.shape[1]} != cfg.d {cfg.d}"

                                                                          
        stats_path = STATS_NPZ
        if getattr(cfg, "elemental_only", False):
                                                          
            stats_path = DATA_DIR / "stats_45.npz"
                                                                                   
        if stats_path.exists():
            loaded = np.load(stats_path)
                                                 
            if loaded["x_mean"].shape[0] != cfg.d:
                stats = compute_stats(x0, props)
                np.savez(stats_path, **stats)
            else:
                stats = {k: loaded[k] for k in ("x_mean", "x_std", "prop_mean", "prop_std")}
        else:
            stats = compute_stats(x0, props)
            np.savez(stats_path, **stats)
                                                                               
        if not STATS_NPZ.exists() and stats_path != STATS_NPZ:
                                                   
            pass

        self.x_mean, self.x_std = stats["x_mean"], stats["x_std"]
        self.prop_mean, self.prop_std = stats["prop_mean"], stats["prop_std"]

        self.x = (x0 - self.x_mean) / self.x_std
        self.props = (props - self.prop_mean) / self.prop_std

        if split == "train":
            idx = np.random.RandomState(cfg.seed).permutation(len(self.x))[: int(0.9 * len(self.x))]
            self.x, self.props = self.x[idx], self.props[idx]
            self.df = self.df.iloc[idx].reset_index(drop=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.as_tensor(self.x[i]), torch.as_tensor(self.props[i])

    @property
    def n_features(self) -> int:
        return self.cfg.d
