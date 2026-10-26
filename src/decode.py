"""Nearest-neighbour decoding of generated descriptor vectors into ABX3 compositions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pymatgen.core import Element

from src.config import A_SITES, B_SITES, X_SITES, DiffusionConfig
from src.data.dataset import PerovskiteDataset
from src.data.descriptors import split_sites, tolerance_factor
from src.data.magpie import site_table


@dataclass
class Candidate:
    formula: str
    a: str
    b: str
    x: str
    dist_a: float
    dist_b: float
    dist_x: float
    tol_factor: float
    charge_balanced: bool
    in_training_set: bool
    pred_bandgap: float
    pred_ehull: float


class Decoder:
    """Decode x0 (normalized space) -> nearest-element composition, then filter."""

    def __init__(self, cfg: DiffusionConfig, stats: dict[str, np.ndarray]):
        self.cfg = cfg
        self.x_mean = stats["x_mean"]
        self.x_std = stats["x_std"]
        n = cfg.n_features_site

        self.a_elems = A_SITES
        self.b_elems = B_SITES
        self.x_elems = X_SITES

                                                                                      
        raw = site_table(self.a_elems)
        self.A = (raw - self.x_mean[0:n]) / self.x_std[0:n]
        raw = site_table(self.b_elems)
        self.B = (raw - self.x_mean[n:2 * n]) / self.x_std[n:2 * n]
        raw = site_table(self.x_elems)
        self.X = (raw - self.x_mean[2 * n:3 * n]) / self.x_std[2 * n:3 * n]

        self._training_formulas = self._load_training_formulas()

    def _load_training_formulas(self) -> set[str]:
        ds = PerovskiteDataset(self.cfg, split="train")
        return {f"{r.a}{r.b}{r.x}3" for r in ds.df.itertuples()}

    @staticmethod
    def _nearest(q: np.ndarray, table: np.ndarray) -> tuple[int, float]:
        d = np.linalg.norm(table - q, axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    @staticmethod
    def charge_balanced(a: str, b: str, x: str) -> bool:
        ea, eb, ex = Element(a), Element(b), Element(x)
        a_ox = [s for s in ea.common_oxidation_states if s > 0] or [0]
        b_ox = [s for s in eb.common_oxidation_states if s > 0] or [0]
        x_ox = [s for s in ex.common_oxidation_states if s < 0] or [-1]
        for ao in a_ox:
            for bo in b_ox:
                for xo in x_ox:
                    if abs(ao + bo + 3.0 * xo) < 1e-6:
                        return True
        return False

    def decode(self, x0: np.ndarray) -> tuple[str, str, str, float, float, float]:
        """Return (a, b, x, dist_a, dist_b, dist_x)."""
        a_q, b_q, x_q, _ = split_sites(x0, self.cfg)
        ia, da = self._nearest(a_q, self.A)
        ib, db = self._nearest(b_q, self.B)
        ix, dx = self._nearest(x_q, self.X)
        return self.a_elems[ia], self.b_elems[ib], self.x_elems[ix], da, db, dx

    def decode_batch(self, x0s: np.ndarray) -> list[Candidate]:
        """Decode a batch of normalized descriptor vectors and apply physical filters."""
        cands = []
        for x0 in x0s:
            a, b, x, da, db, dx = self.decode(x0)
            formula = f"{a}{b}{x}3"
            t = tolerance_factor(a, b, x)
            balanced = self.charge_balanced(a, b, x)
            known = formula in self._training_formulas
            cands.append(
                Candidate(
                    formula=formula, a=a, b=b, x=x,
                    dist_a=da, dist_b=db, dist_x=dx,
                    tol_factor=t, charge_balanced=balanced,
                    in_training_set=known, pred_bandgap=0.0, pred_ehull=0.0,
                )
            )
        return cands

    def filter(self, cands: list[Candidate], tol_min: float = 0.8, tol_max: float = 1.0,
               require_charge_balance: bool = True, novelty: bool = True) -> list[Candidate]:
        out = []
        for c in cands:
            if require_charge_balance and not c.charge_balanced:
                continue
            if not (tol_min <= c.tol_factor <= tol_max):
                continue
            if novelty and c.in_training_set:
                continue
            out.append(c)
        return out
