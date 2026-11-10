"""Chemically grouped splits and a constant-baseline check for property predictors."""
from __future__ import annotations
import numpy as np


def b_site_disjoint_split(b_sets, test_fraction=0.2, seed=0):
    """Split so that no B-site element appears in both train and test. ``b_sets`` is a list of sets of B elements per
    structure. Elements are assigned to the test side at random until the test fraction is reached; a structure goes to
    test only if all its B elements are test elements and to train only if none are; mixed structures are dropped.
    Returns (train_idx, test_idx, dropped_idx)."""
    rng = np.random.default_rng(seed)
    els = sorted({e for b in b_sets for e in b}); rng.shuffle(els)
    test_els, target = set(), test_fraction * len(b_sets)
    count = lambda T: sum(1 for b in b_sets if b and set(b) <= T)
    for e in els:
        if count(test_els) >= target:
            break
        test_els.add(e)
    tr, te, dr = [], [], []
    for i, b in enumerate(b_sets):
        b = set(b)
        (te if b and b <= test_els else tr if not (b & test_els) else dr).append(i)
    return tr, te, dr


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / d) if d else float("nan")


def vs_constant_baseline(y_train, y_test, p_test):
    """Compare model MAE with a predictor that always returns the training mean (and median). The model is useful on
    this test only if its MAE is clearly below both baselines."""
    y_test = np.asarray(y_test, float)
    out = dict(model_mae=mae(y_test, p_test), model_r2=r2(y_test, p_test),
               mean_baseline_mae=mae(y_test, np.full_like(y_test, np.mean(y_train))),
               median_baseline_mae=mae(y_test, np.full_like(y_test, np.median(y_train))))
    out["beats_constant"] = out["model_mae"] < min(out["mean_baseline_mae"], out["median_baseline_mae"])
    return out
