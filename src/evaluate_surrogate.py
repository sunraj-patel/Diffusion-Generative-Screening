"""Evaluate the surrogate predictor on held-out test sets with physical-unit metrics.

Produces (into outputs/):
  - outputs/surrogate_test_results.csv : MAE / RMSE / R2 for bandgap & E_hull
  - outputs/figures/parity_*.png        : DFT vs ML parity plots
  - outputs/figures/error_dist_*.png    : residual error distributions

Two splits:
  1. "random" : random 80/20 hold-out
  2. "hard_b" : B-site rare-earth family held out (chemistry under-represented
                in training -> tests extrapolation to unseen B-site chemistry)

Training-set normalization statistics are computed on the training split only
(no leakage into the test set).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, DiffusionConfig
from src.data.dataset import get_records_df
from src.data.descriptors import composition_to_x0
from src.models.surrogate import Surrogate
from src.train import resolve_device

                                                                                   
HELD_OUT_B_SITES = ["Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def build_matrices(cfg):
    df = get_records_df().reset_index(drop=True)
    x0 = np.stack([composition_to_x0(r.a, r.b, r.x, cfg) for r in df.itertuples()]).astype(np.float32)
    props = df[["bandgap", "ehull"]].to_numpy(dtype=np.float32)
    return df, x0, props


def fit_surrogate(X, y, device, epochs=500, lr=1e-3, bs=256):
    """Train a fresh surrogate on normalized (X, y); returns trained model in eval mode."""
    model = Surrogate(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    n = len(Xt)
    perm = torch.randperm(n, device=device)
    Xt, yt = Xt[perm], yt[perm]
    for ep in range(epochs):
        model.train()
        for i in range(0, n, bs):
            opt.zero_grad()
            loss = lossf(model(Xt[i : i + bs]), yt[i : i + bs])
            loss.backward()
            opt.step()
    model.eval()
    return model


def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def split_train_test(cfg, name, seed=0):
    """Return (df, Xn, y, train_idx, test_idx) with train-only normalization."""
    df, x0, props = build_matrices(cfg)
    rng = np.random.RandomState(seed)
    n = len(df)

    if name == "random":
        idx = rng.permutation(n)
        n_test = int(0.2 * n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
    elif name == "hard_b":
        test_idx = np.where(df["b"].isin(HELD_OUT_B_SITES))[0]
        train_idx = np.array([i for i in range(n) if df["b"].iloc[i] not in set(HELD_OUT_B_SITES)])
    else:
        raise ValueError(name)

    x_mean, x_std = x0[train_idx].mean(0), x0[train_idx].std(0) + 1e-8
    p_mean, p_std = props[train_idx].mean(0), props[train_idx].std(0) + 1e-8
    Xn = (x0 - x_mean) / x_std
    y = (props - p_mean) / p_std
    return df, x0, props, Xn, y, train_idx, test_idx, (p_mean, p_std)


def run_split(name, cfg, device):
    df, x0, props, Xn, y, tr, te, (p_mean, p_std) = split_train_test(cfg, name)
    print(f"\n=== split '{name}'  train={len(tr)}  test={len(te)} ===")
    model = fit_surrogate(Xn[tr], y[tr], device)
    with torch.no_grad():
        pred = model(torch.as_tensor(Xn[te], device=device)).cpu().numpy()
    pred = pred * p_std + p_mean                                

    rows = [
        {"split": name, "property": "Bandgap (eV)", **dict(zip(["MAE", "RMSE", "R2"], metrics(props[te, 0], pred[:, 0])))},
        {"split": name, "property": "E_hull (eV/atom)", **dict(zip(["MAE", "RMSE", "R2"], metrics(props[te, 1], pred[:, 1])))},
    ]
    print(pd.DataFrame(rows).to_string(index=False))

                                                 
    for j, prop in enumerate(["Bandgap", "E_hull"]):
        err = pred[:, j] - props[te, j]
        err = err[np.isfinite(err)]
        if len(err) == 0:
            continue
        print(
            f"  {prop} errors [eV]: mean={err.mean():+.3f}  median={np.median(err):+.3f} "
            f"(abs) p50={np.percentile(np.abs(err),50):.3f} p90={np.percentile(np.abs(err),90):.3f} "
            f"p99={np.percentile(np.abs(err),99):.3f}"
        )
        f = OUTPUT_DIR / "figures"
        f.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(err, bins=40, color="#4C72B0", alpha=0.85)
        ax.set_title(f"{prop} prediction error ({name})")
        ax.set_xlabel("error [DFT - ML], physical units")
        ax.set_ylabel("count")
        ax.axvline(0, color="k", lw=1)
        fig.tight_layout()
        fig.savefig(f / f"error_dist_{prop}_{name}.png", dpi=150)
        plt.close(fig)

                   
    for j, prop, fn in [
        (0, "Bandgap", "eg"),
        (1, "E_hull", "ehull"),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        vmax = max(props[te, j].max(), pred[:, j].max())
        vmin = min(props[te, j].min(), pred[:, j].min())
        ax.scatter(props[te, j], pred[:, j], s=22, alpha=0.6, color="#55A868")
        lims = [min(vmin, 0), max(vmax, 0.01)]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel(f"DFT {prop} (eV)")
        ax.set_ylabel(f"ML {prop} (eV)")
        ax.set_title(f"{prop}: DFT vs ML ({name} split)\nMAE={rows[j]['MAE']:.3f}  R2={rows[j]['R2']:.3f}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figures" / f"parity_{fn}_{name}.png", dpi=150)
        plt.close(fig)

    return rows


def main():
    cfg = DiffusionConfig()
    device = resolve_device(cfg)
    print(f"device={device}")
    all_rows = []
                                                                              
    all_rows += run_split("random", cfg, device)
    all_rows += run_split("hard_b", cfg, device)
    df = pd.DataFrame(all_rows)
    out = OUTPUT_DIR / "surrogate_test_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()