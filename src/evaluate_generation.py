"""Characterize the full generated distribution per classifier-free-guidance setting.

For each w in {1.0, 1.5, 2.0, 2.5, 3.0}: generate 5,000 candidates, decode them,
and report the standard generative-model metrics over the whole distribution:

  CFG   Valid   Unique   Novel   Target MAE   Stable*   Diversity
  1.0
  ...

Definitions (all fractions are of the 5,000 decoded candidates unless noted):
  - Valid      : fraction passing physical filters (charge balance + tolerance
                 factor in [0.8, 1.0]), independent of novelty.
  - Unique     : fraction of valid candidates with a distinct formula.
  - Novel      : fraction of valid candidates not present in the training set.
  - Target MAE : mean |pred_bandgap - 1.4 eV| over valid candidates (eV).
  - Stable*    : fraction of valid candidates with surrogate-predicted
                 E_hull < 0.1 eV/atom (near-hull, asterisk = surrogate proxy).
  - Diversity  : mean pairwise Euclidean distance between the normalized
                 descriptors of valid candidates (subsampled to 1,000) plus the
                 number of distinct B-site elements used.

Also saves:
  - outputs/generation/w{weight}.csv        : per-candidate details for each CFG
  - outputs/generation_summary.csv          : the table above
  - outputs/figures/bandgap_dist_*.png      : training vs generated distributions
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CKPT_DIR, OUTPUT_DIR, STATS_NPZ, DiffusionConfig
from src.data.dataset import PerovskiteDataset, get_records_df
from src.data.descriptors import composition_to_x0
from src.decode import Decoder
from src.models.denoiser import Denoiser
from src.models.scheduler import GaussianDiffusion
from src.models.surrogate import Surrogate
from src.train import resolve_device

GUIDANCE = [1.0, 1.5, 2.0, 2.5, 3.0]
N_PER_SETTING = 5000
BATCH = 512
EHULL_STABLE_CUT = 0.1           
DIV_SUBSAMPLE = 1000


def main():
    cfg = DiffusionConfig()
    device = resolve_device(cfg)
    stats = np.load(STATS_NPZ)
    print(f"device={device}")

    ds = PerovskiteDataset(cfg, split="train")

    model = Denoiser(
        d=cfg.d, cond_dim=cfg.cond_dim, hidden=cfg.hidden_dim, n_blocks=cfg.n_blocks,
        t_emb_dim=cfg.t_emb_dim, cond_emb_dim=cfg.cond_emb_dim, null_prob=0.0,
    ).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "diffusion.pt", map_location=device))
    model.eval()

    surrogate = Surrogate(cfg.d).to(device)
    surrogate.load_state_dict(torch.load(CKPT_DIR / "surrogate.pt", map_location=device))
    surrogate.eval()

    diffusion = GaussianDiffusion(T=cfg.T, s=cfg.s, device=str(device))
    decoder = Decoder(cfg, stats)

    target_cond = torch.as_tensor(
        (np.asarray([cfg.target_bandgap, cfg.target_ehull], dtype=np.float32) - stats["prop_mean"]) / stats["prop_std"],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)

    training_formulas = set(ds.df["a"] + ds.df["b"] + ds.df["x"] + "3")

    gen_dir = OUTPUT_DIR / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for w in GUIDANCE:
        print(f"--- guidance w={w} ---")
        cands = []
        for start in range(0, N_PER_SETTING, BATCH):
            bs = min(BATCH, N_PER_SETTING - start)
            cond = target_cond.expand(bs, -1)
            x_gen = diffusion.ddim_sample(
                model, (bs, cfg.d), cond=cond, steps=cfg.ddim_steps, eta=cfg.eta, guidance_w=w
            ).detach().cpu().numpy()
            cands.extend(decoder.decode_batch(x_gen))

                                                                                   
        valid = decoder.filter(cands, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=False)
        print(f"  decoded {len(cands)}, physically valid {len(valid)}")

        if valid:
            X = np.stack([composition_to_x0(c.a, c.b, c.x, cfg) for c in valid], dtype=np.float32)
            Xn = (X - stats["x_mean"]) / stats["x_std"]
            with torch.no_grad():
                pred = surrogate(torch.as_tensor(Xn, device=device)).cpu().numpy()
            pred_bg = pred[:, 0] * stats["prop_std"][0] + stats["prop_mean"][0]
            pred_hull = pred[:, 1] * stats["prop_std"][1] + stats["prop_mean"][1]
            for c, pb, ph in zip(valid, pred_bg, pred_hull):
                c.pred_bandgap = float(pb)
                c.pred_ehull = float(ph)

        formulas = np.array([c.formula for c in valid])
        uniq_mask = np.unique(formulas, return_index=True)[1]
        uniq_formulas = formulas[uniq_mask]
        novel_mask = np.array([f not in training_formulas for f in formulas])
        novel_uniq = np.array([f not in training_formulas for f in uniq_formulas])

        valid_frac = len(valid) / N_PER_SETTING
        uniq_frac = len(uniq_formulas) / max(len(valid), 1)
        novel_frac = novel_mask.sum() / max(len(valid), 1)

        if len(valid):
            target_mae = float(np.abs(np.array([c.pred_bandgap for c in valid]) - cfg.target_bandgap).mean())
            stable_frac = float(np.mean(np.array([c.pred_ehull for c in valid]) < EHULL_STABLE_CUT))
            desc = Xn
            if len(desc) > DIV_SUBSAMPLE:
                rng = np.random.RandomState(0)
                idx = rng.choice(len(desc), DIV_SUBSAMPLE, replace=False)
                desc = desc[idx]
            D = desc[:, None, :] - desc[None, :, :]
            diversity = float(np.sqrt((D ** 2).sum(-1))[np.triu_indices(len(desc), 1)].mean())
            b_sites = sorted({c.b for c in valid})
            n_b_sites = len(b_sites)
        else:
            target_mae = float("nan")
            stable_frac = float("nan")
            diversity = float("nan")
            n_b_sites = 0
            b_sites = []

        row = {
            "CFG": w,
            "Valid": f"{valid_frac:.3f}",
            "Unique": f"{uniq_frac:.3f}",
            "Novel": f"{novel_frac:.3f}",
            "Target MAE": f"{target_mae:.3f}",
            "Stable*": f"{stable_frac:.3f}",
            "Diversity": f"{diversity:.3f}",
            "n_valid": len(valid),
            "n_unique": len(uniq_formulas),
            "n_novel": int(novel_mask.sum()),
            "n_B_sites": n_b_sites,
        }
        summary_rows.append(row)
        print(row)

        df = pd.DataFrame(
            [
                {
                    "formula": c.formula, "A": c.a, "B": c.b, "X": c.x,
                    "tol_factor": c.tol_factor, "charge_balanced": c.charge_balanced,
                    "in_training_set": c.in_training_set,
                    "dist_a": c.dist_a, "dist_b": c.dist_b, "dist_x": c.dist_x,
                    "pred_bandgap": c.pred_bandgap, "pred_ehull": c.pred_ehull,
                    "valid": c.charge_balanced and 0.8 <= c.tol_factor <= 1.0,
                    "novel": c.formula not in training_formulas,
                }
                for c in cands
            ]
        )
        df.to_csv(gen_dir / f"w{w}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "generation_summary.csv", index=False)
    print(f"\nSummary saved -> {OUTPUT_DIR / 'generation_summary.csv'}")
    print(summary[["CFG", "Valid", "Unique", "Novel", "Target MAE", "Stable*", "Diversity"]].to_string(index=False))

                                                                           
    train_df = get_records_df()
    train_bg = train_df["bandgap"].to_numpy()
    figdir = OUTPUT_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    bins = np.linspace(0, max(train_bg.max(), 5), 40)
    for k, (w, ax) in enumerate(zip([None] + GUIDANCE, axes.ravel())):
        if w is None:
            ax.hist(train_bg, bins=bins, color="#4C72B0", alpha=0.85, label="training (DFT)")
            ax.set_title("Training set (MP)")
        else:
            g = pd.read_csv(gen_dir / f"w{w}.csv")
            g = g[g["valid"]]
            gb = g["pred_bandgap"].to_numpy()
            ax.hist(gb, bins=bins, color="#DD8452", alpha=0.85, label="generated (surrogate)")
            ax.set_title(f"generated, w={w}")
            ax.legend(fontsize=8)
        ax.set_ylabel("count")
    axes[-1, -1].axis("off")
    fig.suptitle(f"Bandgap distribution: training vs generated (target {cfg.target_bandgap} eV)")
    fig.tight_layout()
    fig.savefig(figdir / "bandgap_dist_by_cfg.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(train_bg, bins=40, color="#4C72B0", alpha=0.75, density=True, label="training (DFT)")
    for w, col in zip(GUIDANCE, ["#55A868", "#DD8452", "#C44E52", "#8172B3", "#CCB974"]):
        g = pd.read_csv(gen_dir / f"w{w}.csv")
        g = g[g["valid"]]
        ax.hist(g["pred_bandgap"].to_numpy(), bins=40, histtype="step", density=True, lw=1.5,
                color=col, label=f"generated w={w}")
    ax.set_xlabel("bandgap (eV)")
    ax.set_ylabel("density")
    ax.set_title(f"Bandgap distribution: training vs generated (target {cfg.target_bandgap} eV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "bandgap_dist_train_vs_generated.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
