"""Sample candidates with DDIM + classifier-free guidance sweep, decode, filter, rank."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src.config import CKPT_DIR, OUTPUT_DIR, STATS_NPZ, DiffusionConfig
from src.data.dataset import PerovskiteDataset
from src.decode import Decoder
from src.models.denoiser import Denoiser
from src.models.scheduler import GaussianDiffusion
from src.models.surrogate import Surrogate
from src.train import resolve_device


def normalize_target(cfg: DiffusionConfig, bandgap: float, ehull: float) -> np.ndarray:
    stats = np.load(STATS_NPZ)
    raw = np.asarray([bandgap, ehull], dtype=np.float32)
    return (raw - stats["prop_mean"]) / stats["prop_std"]


def _load_stats(cfg: DiffusionConfig):
                          
    if getattr(cfg, "elemental_only", False):
        p = DATA_DIR / "stats_45.npz"
        if p.exists():
            return np.load(p)
    return np.load(STATS_NPZ)

def sample(cfg: DiffusionConfig, surrogate_guidance: float = 0.0):
    device = resolve_device(cfg)
    ds = PerovskiteDataset(cfg, split="train")
    stats = _load_stats(cfg)

    model = Denoiser(
        d=cfg.d, cond_dim=cfg.cond_dim, hidden=cfg.hidden_dim, n_blocks=cfg.n_blocks,
        t_emb_dim=cfg.t_emb_dim, cond_emb_dim=cfg.cond_emb_dim, null_prob=0.0,
        dropout=0.0,
    ).to(device)
                                                 
    ckpt_path = CKPT_DIR / "diffusion.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "diffusion_elemental.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    surrogate = Surrogate(cfg.d).to(device)
    sur_path = CKPT_DIR / "surrogate.pt"
                                                                                       
    surrogate.load_state_dict(torch.load(sur_path, map_location=device))
    surrogate.eval()

    diffusion = GaussianDiffusion(T=cfg.T, s=cfg.s, device=str(device))
    decoder = Decoder(cfg, stats)

    target_cond = torch.as_tensor(
        normalize_target(cfg, cfg.target_bandgap, cfg.target_ehull), dtype=torch.float32, device=device
    ).unsqueeze(0)

    all_results = []
    for w in cfg.guidance_weights:
        print(f"--- guidance w={w} ---")
        n = cfg.n_candidates_per_setting
        batch = min(512, n)
        cands = []
        for start in range(0, n, batch):
            bs = min(batch, n - start)
            cond = target_cond.expand(bs, -1)
            if surrogate_guidance > 0:
                x_gen = diffusion.ddim_sample_guided(
                    model, surrogate, (bs, cfg.d), cond=cond, steps=cfg.ddim_steps, eta=cfg.eta, guidance_w=w, surrogate_scale=surrogate_guidance
                )
            else:
                x_gen = diffusion.ddim_sample(
                    model, (bs, cfg.d), cond=cond, steps=cfg.ddim_steps, eta=cfg.eta, guidance_w=w
                )
            if isinstance(x_gen, torch.Tensor):
                x_gen = x_gen.detach().cpu().numpy()
            cands.extend(decoder.decode_batch(x_gen))

        filtered = decoder.filter(cands, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=True)
        print(f"  generated {len(cands)}, passed filters {len(filtered)}")

                                                                     
        if filtered:
            from src.data.descriptors import composition_to_x0

            X = np.stack([composition_to_x0(c.a, c.b, c.x, cfg) for c in filtered], dtype=np.float32)
            Xn = (X - stats["x_mean"]) / stats["x_std"]
            with torch.no_grad():
                pred = surrogate(torch.as_tensor(Xn, device=device)).cpu().numpy()
            pred_bg = pred[:, 0] * stats["prop_std"][0] + stats["prop_mean"][0]
            pred_hull = pred[:, 1] * stats["prop_std"][1] + stats["prop_mean"][1]

            score = np.abs(pred_bg - cfg.target_bandgap) + 0.5 * np.clip(pred_hull, 0, None)
            order = np.argsort(score)
            for c, s, pb, ph in zip([filtered[i] for i in order], score[order], pred_bg[order], pred_hull[order]):
                c.pred_bandgap = float(pb)
                c.pred_ehull = float(ph)
                c.score = float(s)

            top = [filtered[i] for i in order[:20]]
            for c in top:
                print(
                    f"    {c.formula:10s} t={c.tol_factor:.3f} charge={c.charge_balanced}"
                    f" pred_bg={c.pred_bandgap:.3f} pred_hull={c.pred_ehull:.3f} score={c.score:.3f}"
                )
            all_results.extend(
                [
                    {
                        "formula": c.formula, "A": c.a, "B": c.b, "X": c.x,
                        "tol_factor": c.tol_factor, "charge_balanced": c.charge_balanced,
                        "in_training_set": c.in_training_set,
                        "dist_a": c.dist_a, "dist_b": c.dist_b, "dist_x": c.dist_x,
                        "pred_bandgap": c.pred_bandgap, "pred_ehull": c.pred_ehull,
                        "score": c.score, "guidance_w": w,
                    }
                    for c in top
                ]
            )

    df = pd.DataFrame(all_results)
                                                                                 
    df = df.loc[df.groupby("formula")["score"].idxmin()].sort_values("score").reset_index(drop=True)
    out_csv = OUTPUT_DIR / "candidates.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} unique ranked candidates -> {out_csv}")

                                                         
    if not df.empty:
        overall = df.head(20)
        out_json = OUTPUT_DIR / "top20_for_dft.json"
        with open(out_json, "w") as f:
            json.dump(overall.to_dict("records"), f, indent=2)
        print(f"Saved top-20 DFT candidates -> {out_json}")
        print("\nTOP 20 CANDIDATES (for DFT validation):")
        for r in overall.to_dict("records"):
            print(f"  {r['formula']:10s} bg~{r['pred_bandgap']:.3f} eV  hull~{r['pred_ehull']:.3f}  w={r['guidance_w']}")


if __name__ == "__main__":
    sample(DiffusionConfig())
