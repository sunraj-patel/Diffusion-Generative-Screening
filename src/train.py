"""Training loop for the conditional diffusion model.

Loss: standard DDPM eps objective + auxiliary property-consistency term in which a
frozen surrogate predictor is applied to the estimated x0 at sampled timesteps and
penalized for deviating from the conditioning target.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import CKPT_DIR, DiffusionConfig
from src.data.dataset import PerovskiteDataset
from src.models.denoiser import Denoiser
from src.models.scheduler import GaussianDiffusion
from src.models.surrogate import Surrogate, train_surrogate


def resolve_device(cfg: DiffusionConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        self.backup = {}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply_to(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    v.copy_(self.shadow[k])


def train(cfg: DiffusionConfig):
    device = resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    print(f"device={device}  d={cfg.d}  T={cfg.T}  elemental_only={getattr(cfg,'elemental_only',False)}")

    ds = PerovskiteDataset(cfg, split="train")
                                                           
    if getattr(cfg, "balanced_sampling", False):
        props = ds.df[["bandgap"]].to_numpy().ravel()
                                                                
        bins = np.digitize(props, bins=[0.01, 1.0, 2.0, 3.5, 5.0])
                   
        counts = np.bincount(bins, minlength=6).astype(float)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        weights = inv[bins]
        weights = weights / weights.mean()             
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True)
        print(f"balanced sampling enabled: bin counts {counts.astype(int).tolist()}, weights {[f'{w:.2f}' for w in inv/ inv.mean()]}")
    else:
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

                                                                                 
    print("Training surrogate predictor...")
    surrogate = train_surrogate(cfg, epochs=300, device=str(device))
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)
    torch.save(surrogate.state_dict(), CKPT_DIR / "surrogate.pt")
    print("Surrogate saved.")

    model = Denoiser(
        d=cfg.d, cond_dim=cfg.cond_dim, hidden=cfg.hidden_dim, n_blocks=cfg.n_blocks,
        t_emb_dim=cfg.t_emb_dim, cond_emb_dim=cfg.cond_emb_dim, null_prob=cfg.null_prob,
        dropout=getattr(cfg, "dropout", 0.1),
    ).to(device)
    ema = EMAModel(model, decay=0.9999)
    diffusion = GaussianDiffusion(T=cfg.T, s=cfg.s, device=str(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    mse = nn.MSELoss()

                                                                                    
                              
    def aux_loss(x0_est: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        pred = surrogate(x0_est)
        return mse(pred, cond)

    model.train()
    for epoch in range(cfg.epochs):
        total_eps = 0.0
        total_aux = 0.0
        nb = 0
        for x0, cond in loader:
            x0, cond = x0.to(device), cond.to(device)
            b = x0.shape[0]
            t = torch.randint(0, cfg.T, (b,), device=device)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)

            opt.zero_grad()
            eps_pred = model(x_t, t, cond)
                                                                   
            snr = diffusion.alpha_bar[t] / (1 - diffusion.alpha_bar[t] + 1e-8)       
                                                                                    
            gamma = 5.0
            w_snr = torch.clamp(snr, max=gamma) / snr.clamp(min=1e-8)
                                  
            loss_eps = ((eps_pred - noise) ** 2).mean(dim=1)       
            loss = (w_snr * loss_eps).mean()

                                                                                  
            if cfg.aux_loss_weight > 0:
                t_aux = torch.randint(0, cfg.T // 4, (cfg.aux_timesteps * b,), device=device)
                x_aux = diffusion.q_sample(
                    x0.repeat(cfg.aux_timesteps, 1), t_aux,
                    torch.randn(cfg.aux_timesteps * b, cfg.d, device=device),
                )
                eps_aux = model(x_aux, t_aux, cond.repeat(cfg.aux_timesteps, 1))
                x0_est = diffusion.predict_x0_from_eps(x_aux, t_aux, eps_aux)
                aux = aux_loss(x0_est, cond.repeat(cfg.aux_timesteps, 1))
                w = diffusion.alpha_bar[t_aux] * cfg.aux_loss_weight
                loss = loss + (w * aux).mean()
                total_aux += (w * aux).mean().item() * b

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            total_eps += loss_eps.mean().item() * b
            nb += b

        sched.step()
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:4d}/{cfg.epochs}  eps_mse={total_eps / nb:.5f}  aux={total_aux / nb:.5f}  lr={sched.get_last_lr()[0]:.2e}")

    ema.apply_to(model)
    torch.save(model.state_dict(), CKPT_DIR / "diffusion.pt")
    print(f"Saved diffusion.pt to {CKPT_DIR / 'diffusion.pt'}")


if __name__ == "__main__":
    train(DiffusionConfig())
