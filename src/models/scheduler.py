"""Cosine-schedule DDPM noise process + DDIM sampler."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, T: int = 300, s: float = 0.008, device: str = "cuda"):
        self.T = T
        self.device = torch.device(device)
        self.register_schedule(T, s)

    def register_schedule(self, T: int, s: float):
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        alpha_bar = torch.cos(((steps + s) / (1 + s)) * (torch.pi / 2)).pow(2)
        alpha_bar = alpha_bar / alpha_bar[0]
        self.alpha_bar = (alpha_bar[1:].float()).to(self.device)       
                                                                                     
        self.alpha_bar = torch.clamp(self.alpha_bar, min=1e-4)
        betas = 1.0 - self.alpha_bar / torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]])
        self.betas = torch.clamp(betas, 1e-5, 0.999).float()
        self.alpha = 1.0 - self.betas
        self.sqrt_alpha_bar = self.alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bar).sqrt()
                                                                
        self.log_one_minus_alpha_bar = (1.0 - self.alpha_bar).log()
        self.log_alpha_bar = self.alpha_bar.log()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_1m = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_1m * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_1m = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return (x_t - sqrt_1m * eps) / sqrt_ab

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        steps: int = 50,
        eta: float = 0.0,
        guidance_w: float = 1.0,
    ) -> torch.Tensor:
        """DDIM sampling with optional classifier-free guidance.

        With guidance_w > 1, `cond` is used as the conditional and the model is
        evaluated on the null condition to steer the sample.
        """
        device = self.device
        timesteps = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_b = t.expand(shape[0])
            if guidance_w > 1.0:
                null_cond = torch.zeros_like(cond)
                eps_cond = model(x, t_b, cond)
                eps_uncond = model(x, t_b, null_cond)
                eps = eps_uncond + guidance_w * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_b, cond)

                        
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones(1, device=device)
            sqrt_ab_t = alpha_bar_t.sqrt()
            sqrt_1m_ab_t = (1 - alpha_bar_t).sqrt()

            x0_pred = (x - sqrt_1m_ab_t * eps) / sqrt_ab_t
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(
                1 - alpha_bar_t / alpha_bar_prev
            )
            c1 = alpha_bar_prev.sqrt()
            c2 = (1 - alpha_bar_prev - sigma_t.pow(2)).sqrt()
            noise = torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)
            x = c1 * x0_pred + c2 * eps + sigma_t * noise
        return x

    def ddim_sample_guided(
        self,
        model,
        surrogate,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        steps: int = 50,
        eta: float = 0.0,
        guidance_w: float = 1.0,
        surrogate_scale: float = 0.0,
    ) -> torch.Tensor:
        """DDIM with optional surrogate gradient guidance (classifier-guided).

        After CFG-combined epsilon is computed, the predicted x0 is nudged along
        the gradient of surrogate property error:  x0 <- x0 - s * grad_x ||surrogate(x0)-cond||^2
        This is a novel structure-consistent guidance that improves target fidelity
        without extra training.
        """
        device = self.device
        timesteps = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)
                                        
        for i, t in enumerate(timesteps):
            t_b = t.expand(shape[0])
            with torch.no_grad():
                if guidance_w > 1.0:
                    null_cond = torch.zeros_like(cond)
                    eps_cond = model(x, t_b, cond)
                    eps_uncond = model(x, t_b, null_cond)
                    eps = eps_uncond + guidance_w * (eps_cond - eps_uncond)
                else:
                    eps = model(x, t_b, cond)
                alpha_bar_t = self.alpha_bar[t]
                alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones(1, device=device)
                sqrt_ab_t = alpha_bar_t.sqrt()
                sqrt_1m_ab_t = (1 - alpha_bar_t).sqrt()
                x0_pred = (x - sqrt_1m_ab_t * eps) / sqrt_ab_t

                                                                         
            if surrogate_scale > 0 and t < self.T * 0.6:                         
                x0_pred = x0_pred.detach().requires_grad_(True)
                pred = surrogate(x0_pred)
                loss = ((pred - cond) ** 2).mean()
                grad = torch.autograd.grad(loss, x0_pred)[0]
                                
                grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
                                                 
                s = surrogate_scale * (1 - t.float() / self.T)
                x0_pred = x0_pred - s * grad
                x0_pred = x0_pred.detach()

            with torch.no_grad():
                sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(
                    1 - alpha_bar_t / alpha_bar_prev
                )
                c1 = alpha_bar_prev.sqrt()
                c2 = (1 - alpha_bar_prev - sigma_t.pow(2)).sqrt()
                noise = torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)
                                                                                                 
                                                                                
                x = c1 * x0_pred + c2 * eps + sigma_t * noise
        return x

    def ddim_sample_osgd(
        self,
        model,
        surrogate_ensemble: list,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        steps: int = 50,
        eta: float = 0.0,
        w_base: float = 2.0,
        w_min: float = 1.0,
        w_max: float = 3.0,
        alpha: float = 1.0,
        mu_U: float = 0.08,
        sigma_U: float = 0.30,
        lambda_p: float = 0.5,
        lambda_u: float = 0.3,
        lambda_n: float = 0.3,
        D_max: float = 4.5,
        train_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """OSGD: Structure-Aware Risk-Adaptive Conditional Diffusion.

        ε_new = ε_diff(w_adaptive) + λp∇Lp - λu∇U + λn∇L_novelty
        w(x_t)= w_min + (w_max-w_min)·sigmoid(-α·Ũ),  Ũ=(U-μ)/σ,  w_min>0
        novelty bounded: L_novelty = -min(D_chem, D_max),  D_chem = min_i‖z - z_i‖
        D_chem uses descriptor proxy; with ALIGNN use ‖z_ALIGNN - z_i‖.
        """
        device = self.device
        timesteps = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)
        for i, t in enumerate(timesteps):
            t_b = t.expand(shape[0])
                                                            
                                                                                                         
                                                                                
            if surrogate_ensemble is not None and len(surrogate_ensemble) > 0:
                                                                                    
                with torch.no_grad():
                                                                          
                    try:
                        null_c = torch.zeros_like(cond)
                        eps_c = model(x, t_b, cond)
                        eps_u = model(x, t_b, null_c)
                        eps_tmp = eps_u + w_base * (eps_c - eps_u)
                    except Exception:
                        eps_tmp = model(x, t_b, cond)
                    ab_t = self.alpha_bar[t]
                    x0_tmp = (x - (1 - ab_t).sqrt() * eps_tmp) / ab_t.sqrt()
                    preds = torch.stack([s(x0_tmp).detach() for s in surrogate_ensemble], dim=0)             
                    var = preds.var(dim=0).mean(dim=1)     
                    U = var     
                    U_tilde = (U - mu_U) / (sigma_U + 1e-8)
                    g = torch.sigmoid(-alpha * U_tilde)                
                    w_adaptive = w_min + (w_max - w_min) * g     
                                       
                with torch.no_grad():
                    null_cond = torch.zeros_like(cond)
                    eps_cond = model(x, t_b, cond)
                    eps_uncond = model(x, t_b, null_cond)
                                         
                    w_exp = w_adaptive.view(-1, 1) if hasattr(w_adaptive, 'view') else w_base
                    eps = eps_uncond + w_exp * (eps_cond - eps_uncond)
            else:
                with torch.no_grad():
                    if w_base > 1.0:
                        null_cond = torch.zeros_like(cond)
                        eps_cond = model(x, t_b, cond)
                        eps_uncond = model(x, t_b, null_cond)
                        eps = eps_uncond + w_base * (eps_cond - eps_uncond)
                    else:
                        eps = model(x, t_b, cond)

                           
            with torch.no_grad():
                ab_t = self.alpha_bar[t]
                ab_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones(1, device=device)
                x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()

                                                                  
            if t < self.T * 0.6 and (lambda_p > 0 or lambda_u > 0 or lambda_n > 0):
                                        
                x0_g = x0_pred.detach().requires_grad_(True)
                loss = 0.0
                                                           
                if lambda_p > 0 and surrogate_ensemble is not None:
                                                           
                    preds = torch.stack([s(x0_g) for s in surrogate_ensemble], dim=0)
                    mean_pred = preds.mean(dim=0)
                    Lp = ((mean_pred - cond) ** 2).mean()
                    loss = loss + lambda_p * Lp * (1 - t.float() / self.T)
                                                                                    
                                                                   
                if lambda_u > 0 and surrogate_ensemble is not None and len(surrogate_ensemble) > 1:
                    preds = torch.stack([s(x0_g) for s in surrogate_ensemble], dim=0)
                    U = preds.var(dim=0).mean()                        
                    loss = loss - lambda_u * U * (1 - t.float() / self.T)                                           
                                                       
                                       
                if lambda_n > 0 and train_embeddings is not None:
                                                                         
                                                                
                                                        
                                      
                                                                 
                                     
                    dists = torch.cdist(x0_g, train_embeddings)         
                    D, _ = dists.min(dim=1)     
                    D_clamped = torch.clamp(D, max=D_max)
                                                                     
                    L_n = -D_clamped.mean()
                    loss = loss + lambda_n * L_n * (1 - t.float() / self.T)

                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    grad = torch.autograd.grad(loss, x0_g, retain_graph=False)[0]
                    grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
                                                                        
                    x0_pred = (x0_pred - grad).detach()
                else:
                    x0_pred = x0_pred.detach()

            with torch.no_grad():
                ab_t = self.alpha_bar[t]
                ab_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones(1, device=device)
                sigma_t = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
                c1 = ab_prev.sqrt()
                c2 = (1 - ab_prev - sigma_t.pow(2)).sqrt()
                noise = torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)
                x = c1 * x0_pred + c2 * eps + sigma_t * noise
        return x
