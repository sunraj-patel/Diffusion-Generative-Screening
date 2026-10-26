"""Conditional denoiser epsilon_theta(x_t, t, c) with FiLM conditioning.

Architecture
------------
* Time embedding: sinusoidal (dim 64) -> 2-layer MLP -> t_emb (128)
* Condition embedding: property vector (bandgap, E_hull) -> 2-layer MLP -> c_emb (128)
  (dropped to a learned null embedding 10-20% of the time for classifier-free guidance)
* Backbone: N residual MLP blocks (hidden 256). Each block applies
  Linear -> FiLM(scale/shift from t_emb + c_emb) -> SiLU -> Linear, with a residual
  connection around the whole block.
* Output head: Linear -> R^d predicting the noise.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FiLMBlock(nn.Module):
    """Residual MLP block with FiLM conditioning from a shared conditioning vector."""

    def __init__(self, hidden: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
                                                                         
        self.scale = nn.Linear(cond_dim, hidden)
        self.shift = nn.Linear(cond_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
                                                                                  
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        gamma = self.scale(cond)
        beta = self.shift(cond)
        h = h * gamma + beta
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm(h)
        return x + h


class Denoiser(nn.Module):
    def __init__(self, d: int, cond_dim: int, hidden: int = 256, n_blocks: int = 5,
                 t_emb_dim: int = 64, cond_emb_dim: int = 128, null_prob: float = 0.15, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.cond_dim = cond_dim
        self.t_emb_dim = t_emb_dim
        self.cond_emb_dim = cond_emb_dim
        self.null_prob = null_prob

                                                           
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, cond_emb_dim),
        )
                                                                     
                                                                          
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.SiLU(),
            nn.Linear(256, cond_emb_dim),
        )
                                                              
        self.null_emb = nn.Parameter(torch.randn(cond_emb_dim) * 0.02)
                                                         
        self.null_mask = nn.Parameter(torch.randn(cond_dim) * 0.02)
        self.null_mask.requires_grad = False

        self.input_proj = nn.Linear(d, hidden)
                                                     
        self.input_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, cond_emb_dim, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, d)
                                         
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def embed(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, train: bool = True) -> torch.Tensor:
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_emb_dim))
                                                 
        if self.training and train and self.null_prob > 0:
            mask = torch.rand(c.shape[0], device=c.device) < self.null_prob       
            c_emb_full = self.c_mlp(c)
                                           
            if mask.any():
                c_emb_full = c_emb_full.clone()
                c_emb_full[mask] = self.null_emb
            c_emb = c_emb_full
        else:
                                                                                        
            is_null = (c == 0).all(dim=1)       
            c_emb_full = self.c_mlp(c)
            if is_null.any():
                c_emb_full = c_emb_full.clone()
                c_emb_full[is_null] = self.null_emb
            c_emb = c_emb_full
        return t_emb + c_emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, train: bool = True) -> torch.Tensor:
        cond = self.embed(x, t, c, train)
        h = self.input_proj(x)
        h = self.input_norm(h)
        for block in self.blocks:
            h = block(h, cond)
        return self.head(h)
