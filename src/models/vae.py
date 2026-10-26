"""Conditional VAE for 45D elemental descriptors.

Encoder: 45 -> 256 -> 128 -> (mu, logvar) 32
Decoder: 32 + 2(cond) -> 256 -> 45
Cond = normalized [Eg, Ehull] (2D) via stats_45.npz
KL weight beta = 1e-3
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, d: int = 45, hidden1: int = 256, hidden2: int = 128, latent: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.mu = nn.Linear(hidden2, latent)
        self.logvar = nn.Linear(hidden2, latent)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent: int = 32, cond_dim: int = 2, hidden: int = 256, out_dim: int = 45):
        super().__init__()
        self.fc1 = nn.Linear(latent + cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor):
                                
        h = torch.cat([z, cond], dim=1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class VAE(nn.Module):
    def __init__(self, d: int = 45, cond_dim: int = 2, latent: int = 32, hidden1: int = 256, hidden2: int = 128, dec_hidden: int = 256):
        super().__init__()
        self.d = d
        self.cond_dim = cond_dim
        self.latent = latent
        self.encoder = Encoder(d, hidden1, hidden2, latent)
        self.decoder = Decoder(latent, cond_dim, dec_hidden, d)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)
        return recon, mu, logvar

    @torch.no_grad()
    def decode(self, z: torch.Tensor, cond: torch.Tensor):
        return self.decoder(z, cond)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)
