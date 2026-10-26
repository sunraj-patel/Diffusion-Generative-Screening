"""Lightweight surrogate predictor used for the auxiliary property-consistency term.

The surrogate maps normalized descriptors x0 -> (bandgap, E_hull) in normalized
property space. It is trained once on the training set, frozen, and reused by the
diffusion trainer to penalize estimated-x0 reconstructions that drift from the
conditioning target. ALIGNN can be swapped in here as a drop-in replacement by
overriding `predict_props`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.dataset import PerovskiteDataset
from src.config import DiffusionConfig


class Surrogate(nn.Module):
    def __init__(self, d: int, out_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_props(self, x0_normalized: torch.Tensor) -> torch.Tensor:
        """Return predicted properties in the same normalized space used for conditioning."""
        return self.forward(x0_normalized)


def train_surrogate(cfg: DiffusionConfig, epochs: int = 300, lr: float = 1e-3, device: str = "cpu") -> Surrogate:
    ds = PerovskiteDataset(cfg, split="train")
    model = Surrogate(ds.n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.as_tensor(ds.x, dtype=torch.float32, device=device)
    y = torch.as_tensor(ds.props, dtype=torch.float32, device=device)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        bs = 256
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  surrogate epoch {ep:4d} mse={total / n:.6f}")
    return model
