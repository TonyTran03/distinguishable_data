# models/cvae.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.config import Config
from util.transformation import make_transform


class CVAE(nn.Module):
    """
    Conditional VAE for tabular features.

    Encoder: [x, c] -> (mu, logvar)
    Reparam: z = mu + exp(0.5*logvar) * eps
    Decoder: [z, c] -> x_hat
    """
    def __init__(self, x_dim: int, c_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        xc = torch.cat([x, c], dim=1)
        h = self.enc(xc)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        eps = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        zc = torch.cat([z, c], dim=1)
        return self.dec(zc)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar


# training / sampling helpers 

def _elbo_loss(x, x_hat, mu, logvar, beta: float):
    recon = ((x_hat - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)).mean()
    return recon + beta * kl, recon, kl


@torch.no_grad()
def _evaluate(model, loader, device, beta: float):
    model.eval()
    tot = rec = kl = 0.0
    n = 0
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        x_hat, mu, logvar = model(x, c)
        loss, r, k = _elbo_loss(x, x_hat, mu, logvar, beta)
        tot += loss.item(); rec += r.item(); kl += k.item(); n += 1
    return {"loss": tot/n, "recon": rec/n, "kl": kl/n}


def train_cvae(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Optional[Config] = None,
    device=None,
    verbose: bool = True,
) -> Tuple["CVAE", np.ndarray, np.ndarray, object]:
    """
    Train a CVAE on (X, y).

    Returns (model, scaler_mean, scaler_scale, transform).
    """
    from src.data import make_loaders

    if cfg is None:
        cfg = Config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, scaler = make_loaders(
        X, y,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        x_transform=cfg.x_transform,
    )

    x_dim = X.shape[1]
    model = CVAE(x_dim=x_dim, c_dim=2, z_dim=cfg.z_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot = rec = kl = 0.0
        n = 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(x, c)
            if cfg.decoder_noise > 0:
                x_hat = x_hat + cfg.decoder_noise * torch.randn_like(x_hat)
            loss, r, k = _elbo_loss(x, x_hat, mu, logvar, beta=cfg.beta)
            loss.backward()
            opt.step()
            tot += loss.item(); rec += r.item(); kl += k.item(); n += 1

        train_metrics = {"loss": tot/n, "recon": rec/n, "kl": kl/n}
        val_metrics = _evaluate(model, val_loader, device, beta=cfg.beta)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={train_metrics['loss']:.4f} recon={train_metrics['recon']:.4f} kl={train_metrics['kl']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    transform = make_transform(cfg.x_transform)
    return model, mean, scale, transform


@torch.no_grad()
def _sample_from_model(model, n0, n1, mean, scale, transform, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)

    def _one(n, label):
        z = torch.randn(n, model.z_dim, device=device, generator=g)
        c = F.one_hot(
            torch.full((n,), label, dtype=torch.long, device=device),
            num_classes=model.c_dim,
        ).float()
        x_scaled = model.decode(z, c).cpu().numpy()
        x_t = x_scaled * scale + mean
        return transform.inverse(x_t).astype(np.float32)

    X0 = _one(n0, 0)
    X1 = _one(n1, 1)
    X_syn = np.vstack([X0, X1])
    y_syn = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return X_syn, y_syn


def sample_cvae(
    X: np.ndarray,
    y: np.ndarray,
    n0: int,
    n1: int,
    seed: int = 42,
    cfg: Optional[Config] = None,
    device=None,
):
    """
    Train a CVAE on (X, y), then sample n0+n1 synthetic rows.
    Returns (X_syn, y_syn) with class 0 rows first, then class 1.
    """
    if cfg is None:
        cfg = Config()
    cfg.seed = seed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, mean, scale, transform = train_cvae(X, y, cfg=cfg, device=device)

    return _sample_from_model(model, n0, n1, mean, scale, transform, device, seed)


def sample_trained_cvae(cvae_state, n0: int, n1: int, seed: int = 42):
    """
    Sample from the in-memory state returned by train_cvae.
    """
    model, mean, scale, transform = cvae_state
    device = next(model.parameters()).device
    return _sample_from_model(model, n0, n1, mean, scale, transform, device, seed)


if __name__ == "__main__":
    x_dim, c_dim, z_dim, hidden = 10, 2, 4, 16
    model = CVAE(x_dim, c_dim, z_dim, hidden)
    x = torch.randn(5, x_dim)
    c = F.one_hot(torch.randint(0, c_dim, (5,)), num_classes=c_dim).float()
    x_hat, mu, logvar = model(x, c)
    print("x_hat shape:", x_hat.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
