"""Conditional WGAN-GP sampler for continuous tabular features."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _Generator(nn.Module):
    def __init__(self, noise_dim, n_classes, x_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + n_classes, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, z, labels):
        return self.net(torch.cat([z, labels], dim=1))


class _Critic(nn.Module):
    def __init__(self, x_dim, n_classes, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + n_classes, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, labels):
        return self.net(torch.cat([x, labels], dim=1)).reshape(-1)


def _one_hot(y, n_classes=2):
    return torch.nn.functional.one_hot(y.long(), num_classes=n_classes).float()


def sample_wgan_gp(
    X,
    y,
    n0,
    n1,
    seed=42,
    epochs=200,
    batch_size=64,
    noise_dim=32,
    hidden=128,
    critic_steps=5,
    gradient_penalty=10.0,
    learning_rate=1e-4,
    device=None,
    verbose=False,
):
    """Train a conditional WGAN-GP and return exact per-class sample counts."""

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("sample_wgan_gp currently expects binary labels encoded 0/1.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    X_tensor = torch.as_tensor(X_scaled, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    generator = _Generator(noise_dim, 2, X.shape[1], hidden).to(device)
    critic = _Critic(X.shape[1], 2, hidden).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    rng = torch.Generator(device=device).manual_seed(seed)
    batch_size = min(int(batch_size), len(X))
    steps_per_epoch = max(1, int(np.ceil(len(X) / batch_size)))

    def gradient_penalty_term(real, fake, labels):
        alpha = torch.rand(len(real), 1, device=device, generator=rng)
        mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        scores = critic(mixed, labels)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=mixed,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    for epoch in range(int(epochs)):
        for _ in range(steps_per_epoch):
            index = torch.randint(len(X), (batch_size,), device=device, generator=rng)
            real = X_tensor[index]
            labels = _one_hot(y_tensor[index])
            for _ in range(int(critic_steps)):
                z = torch.randn(batch_size, noise_dim, device=device, generator=rng)
                fake = generator(z, labels).detach()
                loss_c = (
                    critic(fake, labels).mean()
                    - critic(real, labels).mean()
                    + gradient_penalty * gradient_penalty_term(real, fake, labels)
                )
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()

            sampled_y = torch.randint(2, (batch_size,), device=device, generator=rng)
            sampled_labels = _one_hot(sampled_y)
            z = torch.randn(batch_size, noise_dim, device=device, generator=rng)
            loss_g = -critic(generator(z, sampled_labels), sampled_labels).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
        if verbose and (epoch == 0 or (epoch + 1) % 25 == 0):
            print(
                f"[WGAN-GP] epoch {epoch + 1}/{epochs} "
                f"critic={loss_c.item():.4f} generator={loss_g.item():.4f}"
            )

    requested_y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
    with torch.no_grad():
        label_tensor = torch.as_tensor(requested_y, dtype=torch.long, device=device)
        labels = _one_hot(label_tensor)
        z = torch.randn(len(requested_y), noise_dim, device=device, generator=rng)
        generated = generator(z, labels).cpu().numpy()
    return scaler.inverse_transform(generated).astype(np.float32), requested_y
