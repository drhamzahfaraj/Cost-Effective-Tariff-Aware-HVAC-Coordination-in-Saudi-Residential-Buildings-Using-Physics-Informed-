"""
nn_surrogate.py
---------------
Pre-training script for the Neural Network Surrogate that maps
buildng parameters to optimal concurrency limit k* and cooling setpoints.

Architecture: 3-layer MLP (128-256-128)
Input:  (Ta, {C_i, K_i, l_i, h_i}) for all zones  → flattened
Output: (k*, {Q_i*}) for all zones

Pre-trained on 50,000 randomly sampled configurations.
Usage:
    python nn_surrogate.py --samples 50000 --save surrogate.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Parameter bounds for random sampling ──────────────────────────────────
TA_RANGE   = (18.0, 45.0)   # outdoor temperature range (Jeddah year-round)
K_RANGE    = (0.20, 0.35)   # zone conductance kW/K
C_RANGE    = (2000, 4000)   # zone thermal capacity kJ/K
L_OPTIONS  = [22.0, 23.0]   # lower comfort bound
H_OPTIONS  = [25.0, 26.0]   # upper comfort bound
AC_COOLING = 5.3            # kW
AC_ELEC    = 1.8            # kW


class NNSurrogate(nn.Module):
    """
    3-layer MLP surrogate: maps (Ta, zone params) → (k*, Q_i*).
    Input dim: 1 + n_zones * 4  (Ta, C_i, K_i, l_i, h_i per zone)
    Output dim: 1 + n_zones     (k*, Q_i* per zone)
    """
    def __init__(self, n_zones: int):
        super().__init__()
        in_dim  = 1 + n_zones * 4
        out_dim = 1 + n_zones
        self.net = nn.Sequential(
            nn.Linear(in_dim,  128), nn.ReLU(),
            nn.Linear(128,     256), nn.ReLU(),
            nn.Linear(256,     128), nn.ReLU(),
            nn.Linear(128,  out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_min_utilization(K_i, C_i, Ta, Q_int, l, h,
                             Q_cool=-5.3):
    """Eq. (7): minimum utilization d_i evaluated at upper threshold h."""
    a_minus = K_i / C_i
    b_minus = (K_i * Ta + Q_int) / C_i
    a_plus  = K_i / C_i
    b_plus  = (K_i * Ta + Q_int + Q_cool) / C_i
    num = a_minus * h - b_minus
    den = num - (a_plus * h - b_plus)
    return num / den if abs(den) > 1e-9 else 0.5


def generate_dataset(n_samples: int, n_zones: int, seed: int = 42):
    """Generate random (input, target) pairs for surrogate pre-training."""
    rng = np.random.default_rng(seed)
    X, Y = [], []

    for _ in range(n_samples):
        Ta    = rng.uniform(*TA_RANGE)
        K     = rng.uniform(*K_RANGE, size=n_zones)
        C     = rng.uniform(*C_RANGE, size=n_zones)
        l     = rng.choice(L_OPTIONS)
        h     = rng.choice(H_OPTIONS)
        Q_int = rng.uniform(0.1, 0.5, size=n_zones)

        # Compute k* via sum of utilizations
        utils = [compute_min_utilization(K[i], C[i], Ta, Q_int[i], l, h)
                 for i in range(n_zones)]
        k_star = min(n_zones, max(1, int(sum(utils)) + 1))

        # Q_i* is AC_COOLING when ON (fixed for On/Off units)
        Q_stars = np.full(n_zones, -5.3)

        # Build input vector: [Ta, C_0,K_0,l,h, C_1,K_1,l,h, ...]
        x_vec = [Ta]
        for i in range(n_zones):
            x_vec.extend([C[i], K[i], l, h])

        # Build output vector: [k*, Q_0*, Q_1*, ...]
        y_vec = [float(k_star)] + list(Q_stars)

        X.append(x_vec)
        Y.append(y_vec)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_surrogate(n_samples: int, n_zones: int, epochs: int,
                    batch_size: int, lr: float, save_path: str):
    X, Y = generate_dataset(n_samples, n_zones)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NNSurrogate(n_zones)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Training NN Surrogate: {n_zones} zones, {n_samples} samples, {epochs} epochs")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(xb)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={total_loss/n_samples:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Surrogate saved to {save_path}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description='Pre-train NN Surrogate')
    p.add_argument('--zones',      type=int,   default=5,            help='Number of zones')
    p.add_argument('--samples',    type=int,   default=50000,        help='Training samples')
    p.add_argument('--epochs',     type=int,   default=100,          help='Training epochs')
    p.add_argument('--batch-size', type=int,   default=512,          help='Batch size')
    p.add_argument('--lr',         type=float, default=1e-3,         help='Learning rate')
    p.add_argument('--save',       type=str,   default='surrogate.pt', help='Save path')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_surrogate(
        n_samples  = args.samples,
        n_zones    = args.zones,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        save_path  = args.save
    )
