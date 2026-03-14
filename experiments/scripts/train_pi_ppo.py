"""
train_pi_ppo.py
---------------
Training script for PI-PPO: Physics-Informed Proximal Policy Optimization
for HVAC scheduling in Saudi residential buildings.

Usage:
    python train_pi_ppo.py --zones 5 --months july --comfort strict
    python train_pi_ppo.py --zones 20 --months jul --comfort extended
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from typing import List, Tuple

# ── Reward weights (from paper Table ablation) ────────────────────────────
OMEGA_PEAK    = 1.0   # omega_0: quadratic peak penalty
OMEGA_TARIFF  = 0.8   # omega_1: tiered tariff marginal rate
OMEGA_COMFORT = 5.0   # omega_2: comfort violation penalty
OMEGA_PHYSICS = 2.0   # omega_3: heat-balance residual
OMEGA_FEAS    = 1.5   # omega_4: feasibility boundary
OMEGA_SWITCH  = 0.3   # omega_5: compressor cycling

# ── Tariff thresholds (SAR) ───────────────────────────────────────────────
TARIFF_LOW        = 0.18   # SAR/kWh below threshold
TARIFF_HIGH       = 0.30   # SAR/kWh above threshold
TARIFF_THRESHOLD  = 6000.0 # kWh

# ── AC unit specs ─────────────────────────────────────────────────────────
AC_ELECTRICAL_KW  = 1.8    # electrical input per unit
AC_COOLING_KW     = 5.3    # thermal output per unit
DT_MINUTES        = 15     # control timestep


def marginal_tariff(e_cum: float) -> float:
    """Return current marginal tariff rate based on cumulative consumption."""
    return TARIFF_HIGH if e_cum >= TARIFF_THRESHOLD else TARIFF_LOW


def heat_balance_residual(
    x: np.ndarray,
    x_dot_obs: np.ndarray,
    m: np.ndarray,
    params: dict
) -> float:
    """
    Compute |dx_obs/dt - dx_pred/dt| summed over all zones.
    Equation (1) from the paper (without inter-zone coupling for residual).

    Args:
        x:          zone temperatures [n]
        x_dot_obs:  observed temperature derivatives [n]
        m:          compressor modes {0,1} [n]
        params:     dict with keys K, C, Ta, Q_int per zone
    Returns:
        scalar residual
    """
    residual = 0.0
    n = len(x)
    for i in range(n):
        K_i   = params['K'][i]
        C_i   = params['C'][i]
        Ta    = params['Ta']
        Q_int = params['Q_int'][i]
        Q_i   = -AC_COOLING_KW if m[i] == 1 else 0.0
        x_dot_pred = (K_i * (Ta - x[i]) + Q_int + Q_i) / C_i
        residual  += abs(x_dot_obs[i] - x_dot_pred)
    return residual


def compute_reward(
    x: np.ndarray,
    m: np.ndarray,
    m_prev: np.ndarray,
    x_dot_obs: np.ndarray,
    e_cum: float,
    comfort_bounds: Tuple[float, float],
    k: int,
    params: dict,
    d_hat: float
) -> float:
    """
    Physics-informed reward r(t) = r_peak + r_tariff + r_comfort
                                  + r_physics + r_feas + r_switch
    Equation (9) from the paper.
    """
    l, h = comfort_bounds
    dt_h = DT_MINUTES / 60.0  # timestep in hours

    # Peak penalty (quadratic)
    agg_power = AC_ELECTRICAL_KW * m.sum()
    r_peak = -OMEGA_PEAK * (agg_power ** 2)

    # Tiered tariff penalty
    p = marginal_tariff(e_cum)
    r_tariff = -OMEGA_TARIFF * p * agg_power * dt_h

    # Comfort violation penalty
    violations = np.maximum(0, x - h) + np.maximum(0, l - x)
    r_comfort = -OMEGA_COMFORT * violations.sum()

    # Physics residual
    r_physics = -OMEGA_PHYSICS * heat_balance_residual(x, x_dot_obs, m, params)

    # Feasibility boundary penalty
    r_feas = -OMEGA_FEAS * max(0.0, d_hat - k + 0.5)

    # Switching penalty
    r_switch = -OMEGA_SWITCH * np.abs(m - m_prev).sum()

    return r_peak + r_tariff + r_comfort + r_physics + r_feas + r_switch


class PIPPOPolicy(nn.Module):
    """
    2-layer (2x256) policy network with Bernoulli output heads.
    State: [x(t), m(t-1), Ta(t), E_cum(t), k, dx(t), t_hour]
    Action: m(t) in {0,1}^n with top-k projection
    """
    def __init__(self, n_zones: int, hidden: int = 256):
        super().__init__()
        state_dim = n_zones * 3 + 3  # x, m_prev, dx per zone + Ta, E_cum, t_hour
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_zones),
            nn.Sigmoid()
        )
        self.n_zones = n_zones

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Bernoulli probabilities for each zone."""
        return self.net(state)

    def act(self, state: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with top-k projection and return log-prob."""
        probs = self.forward(state)
        dist = Bernoulli(probs)
        m = dist.sample()
        # Top-k projection: keep at most k ON
        if m.sum() > k:
            on_indices = m.nonzero(as_tuple=True)[0]
            topk_idx = probs[on_indices].topk(k).indices
            m = torch.zeros_like(m)
            m[on_indices[topk_idx]] = 1.0
        log_prob = dist.log_prob(m).sum()
        return m, log_prob


class PPOTrainer:
    """PPO training loop with clipped surrogate objective (Schulman et al., 2017)."""

    def __init__(
        self,
        n_zones: int,
        k: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        ppo_epochs: int = 10
    ):
        self.policy = PIPPOPolicy(n_zones)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma     = gamma
        self.eps_clip  = eps_clip
        self.ppo_epochs = ppo_epochs
        self.k         = k

    def update(self, trajectories: List[dict]):
        """One PPO update step over collected trajectories."""
        states      = torch.stack([t['state']    for t in trajectories])
        actions     = torch.stack([t['action']   for t in trajectories])
        old_log_probs = torch.tensor([t['log_prob'] for t in trajectories])
        rewards     = torch.tensor([t['reward']  for t in trajectories], dtype=torch.float32)

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards.tolist()):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            probs = self.policy(states)
            dist  = Bernoulli(probs)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy   = dist.entropy().sum(dim=-1).mean()

            ratio  = torch.exp(log_probs - old_log_probs)
            surr1  = ratio * returns
            surr2  = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * returns
            loss   = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser(description='Train PI-PPO for HVAC scheduling')
    parser.add_argument('--zones',   type=int,   default=5,      help='Number of AC zones')
    parser.add_argument('--months',  nargs='+',  default=['july'],
                        choices=['jan','apr','jul','oct','january','april','july','october'],
                        help='Simulation months')
    parser.add_argument('--comfort', type=str,   default='strict',
                        choices=['strict','extended'],
                        help='strict=[23,25]C  extended=[22,26]C')
    parser.add_argument('--episodes',type=int,   default=500,    help='Training episodes')
    parser.add_argument('--seed',    type=int,   default=42,     help='Random seed')
    parser.add_argument('--save',    type=str,   default='pi_ppo_policy.pt',
                        help='Path to save trained policy')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    comfort_map = {
        'strict':   (23.0, 25.0),
        'extended': (22.0, 26.0)
    }
    comfort_bounds = comfort_map[args.comfort]

    # Concurrency limit: floor(n/2) for strict, floor(n/2.5) for extended
    k = max(1, args.zones // 2) if args.comfort == 'strict' else max(1, int(args.zones / 2.5))

    print(f'PI-PPO Training')
    print(f'  Zones:   {args.zones}')
    print(f'  Months:  {args.months}')
    print(f'  Comfort: {comfort_bounds[0]}-{comfort_bounds[1]}C  (k={k})')
    print(f'  Episodes:{args.episodes}')

    trainer = PPOTrainer(n_zones=args.zones, k=k)
    # NOTE: Connect to Sinergym/EnergyPlus environment here.
    # See experiments/scripts/evaluate.py for environment setup.
    print('Policy network ready. Connect EnergyPlus/Sinergym environment to begin training.')
    print(f'Save path: {args.save}')


if __name__ == '__main__':
    main()
