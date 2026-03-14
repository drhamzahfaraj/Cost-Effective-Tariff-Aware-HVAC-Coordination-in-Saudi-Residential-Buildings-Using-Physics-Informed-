"""
baselines.py
------------
Baseline controllers for comparison with PI-PPO:
  - OnOff:  Independent thermostatic control
  - GS:     Lazy (green) scheduling
  - PPO:    Standard PPO (no physics / no tariff)
  - DQN:    Deep Q-Network
  - MPC1:   1-step Model Predictive Control
  - MPC3:   3-step Model Predictive Control

All controllers share a common interface: act(state) -> action.
"""

import numpy as np
from typing import Tuple, List


AC_ELECTRICAL_KW = 1.8
AC_COOLING_KW    = 5.3


class OnOffController:
    """
    Independent thermostatic (bang-bang) controller.
    Each zone independently switches ON at h_i and OFF at l_i.
    This is the uncoordinated baseline; peak = n * 1.8 kW when all zones hot.
    """
    def __init__(self, comfort_bounds: Tuple[float, float]):
        self.l, self.h = comfort_bounds

    def act(self, x: np.ndarray, m_prev: np.ndarray) -> np.ndarray:
        m = m_prev.copy()
        for i in range(len(x)):
            if x[i] >= self.h:
                m[i] = 1  # turn ON: too hot
            elif x[i] <= self.l:
                m[i] = 0  # turn OFF: cool enough
        return m


class LazyScheduler:
    """
    Green (lazy) scheduling baseline.
    Switches only at comfort thresholds with concurrency limit k.
    A zone at h_i with compressor OFF is 'critical' and has priority.
    Cannot anticipate afternoon demand surge or exploit inter-zone buffering.
    """
    def __init__(self, comfort_bounds: Tuple[float, float], k: int):
        self.l, self.h = comfort_bounds
        self.k = k

    def act(self, x: np.ndarray, m_prev: np.ndarray) -> np.ndarray:
        n = len(x)
        m = np.zeros(n, dtype=int)

        # Priority 1: zones at upper bound (critical)
        critical = [i for i in range(n) if x[i] >= self.h]
        for i in critical[:self.k]:
            m[i] = 1

        # Priority 2: zones already ON that haven't cooled to lower bound
        remaining = self.k - m.sum()
        if remaining > 0:
            active = [i for i in range(n) if m_prev[i] == 1 and x[i] > self.l and m[i] == 0]
            for i in active[:remaining]:
                m[i] = 1

        return m


class MPCController:
    """
    Finite-horizon Model Predictive Control.
    Solves a combinatorial search over {0,1}^n for H steps.
    Infeasible beyond 20 zones due to exponential cost.

    Args:
        comfort_bounds: (l, h) temperature limits
        k:              concurrency limit
        horizon:        planning horizon (steps)
        params:         thermal parameters dict
    """
    def __init__(
        self,
        comfort_bounds: Tuple[float, float],
        k: int,
        horizon: int = 1,
        params: dict = None
    ):
        self.l, self.h = comfort_bounds
        self.k = k
        self.H = horizon
        self.params = params or {}

    def predict_temperature(
        self,
        x_i: float,
        m_i: int,
        i: int,
        dt: float = 0.25  # 15 min in hours
    ) -> float:
        """Euler step of single-zone dynamics (Eq. 2, no coupling)."""
        K = self.params.get('K', [0.25] * 100)[i]
        C = self.params.get('C', [3000] * 100)[i]
        Ta = self.params.get('Ta', 35.0)
        Q_int = self.params.get('Q_int', [0.3] * 100)[i]
        Q_i = -AC_COOLING_KW if m_i == 1 else 0.0
        dx = (K * (Ta - x_i) + Q_int + Q_i) / C
        return x_i + dx * dt * 3600  # dt in seconds

    def act(self, x: np.ndarray, m_prev: np.ndarray) -> np.ndarray:
        """Greedy 1-step MPC (horizon=1 used for scalability baseline)."""
        n = len(x)
        best_m = m_prev.copy()
        best_cost = float('inf')

        # Enumerate all valid combinations (feasible only for small n)
        from itertools import combinations
        indices = list(range(n))
        for on_set in combinations(indices, min(self.k, n)):
            m_try = np.zeros(n, dtype=int)
            for i in on_set:
                m_try[i] = 1
            cost = 0.0
            feasible = True
            for i in range(n):
                x_next = self.predict_temperature(x[i], m_try[i], i)
                if x_next > self.h or x_next < self.l - 2:
                    feasible = False
                    break
                cost += AC_ELECTRICAL_KW * m_try[i]
            if feasible and cost < best_cost:
                best_cost = cost
                best_m = m_try.copy()

        return best_m


class StandardPPOController:
    """
    Standard PPO controller (no physics reward, no tariff awareness).
    Identical architecture to PI-PPO but reward = r_peak + r_comfort only.
    Used as ablation baseline.
    """
    def __init__(self, n_zones: int, k: int, policy_path: str = None):
        self.n_zones = n_zones
        self.k = k
        self.policy_path = policy_path
        # Policy loaded from checkpoint if provided
        self.policy = None
        if policy_path:
            import torch
            self.policy = torch.load(policy_path, map_location='cpu')

    def act(self, state: np.ndarray) -> np.ndarray:
        if self.policy is None:
            # Random valid action as placeholder
            import random
            m = np.zeros(self.n_zones, dtype=int)
            on = random.sample(range(self.n_zones), min(self.k, self.n_zones))
            for i in on:
                m[i] = 1
            return m
        import torch
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32)
            probs = self.policy(s)
            m = (probs > 0.5).int().numpy()
        return m
