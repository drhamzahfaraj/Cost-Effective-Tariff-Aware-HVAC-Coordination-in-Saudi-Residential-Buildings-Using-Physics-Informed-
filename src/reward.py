"""
Physics-Informed Reward Function for PI-PPO
6-component reward: peak + tariff + comfort + physics + feasibility + switching
"""

import numpy as np


# Reward weights (Table 7 in paper)
OMEGA = {
    'peak': 0.5,      # omega_0: quadratic peak demand penalty
    'tariff': 1.0,     # omega_1: tiered tariff cost signal
    'comfort': 10.0,   # omega_2: comfort violation penalty (high to enforce near-zero)
    'physics': 0.5,    # omega_3: heat balance residual (dense guidance)
    'feas': 2.0,       # omega_4: feasibility margin penalty
    'switch': 0.1,     # omega_5: compressor cycling penalty (mild)
}

# Saudi two-tier tariff
TIER1_RATE = 0.18   # SAR/kWh for first 6,000 kWh
TIER2_RATE = 0.30   # SAR/kWh above 6,000 kWh
TIER_THRESHOLD = 6000  # kWh


def compute_tariff_rate(e_cum: float) -> float:
    """Return current marginal tariff rate based on cumulative consumption."""
    return TIER1_RATE if e_cum <= TIER_THRESHOLD else TIER2_RATE


def compute_bill(e_total: float) -> float:
    """Compute monthly bill using Saudi two-tier tariff formula."""
    if e_total <= TIER_THRESHOLD:
        return TIER1_RATE * e_total
    return TIER1_RATE * TIER_THRESHOLD + TIER2_RATE * (e_total - TIER_THRESHOLD)


def compute_reward(
    modes: np.ndarray,           # m(t): binary compressor modes [n_zones]
    prev_modes: np.ndarray,      # m(t-1): previous modes [n_zones]
    temperatures: np.ndarray,    # x(t): zone temperatures [n_zones]
    temp_derivatives_obs: np.ndarray,  # dx/dt observed [n_zones]
    temp_derivatives_pred: np.ndarray, # dx/dt predicted from Eq. 1 [n_zones]
    comfort_low: np.ndarray,     # l_i: lower comfort bounds [n_zones]
    comfort_high: np.ndarray,    # h_i: upper comfort bounds [n_zones]
    e_cum: float,                # cumulative consumption (kWh)
    d_hat: float,                # estimated total utilization
    k: int,                      # concurrency limit
    p_unit: float = 1.8,         # electrical power per unit (kW)
    dt: float = 0.25,            # time step (hours) = 15 min
) -> tuple[float, dict]:
    """
    Compute 6-component physics-informed reward.
    
    Returns:
        total_reward: scalar reward
        components: dict of individual reward components for logging
    """
    n_active = np.sum(modes)
    
    # r_peak: quadratic penalty on instantaneous aggregate demand
    r_peak = -OMEGA['peak'] * (p_unit * n_active) ** 2
    
    # r_tariff: tiered cost signal
    rate = compute_tariff_rate(e_cum)
    r_tariff = -OMEGA['tariff'] * rate * p_unit * n_active * dt
    
    # r_comfort: comfort violation penalty
    violations_high = np.maximum(0, temperatures - comfort_high)
    violations_low = np.maximum(0, comfort_low - temperatures)
    r_comfort = -OMEGA['comfort'] * np.sum(violations_high + violations_low)
    
    # r_physics: heat balance residual (Eq. 1 consistency)
    residual = np.abs(temp_derivatives_obs - temp_derivatives_pred)
    r_physics = -OMEGA['physics'] * np.sum(residual)
    
    # r_feas: feasibility margin penalty
    r_feas = -OMEGA['feas'] * max(0, d_hat - k + 0.5)
    
    # r_switch: compressor cycling penalty
    switches = np.sum(np.abs(modes - prev_modes))
    r_switch = -OMEGA['switch'] * switches
    
    total = r_peak + r_tariff + r_comfort + r_physics + r_feas + r_switch
    
    components = {
        'r_peak': r_peak,
        'r_tariff': r_tariff,
        'r_comfort': r_comfort,
        'r_physics': r_physics,
        'r_feas': r_feas,
        'r_switch': r_switch,
        'total': total,
        'n_active': n_active,
        'max_violation': float(np.max(violations_high + violations_low)),
        'tariff_rate': rate,
    }
    
    return total, components
