"""Utility functions for thermal parameter computation."""

import numpy as np

def compute_d_i(K_i, C_i, T_a, Q_int, Q_cool, h_i):
    """Compute minimum utilization d_i for zone i (Equation 8)."""
    a = K_i / C_i
    b_off = (K_i * T_a + Q_int) / C_i
    b_on = (K_i * T_a + Q_int + Q_cool) / C_i
    rise_rate = a * h_i - b_off
    cool_rate = a * h_i - b_on
    return rise_rate / (rise_rate - cool_rate)

def compute_cycle_times(K_i, C_i, T_a, Q_int, Q_cool, l_i, h_i):
    """Compute ON/OFF cycle times for worked example."""
    heat_gain = K_i * (T_a - (l_i + h_i) / 2) + Q_int
    net_cooling = abs(Q_cool) - heat_gain
    rise_time_s = C_i * (h_i - l_i) / heat_gain
    cool_time_s = C_i * (h_i - l_i) / net_cooling
    return rise_time_s / 60, cool_time_s / 60  # minutes


def saudi_bill(kwh, tier1=0.18, tier2=0.30, threshold=6000.0):
    """Monthly bill (SAR) under the Saudi two-tier residential tariff (Eq. 3)."""
    return tier1 * min(kwh, threshold) + tier2 * max(0.0, kwh - threshold)


def cost_reduction_pct(baseline_bill, method_bill):
    """Percent cost reduction vs a baseline bill."""
    return (baseline_bill - method_bill) / baseline_bill * 100.0


def tier2_exposure(kwh, threshold=6000.0):
    """kWh billed at the Tier-2 rate."""
    return max(0.0, kwh - threshold)


def aggregate_peak(modes, p_unit=1.8):
    """Peak aggregate electrical power (kW) from a modes array [T, n_zones]."""
    import numpy as np
    return float(np.max(np.sum(np.asarray(modes), axis=1)) * p_unit)
