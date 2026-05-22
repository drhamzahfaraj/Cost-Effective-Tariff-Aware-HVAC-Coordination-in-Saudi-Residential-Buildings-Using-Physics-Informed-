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
