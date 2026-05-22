"""
Lazy Scheduling Controller (GS Baseline)
Reactive controller that switches only at thresholds.
A zone at h_i with compressor OFF is 'critical' and must be cooled immediately.
Safe by construction but purely reactive — cannot anticipate demand surges.
"""

import numpy as np


class LazyScheduler:
    """Green Scheduling lazy controller for On/Off HVAC coordination."""
    
    def __init__(self, n_zones: int, k: int, comfort_low: np.ndarray, comfort_high: np.ndarray):
        self.n_zones = n_zones
        self.k = k
        self.l = comfort_low
        self.h = comfort_high
    
    def step(self, temperatures: np.ndarray, current_modes: np.ndarray) -> np.ndarray:
        """
        Compute next compressor modes using lazy scheduling.
        
        Critical zones (OFF but at h_i) are turned ON immediately.
        Non-critical zones maintain current mode unless resource constraint violated.
        """
        new_modes = current_modes.copy()
        
        # Identify critical zones: OFF but temperature at upper threshold
        critical = np.where((current_modes == 0) & (temperatures >= self.h - 0.05))[0]
        
        # Identify zones that can be turned OFF: ON and temperature at lower threshold
        at_lower = np.where((current_modes == 1) & (temperatures <= self.l + 0.05))[0]
        
        # Turn OFF zones at lower threshold
        for idx in at_lower:
            new_modes[idx] = 0
        
        # Turn ON all critical zones
        for idx in critical:
            new_modes[idx] = 1
        
        # Check resource constraint
        n_on = int(np.sum(new_modes))
        if n_on > self.k:
            # Shed non-critical ON zones (those furthest from h_i)
            on_zones = np.where(new_modes == 1)[0]
            non_critical_on = np.setdiff1d(on_zones, critical)
            # Sort by temperature (coolest first = most margin to shed)
            if len(non_critical_on) > 0:
                sorted_by_temp = non_critical_on[np.argsort(temperatures[non_critical_on])]
                n_to_shed = n_on - self.k
                for idx in sorted_by_temp[:n_to_shed]:
                    new_modes[idx] = 0
        
        return new_modes
