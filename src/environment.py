"""
Sinergym Environment Wrapper with Inter-Zone Thermal Coupling
Wraps EnergyPlus 23.2 via Sinergym for multi-zone HVAC simulation.
Adds coupling terms K_ij(x_j - x_i) for thermal buffering.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ZoneConfig:
    """Configuration for a single thermal zone."""
    name: str
    area_m2: float        # Floor area (m²)
    K_i: float            # Outdoor conductance (kW/K)
    C_i: float            # Thermal capacity (kJ/K)
    Q_int: float = 0.3    # Internal gains (kW)
    Q_cool: float = -5.3  # Cooling power when ON (kW)
    P_elec: float = 1.8   # Electrical power when ON (kW)
    neighbors: list = None # Adjacent zone indices
    K_ij: list = None      # Coupling conductances (kW/K)


def compute_heat_balance(
    x: np.ndarray,          # Zone temperatures [n_zones]
    T_a: float,             # Outdoor temperature (°C)
    modes: np.ndarray,      # Compressor modes [n_zones] (0/1)
    zones: list,            # List of ZoneConfig
    dt_seconds: float = 900 # Time step (s) = 15 min
) -> np.ndarray:
    """
    Compute temperature evolution using Equation 1 (with coupling).
    
    C_i * dx_i/dt = K_i*(T_a - x_i) + sum_j K_ij*(x_j - x_i) + Q_int + Q_i(t)
    """
    n = len(zones)
    dx = np.zeros(n)
    
    for i, zone in enumerate(zones):
        # Outdoor heat gain
        q_outdoor = zone.K_i * (T_a - x[i])
        
        # Inter-zone coupling
        q_coupling = 0.0
        if zone.neighbors and zone.K_ij:
            for j, k_ij in zip(zone.neighbors, zone.K_ij):
                q_coupling += k_ij * (x[j] - x[i])
        
        # Internal gains
        q_int = zone.Q_int
        
        # Cooling (ON/OFF)
        q_cool = zone.Q_cool * modes[i]
        
        # Total heat balance
        q_total = q_outdoor + q_coupling + q_int + q_cool
        
        # Temperature derivative
        dx[i] = q_total / zone.C_i * dt_seconds  # °C change
    
    return x + dx


def build_5zone_villa() -> list:
    """Build the 5-zone villa configuration (Table 1 in paper)."""
    zones = [
        ZoneConfig("Majlis", 30, 0.28, 3600, neighbors=[1], K_ij=[0.05]),
        ZoneConfig("Living", 25, 0.25, 3000, neighbors=[0, 2], K_ij=[0.05, 0.04]),
        ZoneConfig("Master", 25, 0.24, 3000, neighbors=[1, 3], K_ij=[0.04, 0.04]),
        ZoneConfig("Bedroom2", 20, 0.22, 2400, neighbors=[2, 4], K_ij=[0.04, 0.04]),
        ZoneConfig("Bedroom3", 20, 0.22, 2400, neighbors=[3], K_ij=[0.04]),
    ]
    return zones


def build_compound(n_zones: int = 20, seed: int = 42) -> list:
    """Build n-zone compound with procedural parameter generation."""
    rng = np.random.default_rng(seed)
    zones = []
    for i in range(n_zones):
        K_i = rng.uniform(0.20, 0.30)
        C_i = rng.uniform(2000, 4000)
        K_ij_val = rng.uniform(0.03, 0.06)
        neighbors = []
        K_ij_list = []
        if i > 0:
            neighbors.append(i - 1)
            K_ij_list.append(K_ij_val)
        if i < n_zones - 1:
            neighbors.append(i + 1)
            K_ij_list.append(K_ij_val)
        zones.append(ZoneConfig(
            f"Zone_{i}", rng.uniform(20, 30), K_i, C_i,
            neighbors=neighbors, K_ij=K_ij_list
        ))
    return zones
