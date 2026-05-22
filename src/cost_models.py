"""Electricity cost models for Saudi residential HVAC scheduling.
Models L (Linear), E (Exponential), S (Step-wise / actual Saudi tariff).
Reference: Equation (8) and Section 3.3 of the paper.
"""
import numpy as np

class CostModelL:
    """Model L: Constant rate 0.12 SAR/kWh (weighted average)."""
    def __init__(self, rate=0.12):
        self.rate, self.name = rate, "L"
    def price(self, e_cum): return self.rate
    def tier(self, e_cum): return 0
    def interval_cost(self, u, p_hvac, dt, e_cum, c_sw, prev_u):
        return (c_sw if u == 1 and prev_u == 0 else 0) + self.rate * p_hvac * u * dt

class CostModelE:
    """Model E: Exponential rate c_r * exp(beta * E_cum)."""
    def __init__(self, base_rate=0.05, beta=5e-4):
        self.base_rate, self.beta, self.name = base_rate, beta, "E"
    def price(self, e_cum): return self.base_rate * np.exp(self.beta * e_cum)
    def tier(self, e_cum): return 0
    def interval_cost(self, u, p_hvac, dt, e_cum, c_sw, prev_u):
        return (c_sw if u == 1 and prev_u == 0 else 0) + self.price(e_cum) * p_hvac * u * dt

class CostModelS:
    """Model S: Saudi four-tier residential tariff (2018).
    Tiers: 0.05 (≤2000), 0.10 (≤4000), 0.18 (≤6000), 0.30 (>6000) SAR/kWh.
    VAT (15%) and fixed charges excluded (do not affect scheduling).
    """
    TIERS = [(2000, 0.05), (4000, 0.10), (6000, 0.18), (float("inf"), 0.30)]
    def __init__(self): self.name = "S"
    def price(self, e_cum):
        for thresh, rate in self.TIERS:
            if e_cum <= thresh: return rate
        return 0.30
    def tier(self, e_cum):
        for i, (thresh, _) in enumerate(self.TIERS):
            if e_cum <= thresh: return i + 1
        return 4
    def interval_cost(self, u, p_hvac, dt, e_cum, c_sw, prev_u):
        return (c_sw if u == 1 and prev_u == 0 else 0) + self.price(e_cum) * p_hvac * u * dt

def get_cost_model(name):
    return {"L": CostModelL, "E": CostModelE, "S": CostModelS}[name]()
