"""
thermal_params.py
-----------------
Building thermal parameters for the 5-zone Saudi villa and 20-zone compound.
All values match Table 1 in the paper (Section 6.2).

Zone thermal model (single zone, no coupling):
    dx_i/dt = (K_i*(Ta - x_i) + Q_int_i + Q_i) / C_i

AC unit specs (18,500 BTU On/Off split):
    Electrical draw P_i = 1.8 kW (ON) or 0 (OFF)
    Cooling output  Q_i = -5.3 kW (ON) or 0 (OFF)
    EER = 10.25
"""

AC_ELECTRICAL_KW = 1.8   # kW input per unit
AC_COOLING_KW    = 5.3   # kW cooling output per unit
AC_EER           = 10.25

# ── 5-Zone Villa ──────────────────────────────────────────────────────────
ZONES_5 = [
    {
        "name":    "Dining Room",
        "area_m2": 30,
        "K_i":     0.28,   # kW/K  outdoor conductance
        "C_i":     3600,   # kJ/K  thermal capacity
        "K_ij":    0.05,   # kW/K  inter-zone conductance
        "adj":     ["Living Room"],
        "Q_int":   0.3,    # kW    internal gains
    },
    {
        "name":    "Living Room",
        "area_m2": 25,
        "K_i":     0.25,
        "C_i":     3000,
        "K_ij":    0.05,
        "adj":     ["Dining Room", "Master Bedroom"],
        "Q_int":   0.4,
    },
    {
        "name":    "Master Bedroom",
        "area_m2": 25,
        "K_i":     0.24,
        "C_i":     3000,
        "K_ij":    0.04,
        "adj":     ["Living Room", "Boys Bedroom 2"],
        "Q_int":   0.2,
    },
    {
        "name":    "Boys Bedroom 2",
        "area_m2": 20,
        "K_i":     0.22,
        "C_i":     2400,
        "K_ij":    0.04,
        "adj":     ["Master Bedroom", "Girls Bedroom 3"],
        "Q_int":   0.2,
    },
    {
        "name":    "Girls Bedroom 3",
        "area_m2": 20,
        "K_i":     0.22,
        "C_i":     2400,
        "K_ij":    0.04,
        "adj":     ["Boys Bedroom 2"],
        "Q_int":   0.2,
    },
]

# ── 20-Zone Compound ──────────────────────────────────────────────────────
# Four identical 5-zone villas scaled by compound adjacency.
# Zones 0-4: Villa A, 5-9: Villa B, 10-14: Villa C, 15-19: Villa D
import copy

def build_compound_20z():
    """Return 20-zone parameter list by replicating 5-zone villa x4."""
    compound = []
    villa_names = ["A", "B", "C", "D"]
    for idx, villa in enumerate(villa_names):
        for zone in copy.deepcopy(ZONES_5):
            zone["name"] = f"Villa {villa} - {zone['name']}"
            compound.append(zone)
    return compound

ZONES_20 = build_compound_20z()

# ── Comfort bounds ─────────────────────────────────────────────────────────
COMFORT_STRICT   = (23.0, 25.0)   # [l, h] degrees C
COMFORT_EXTENDED = (22.0, 26.0)   # [l', h'] degrees C

# ── Theoretical asymptotes (July, Living Room) ─────────────────────────────
# ON:  b+/a+ = (K_i*Ta + Q_int + Q_cooling) / K_i ≈ 15.0 C
# OFF: b-/a- = (K_i*Ta + Q_int) / K_i             ≈ 36.2 C
ASYMP_ON_C  = 15.0
ASYMP_OFF_C = 36.2

# ── Minimum utilization (Section 4.2) ─────────────────────────────────────
def min_utilization(a_minus, b_minus, a_plus, b_plus, h):
    """
    d_i = (a_minus*h - b_minus) / ((a_minus*h - b_minus) - (a_plus*h - b_plus))
    Eq. (7) in the paper.
    """
    num = a_minus * h - b_minus
    den = num - (a_plus * h - b_plus)
    return num / den

# Strict range:   d_i ≈ 0.48   (h=25)
# Extended range: d_i ≈ 0.30   (h=26)  → 36.9% reduction
