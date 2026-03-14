"""
jeddah_tmy3.py
--------------
Jeddah TMY3 (IWEC 41024) representative 24-hour ambient temperature profiles
for four simulation months. Values match Figure 1 in the paper.

All temperatures in degrees Celsius.
Time axis: hours 0–24 (hourly resolution, linearly interpolated for 15-min steps).
"""

import numpy as np

# ── Representative 24-h profiles (TMY3 data points) ──────────────────────
PROFILES = {
    "january": [
        (0, 19.0), (2, 18.5), (4, 18.0), (6, 18.5), (8, 21.0),
        (10, 24.0), (12, 27.0), (14, 29.0), (15, 29.0), (16, 28.0),
        (18, 25.0), (20, 22.0), (22, 20.0), (24, 19.0),
    ],
    "april": [
        (0, 24.0), (2, 23.5), (4, 23.0), (6, 24.0), (8, 27.0),
        (10, 30.0), (12, 33.0), (14, 35.0), (15, 35.0), (16, 34.0),
        (18, 31.0), (20, 28.0), (22, 26.0), (24, 24.0),
    ],
    "july": [
        (0, 31.0), (2, 30.0), (4, 29.0), (6, 30.0), (8, 34.0),
        (10, 38.0), (12, 42.0), (14, 43.0), (15, 42.5), (16, 41.0),
        (18, 38.0), (20, 35.0), (22, 33.0), (24, 31.0),
    ],
    "october": [
        (0, 27.0), (2, 26.0), (4, 25.5), (6, 26.0), (8, 29.0),
        (10, 32.0), (12, 35.0), (14, 37.0), (15, 36.5), (16, 35.0),
        (18, 33.0), (20, 30.0), (22, 28.0), (24, 27.0),
    ],
}

# Daily mean temperatures (used in worked example, Section 6.2)
DAILY_MEAN = {
    "january": 22.0,
    "april":   28.5,
    "july":    35.0,   # used in worked example: Ta=35C
    "october": 30.0,
}

# Peak afternoon temperatures
DAILY_PEAK = {
    "january": 29.0,
    "april":   35.0,
    "july":    43.0,   # most severe scheduling challenge
    "october": 37.0,
}


def get_profile(month: str, dt_minutes: int = 15) -> np.ndarray:
    """
    Return interpolated temperature profile at dt_minutes resolution.

    Args:
        month:      'january' | 'april' | 'july' | 'october'
        dt_minutes: control timestep in minutes (default 15)
    Returns:
        np.ndarray of shape (n_steps,) with temperatures in degrees C.
    """
    month = month.lower()
    if month not in PROFILES:
        raise ValueError(f"Month '{month}' not found. Choose from {list(PROFILES.keys())}")

    pts = PROFILES[month]
    hours = np.array([p[0] for p in pts], dtype=float)
    temps = np.array([p[1] for p in pts], dtype=float)

    n_steps = int(24 * 60 / dt_minutes)
    t_query = np.linspace(0, 24, n_steps, endpoint=False)
    profile = np.interp(t_query, hours, temps)
    return profile


if __name__ == "__main__":
    for month in PROFILES:
        p = get_profile(month)
        print(f"{month.capitalize():10s}: mean={p.mean():.1f}C  peak={p.max():.1f}C  steps={len(p)}")
