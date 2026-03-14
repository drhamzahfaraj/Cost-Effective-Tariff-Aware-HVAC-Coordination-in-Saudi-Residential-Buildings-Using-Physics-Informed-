"""
jeddah_tmy3.py
--------------
Jeddah TMY3 ambient temperature data (IWEC station 41024).
Provides 24-hour representative profiles for the four simulation months.

Data source: EnergyPlus Weather Data (IWEC), Jeddah, Saudi Arabia
Station: IWEC 41024  Lat: 21.68 N  Lon: 39.15 E  Elev: 12 m

Usage:
    from jeddah_tmy3 import get_hourly_profile, get_daily_mean
"""

import numpy as np
from typing import Dict, List

# 24-hour TMY3 representative profiles (hourly, degrees Celsius)
# Extracted from EnergyPlus IWEC 41024, averaged over the respective month
TMY3_PROFILES: Dict[str, List[float]] = {
    'jan': [19.0, 18.5, 18.0, 18.0, 18.5, 19.0, 20.0, 21.5,
            23.0, 24.5, 26.0, 27.0, 28.5, 29.0, 29.0, 28.0,
            26.0, 24.0, 22.5, 21.5, 21.0, 20.5, 20.0, 19.5],

    'apr': [24.0, 23.5, 23.0, 23.0, 24.0, 25.0, 26.5, 28.0,
            30.0, 32.0, 33.5, 34.5, 35.0, 35.0, 34.5, 33.5,
            32.0, 30.0, 28.5, 27.5, 27.0, 26.5, 26.0, 25.5],

    'jul': [31.0, 30.0, 29.5, 29.0, 30.0, 31.5, 33.0, 35.5,
            38.0, 40.5, 42.0, 42.8, 43.0, 43.0, 42.5, 41.5,
            39.5, 37.5, 36.0, 35.0, 34.0, 33.0, 32.0, 31.5],

    'oct': [27.0, 26.5, 25.5, 25.5, 26.0, 27.0, 28.5, 30.0,
            32.0, 34.0, 35.5, 36.5, 37.0, 37.0, 36.5, 35.5,
            33.5, 31.5, 30.0, 29.0, 28.5, 28.0, 27.5, 27.0]
}

# Month aliases
MONTH_ALIASES = {
    'january': 'jan', 'february': 'feb', 'march': 'mar',
    'april':   'apr', 'may':      'may', 'june':  'jun',
    'july':    'jul', 'august':   'aug', 'september': 'sep',
    'october': 'oct', 'november': 'nov', 'december':  'dec'
}


def get_hourly_profile(month: str) -> np.ndarray:
    """
    Return 24-hour ambient temperature profile for the given month.

    Args:
        month: 'jan', 'apr', 'jul', 'oct' (or full name)
    Returns:
        np.ndarray of shape (24,) with hourly Ta values in Celsius
    """
    key = MONTH_ALIASES.get(month.lower(), month.lower())
    if key not in TMY3_PROFILES:
        available = list(TMY3_PROFILES.keys())
        raise ValueError(f'Month {month!r} not available. Available: {available}')
    return np.array(TMY3_PROFILES[key])


def get_15min_profile(month: str) -> np.ndarray:
    """
    Interpolate hourly profile to 15-minute resolution (96 steps/day).

    Returns:
        np.ndarray of shape (96,)
    """
    hourly = get_hourly_profile(month)
    xp  = np.arange(24)
    xnew = np.linspace(0, 23, 96)
    return np.interp(xnew, xp, hourly)


def get_daily_mean(month: str) -> float:
    """Return the daily mean ambient temperature for the given month."""
    return float(get_hourly_profile(month).mean())


def get_peak_temperature(month: str) -> float:
    """Return the daily peak ambient temperature for the given month."""
    return float(get_hourly_profile(month).max())


if __name__ == '__main__':
    print('Jeddah TMY3 Temperature Profiles (IWEC 41024)')
    for m in ['jan', 'apr', 'jul', 'oct']:
        profile = get_hourly_profile(m)
        print(f'  {m.upper():3s}  mean={profile.mean():.1f}C  '
              f'min={profile.min():.1f}C  max={profile.max():.1f}C')
