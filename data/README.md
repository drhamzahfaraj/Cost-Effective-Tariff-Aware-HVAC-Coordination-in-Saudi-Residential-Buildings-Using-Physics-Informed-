# Weather data

Diurnal ambient-temperature profiles used to drive the RC simulation.

## Format

Each CSV has three columns:

| Column   | Meaning |
|----------|---------|
| `month`  | Month name (January, April, July, October) |
| `hour`   | Time of day in hours, in 0.25 (15-minute) steps: 0.00, 0.25, … 23.75 |
| `T_a_C`  | Ambient (outdoor) temperature in °C at that time |

Each month has 96 rows (24 h × 4 steps/h), matching the simulation's 15-minute control step.

## Files

| File | Climate | July range | January range | Used in paper |
|---|---|---|---|---|
| `jeddah_ambient_profiles.csv` | Coastal, hot-humid | ~29–43 °C | ~18–29 °C | **Yes** (paper scope) |
| `riyadh_ambient_profiles.csv` | Inland, larger diurnal swing | ~28–44 °C | ~8–20 °C | Optional (not in paper) |
| `taif_ambient_profiles.csv`   | High-altitude, cooler | ~20–33 °C | ~5–19 °C | Optional (not in paper) |

**Jeddah** is the climate used for all results in the paper. The Riyadh and Taif profiles are
included as optional extras for users who wish to explore other Saudi climates; they are **not**
part of the reported results. (The paper notes as a hypothesis that cooler locations such as Taif
may show a different savings profile, but this is stated as future work, not a measured result.)

## Provenance

These are representative diurnal profiles constructed to reflect the characteristic seasonal and
daily temperature patterns of each city. They are intended for reproducible relative comparison of
control strategies, not as calibrated meteorological records. For calibrated studies, users should
substitute measured or TMY weather data for the target location (the loader in `rc_sim.py` /
`simulator.py` accepts any CSV in the format above).

## How the loader uses these

`load_weather(month, csv_path)` reads the 96 rows for the requested month and returns the
temperature array that drives the outdoor-conduction term in the RC dynamics. To use a different
city, drop a CSV in this folder in the same format and pass `--city <name>` (single-file) or the
path (modular).
