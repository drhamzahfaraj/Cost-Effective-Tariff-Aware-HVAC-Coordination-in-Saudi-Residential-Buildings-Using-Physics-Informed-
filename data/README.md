# Data

## 1. Weather: Jeddah TMY3 (IWEC Station 41024)
Download the EPW from https://energyplus.net/weather (search "Jeddah"):
`SAU_Jeddah.412024_IWEC.epw`. Not redistributed here (license).

| Month | T_min (C) | T_max (C) | T_mean (C) |
|---|---|---|---|
| January | 18 | 29 | 23.5 |
| April | 23 | 35 | 29 |
| July | 29 | 43 | 36 |
| October | 25.5 | 37 | 31.25 |

## 2. jeddah_ambient_profiles.csv (384 rows)
15-minute diurnal ambient temperature profiles for the four evaluation months
(sinusoidal fit to TMY3 monthly extremes; peak ~15:00, minimum ~03:00).
These are the profiles plotted in paper Figure 1 and used by the RC model.
Columns: month, hour, T_a_C.

## 3. energyplus_trace_july_5zone_onoff.csv (2,880 rows = 30 days x 96 steps)
Representative July trace for the 5-zone villa under On-Off thermostatic
control (strict band [23,25] C). Per zone: RC-model temperature, an
EnergyPlus-style temperature trace (model + calibrated AR(1) residual matched
to the per-zone RMSE of paper Table 10: 0.22-0.31 C), compressor mode, and
aggregate electrical power (1.8 kW per active unit; peak 9.0 kW).

Provenance note: integrating this simplified trace gives 3,890 kWh for July;
the paper's headline figures come from the full EnergyPlus co-simulation
(model 3,950 kWh vs EnergyPlus 3,854 kWh, error 2.5% -- Table 12). This file
is a reproducible extract for inspection and plotting, not the source of the
aggregated tables; those are in ../results/.

## 4. surrogate_training_sample.csv (500 rows)
A 500-row sample of the 50,000-point Latin-hypercube surrogate training set
(paper Section 6). Inputs: T_a in [18,48] C, K_i in [0.18,0.35] kW/K,
C_i in [1800,4200] kJ/K, band in {[23,25],[22,26]}. Targets computed
analytically: d_i (Eq. 7), Q_i* = d_i x |Q_cool|, and k* = ceil(5 x d_i)
for the homogeneous 5-zone illustration.
