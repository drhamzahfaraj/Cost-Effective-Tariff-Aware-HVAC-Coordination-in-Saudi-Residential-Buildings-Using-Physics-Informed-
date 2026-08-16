# Methods (detailed)

This document describes the simulation and the PI-PPO controller in enough detail to understand
and modify the code in `src/simulator.py`.

## 1. Thermal model (RC, lumped-parameter)

Each zone `i` is a single thermal capacitance `C_i` (kJ/K) with conductance `K_i` (kW/K) to
outdoor air, inter-zone coupling `K_ij` through shared walls, an internal gain `Q_int`, and a
cooling term when its unit is ON:

```
C_i dT_i/dt = K_i (T_a - T_i) + sum_j K_ij (T_j - T_i) + Q_int + Q_cool * u_i
```

with `Q_cool = -5.3 kW` (effective cooling), `Q_int = 0.3 kW`, integrated at a 15-minute step
(`DT_H = 0.25 h`, 96 steps/day). This is a first-order model: it captures the dominant time
constant `tau_i = C_i / K_i` but not radiative transfer, furniture mass, humidity/latent load, or
occupancy-driven ventilation. This is a deliberate simplification (see the paper's Limitations).

## 2. Electrical and tariff model

Each unit draws `P_elec = 1.8 kW` when ON. The monthly bill uses the Saudi two-tier inclining-block
tariff:

```
bill(kWh) = 0.18 * min(kWh, 6000) + 0.30 * max(0, kWh - 6000)
```

All bills in the paper are computed from **measured** kWh via this formula, so bill and energy are
consistent by construction.

## 3. Baselines

- **On/Off thermostat:** hysteresis control per zone (turn ON near the upper bound, OFF near the
  lower). Competent and realistic — not a naive fixed schedule. This is the comparison baseline.
- **Greedy / lazy scheduler (GS):** a simple coordination heuristic used as an intermediate
  reference.

## 4. PI-PPO controller

### 4.1 Observation
Per step: normalized zone temperatures, neighbour-average temperatures, previous on/off vector,
normalized ambient temperature, cumulative energy (for tariff-tier awareness), and a
time-of-day encoding (sin/cos).

### 4.2 Physics-informed reward
The reward combines: a peak-power penalty, a tariff-rate-weighted energy cost (using the correct
marginal rate for the current cumulative consumption), a comfort-violation penalty, and a
switching penalty. Because the tariff rate in the reward switches at 6000 kWh, the agent is
directly incentivised to avoid Tier-2 consumption.

### 4.3 Behaviour-cloning warm-start
PPO from scratch converges poorly in this binary, high-dimensional action space. We first solve a
per-zone dynamic-programming (DP) schedule and clone it, then fine-tune with PPO. The DP expert is
a genuine solve (backward induction over a discretised temperature grid), used only to initialise
the policy — the reported performance is PPO's after fine-tuning, evaluated by simulation.

### 4.4 Concurrency limit
A feasibility analysis bounds the number of units that may run simultaneously (`k`). At evaluation,
if more than `k` units are requested, the top-`k` by policy probability are selected (with a hard
safety override forcing ON any zone at its upper bound and OFF any at its lower bound).

## 5. Training regime

- A **separate policy per month** (Jan, Apr, Jul, Oct), each trained with **3 seeds**.
- **1000 PPO fine-tuning episodes** per policy after behaviour cloning.
- Reported values are **means across the 3 seeds**, with standard deviations reported. No
  best-seed selection.

## 6. Evaluation

Each trained policy is rolled out over a 30-day billing cycle. We record: total kWh, bill (via the
tariff), peak kW, fraction of timesteps with any comfort violation, and — importantly — the
**mean and maximum violation magnitude in °C**, since the fraction of timesteps with a violation
can be high while the magnitude remains fractional.

## 7. Known limitations (see paper for full discussion)
- Reduced-order RC model, not validated against a detailed simulator.
- Idealised loads (no occupancy/furniture/humidity/solar); these would increase demand and, if
  anything, push further into Tier-2 (a conservative bias w.r.t. the tariff mechanism).
- On-policy RL seed variance, notably in the 20-zone extended case.
