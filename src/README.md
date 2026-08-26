# Source code

This directory contains two equivalent implementations of the same RC-simulation
PI-PPO pipeline. Both produce identical baselines and consistent results.

## Single-file version (recommended for reproduction)

- **`simulator.py`** — a self-contained implementation of the entire pipeline (RC model,
  tariff, baselines, DP expert, behaviour cloning, PPO, evaluation) in one file. This is the
  easiest way to reproduce the paper's numbers:
  ```
  python simulator.py --full
  ```

## Modular version (for reading and extension)

The same pipeline split into components, for readability and reuse:

| File | Purpose |
|---|---|
| `rc_sim.py`     | RC thermal environment, tariff, `run_month` evaluation, On/Off and GS baselines |
| `train_real.py` | PPO agent training loop, observation construction, physics-informed reward wiring |
| `best_case.py`  | Per-zone dynamic-programming expert + behaviour-cloning warm-start |
| `agent.py`      | PPO actor-critic network with top-k projection |
| `reward.py`     | Physics-informed reward components (peak, tariff, comfort, physics, switching) |
| `scheduler.py`  | Lazy/greedy scheduling baseline (GS) |
| `surrogate.py`  | Neural surrogate for feasibility/concurrency estimation |
| `utils.py`      | Thermal-parameter helper functions |

The modular `rc_sim.py` produces the same On/Off baseline (e.g. villa July: 3912 kWh / 704 SAR)
as `simulator.py`, confirming the two implementations agree.

## Important note on scope

This is a **lumped-parameter RC thermal simulation implemented in NumPy/PyTorch**. It does
**not** use EnergyPlus, Sinergym, or any detailed building-physics engine. The paper describes
the model as such, and validation against a higher-fidelity simulator is stated as
future work. (An earlier `environment.py` wrapper that referenced EnergyPlus/Sinergym was **not**
part of the actual pipeline and has been removed; no module depends on it.)

## Requirements

Python >= 3.10, `numpy`, `pyyaml`, `torch`. See `../requirements.txt`.
