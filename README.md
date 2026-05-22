# PI-PPO: Physics-Informed Deep Reinforcement Learning for Peak Demand Reduction in Saudi Residential Buildings

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-brightgreen)](requirements.txt)

## Author

**Hamzah Faraj**
Department of Science and Technology, Ranyah College, Taif University, Taif 21944, Saudi Arabia  
f.hamzah@tu.edu.sa
ORCID ID:https://orcid.org/0009-0009-8832-0407
---

## Repository Structure

```
pi-ppo-green-scheduling/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── paper/
│   ├── main.tex                 # Manuscript source (LaTeX)
│   ├── references.bib           # Bibliography (36 references, all DOI-verified)
│   └── response_letter.md       # Reviewer response letter
├── src/
│   ├── train_pippo.py           # Main training script
│   ├── environment.py           # Sinergym environment wrapper with coupling
│   ├── reward.py                # 6-component physics-informed reward
│   ├── surrogate.py             # NN surrogate for feasibility estimation
│   ├── scheduler.py             # Lazy scheduling controller (GS baseline)
│   ├── agent.py                 # PPO agent with top-k projection
│   ├── evaluate.py              # Evaluation and billing computation
│   └── utils.py                 # Thermal parameter utilities
├── configs/
│   ├── hyperparameters.yaml     # Training hyperparameters
│   ├── villa_5zone.yaml         # 5-zone villa configuration
│   ├── compound_20zone.yaml     # 20-zone compound configuration
│   └── jeddah_weather.epw       # → Placeholder: download from EnergyPlus weather DB
├── results/
│   ├── seasonal_5zone.csv       # Table 2: 5-zone seasonal results
│   ├── seasonal_20zone.csv      # Table 3: 20-zone seasonal results
│   ├── july_full_comparison.csv  # Table 4: Full baseline comparison (July)
│   ├── comfort_extension.csv    # Table 5: Comfort range extension results
│   ├── ablation.csv             # Table 6: Component ablation study
│   ├── scalability.csv          # Table 8: Scalability results
│   ├── waterfall.csv            # Table 9: Waterfall decomposition
│   └── sensitivity.csv          # Sensitivity analysis on K_i, C_i
└── data/
    └── README_weather.md        # Instructions for obtaining Jeddah TMY3 data
```

---

## Abstract

When multiple On/Off split air-conditioning units in Saudi residential buildings activate simultaneously, the resulting peak demand spike stresses the electrical grid and inflates monthly bills under the kingdom's two-tier tariff (0.18 SAR/kWh ≤ 6,000 kWh; 0.30 SAR/kWh above). This paper proposes a Physics-Informed Proximal Policy Optimization (PI-PPO) framework that learns a *stationary* scheduling policy — applicable over an infinite time horizon — to coordinate 18,500 BTU On/Off split units (1.8 kW input, EER 10.25) across multiple zones. PI-PPO embeds heat balance equations directly into the RL reward, yielding near-zero comfort violations. Simulations using EnergyPlus with Jeddah weather data across four representative months show that PI-PPO reduces peak demand by 40–60% and July cost by 22.5% for a 5-zone villa, rising to 47.0% for a 20-zone compound with ±1°C comfort extension.

---

## Key Results and Findings

### Headline Results
| Configuration | Comfort Range | July Cost Reduction | Peak Demand Reduction |
|--------------|---------------|--------------------|-----------------------|
| 5-zone villa | Strict [23–25°C] | 22.5% | 40% |
| 5-zone villa | Extended [22–26°C] | 31.9% | 60% |
| 20-zone compound | Strict [23–25°C] | 33.0% | 50% |
| 20-zone compound | Extended [22–26°C] | 47.0% | 65% |

### Ablation Study (20-zone, strict, July)
| Variant | Bill (SAR) | Reduction (%) | P_peak (kW) | Violation (°C) |
|---------|-----------|--------------|-------------|----------------|
| **PI-PPO (full, coupled)** | **2,693** | **33.0** | **18.0** | **0.02** |
| w/o r_physics | 2,935 | 27.0 | 21.6 | 0.10 |
| w/o r_peak | 2,854 | 29.0 | 25.2 | 0.02 |
| w/o r_feas | 2,794 | 30.5 | 19.8 | 0.05 |
| w/o coupling | 2,774 | 31.0 | 19.8 | 0.02 |
| w/o tariff awareness | 2,874 | 28.5 | 18.0 | 0.02 |
| w/o physics & feas | 3,276 | 18.5 | 25.2 | 0.18 |

### Benchmarking Against Published DRL Methods
| Reference | Method | Metric | Savings |
|-----------|--------|--------|---------|
| Lu et al. (2024) | PPO+PID | Energy | 5.3% |
| Brandi et al. (2024) | TD3 | Energy | 17.0% |
| Guo et al. (2025) | DRL (SAC) | Energy | 21.4% |
| Chen et al. (2023) | PINN-RC | Peak† | 78% |
| Xiao et al. (2024) | ModNN | Peak† | 90% |
| **This work** | **PI-PPO** | **Peak** | **40–60%** |
| **This work** | **PI-PPO** | **Cost (tiered)** | **22.5–47.0%** |

†Measured under demand-charge tariffs ($/kW penalty), not tiered ($/kWh).

---

## Contributions

1. **Task model with feasibility analysis:** Each zone and its On/Off unit abstracted as a scheduling task with formally analyzed minimum utilization (d_i) and feasibility conditions, adapted for cooling mode
2. **PI-PPO framework:** Physics-informed reward embedding heat balance equations, schedulability conditions, and tiered-tariff awareness into PPO
3. **Inter-zone thermal coupling:** Exploits thermal buffering through shared walls for 2.0 pp additional savings
4. **Comfort extension analysis:** Formal proof that ±1°C comfort relaxation reduces d_i by 36.9%
5. **Year-round evaluation:** Four representative months (Jan/Apr/Jul/Oct) with realistic Saudi AC specifications

---

## Methodology: Mechanism of Energy Savings

```
┌─────────────────────────────────────────────────────────┐
│                    PI-PPO Architecture                    │
│                                                          │
│   NN Surrogate ──→ PPO Agent ──→ Building Environment   │
│   (k, Q_i)          (m(t))       (x(t), E_cum)         │
│                       ↑                  │               │
│                       └── PI Reward + ───┘               │
│                           Tariff Signal                  │
└─────────────────────────────────────────────────────────┘
```

**How PI-PPO reduces costs:**

1. **Scheduling coordination (21.0 pp):** Limits simultaneous compressors to k units, preventing coincident demand spikes
2. **Anticipatory pre-cooling (4.0 pp):** PPO learns to pre-cool zones during cooler morning hours, building thermal reserves before the afternoon peak
3. **Tariff-tier management (4.0 pp):** Tracks cumulative kWh relative to the 6,000 kWh threshold, reducing Tier-2 exposure
4. **Physics-informed guidance (2.0 pp):** Heat balance residual provides dense per-step feedback for physically plausible decisions
5. **Inter-zone buffering (2.0 pp):** Sequences cooling so a cooled zone's shared wall slows its neighbor's temperature rise
6. **Comfort extension (14.0 pp):** Wider [22–26°C] band reduces minimum utilization d_i by 36.9%, permitting fewer simultaneous compressors (design parameter, not algorithmic contribution)

### Feature Selection (State Space)

The state vector s(t) = [x(t), m(t−1), T_a(t), E_cum(t), k, Δx(t), t_hour] was selected based on:

| Feature | Justification |
|---------|---------------|
| x(t): Zone temperatures | Direct observation of comfort state; required for safety constraint |
| m(t−1): Previous modes | Prevents excessive compressor cycling; enables switching penalty |
| T_a(t): Outdoor temperature | Primary heat gain driver; determines cooling demand intensity |
| E_cum(t): Cumulative consumption | Enables tariff-tier tracking relative to 6,000 kWh threshold |
| k: Concurrency limit | From surrogate; constrains action space via top-k projection |
| Δx(t): Temperature derivatives | Provides rate-of-change information for anticipatory pre-cooling |
| t_hour: Hour of day | Captures diurnal patterns; enables time-aware scheduling |

Features excluded: humidity (On/Off units have no humidity control), solar irradiance (captured implicitly via T_a), occupancy schedules (assumed 24/7 operation in Saudi residential context during summer).

---

## Dataset

### Weather Data
- **Source:** Jeddah TMY3 (Typical Meteorological Year), IWEC station 41024
- **Download:** [EnergyPlus Weather Database](https://energyplus.net/weather) → Search "Jeddah" → Download `.epw` file
- **Months evaluated:** January, April, July, October (30-day billing cycles each)

### Building Simulation
- **Engine:** EnergyPlus 23.2 via Sinergym
- **AC Units:** 18,500 BTU, 1.8 kW electrical input, 5.3 kW cooling output, EER 10.25
- **Tariff:** Saudi two-tier: 0.18 SAR/kWh ≤ 6,000 kWh, 0.30 SAR/kWh above

---

## Running Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train PI-PPO on 5-zone villa (July)
python src/train_pippo.py --config configs/villa_5zone.yaml --month july --seed 0

# Train on 20-zone compound
python src/train_pippo.py --config configs/compound_20zone.yaml --month july --seed 0

# Reproduce all results (5 seeds × 4 months × 2 scales)
for seed in 0 1 2 3 4; do
  for month in january april july october; do
    python src/train_pippo.py --config configs/villa_5zone.yaml --month $month --seed $seed
    python src/train_pippo.py --config configs/compound_20zone.yaml --month $month --seed $seed
  done
done
```

### Evaluation

```bash
# Evaluate trained model and compute billing
python src/evaluate.py --model checkpoints/pippo_5z_july_seed0.pt --month july --config configs/villa_5zone.yaml

# Run ablation study
python src/evaluate.py --ablation --config configs/compound_20zone.yaml --month july

# Run sensitivity analysis
python src/evaluate.py --sensitivity --config configs/villa_5zone.yaml --month july
```

---

## Limitations

1. First-order lumped thermal model does not capture radiative transfer, humidity, or furniture thermal mass
2. All results are simulation-based (EnergyPlus + TMY3); no calibration against measured data
3. Four-month evaluation does not demonstrate month-to-month transitions
4. Top-k projection is heuristic (0.02°C transient violation possible); not a formal safety certificate
5. Extended comfort range [22–26°C] requires occupant acceptance validation in Saudi households
6. Limited to On/Off units; inverter-driven units need continuous action spaces
7. Coupling coefficients K_ij estimated, not measured in situ
8. Stronger baselines (MILP, constrained RL) would further validate improvements

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

The author would like to acknowledge the Deanship of Graduate Studies and Scientific Research, Taif University, for funding this work.

---

## Citation

```bibtex
@article{faraj2026pippo,
  author  = {Faraj, Hamzah},
  title   = {Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential
             Buildings Using Physics-Informed Deep Reinforcement Learning},
  journal = {Journal of King Saud University -- Engineering Sciences},
  year    = {2026},
  note    = {Under review}
}
```
