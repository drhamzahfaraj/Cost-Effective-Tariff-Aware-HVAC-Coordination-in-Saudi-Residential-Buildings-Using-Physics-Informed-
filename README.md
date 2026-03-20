# PI-PPO: Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings

> **Physics-Informed Proximal Policy Optimization (PI-PPO)** — A deep reinforcement learning framework for coordinating On/Off split AC units in Saudi residential buildings under the kingdom's two-tier electricity tariff.

---

## Paper

**Title:** Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings Using Physics-Informed Deep Reinforcement Learning  
**Author:** Hamzah Faraj — Department of Science and Technology, Ranyah College, Taif University, Taif 21944, Saudi Arabia  

---

## Abstract

When multiple On/Off split air-conditioning units in Saudi residential buildings activate simultaneously, the resulting peak demand spike stresses the electrical grid and inflates monthly bills under the kingdom's two-tier tariff (0.18 SAR/kWh ≤ 6,000 kWh; 0.30 SAR/kWh above). This paper proposes a **Physics-Informed Proximal Policy Optimization (PI-PPO)** framework that learns a *stationary* scheduling policy — applicable over an infinite time horizon without re-solving any optimization — to coordinate 18,500 BTU On/Off split units (1.8 kW input, EER 10.25) across multiple zones.

Key results:
- **Peak demand reduced by 40–60%** across all seasons (Jeddah TMY3 data)
- **July electricity cost reduced by 22.5%** for a 5-zone villa (strict [23–25°C])
- **47.0% cost reduction** for a 20-zone compound with comfort extension [22–26°C]
- **Near-zero comfort violations** (≤ 0.02°C) guaranteed via physics-informed reward
- **Sub-linear scaling**: inference < 0.04 s at 100 zones
- **Ablation**: 6.0 pp from physics reward, 4.5 pp from tariff awareness, 2.0 pp from inter-zone coupling, 14.0 pp from comfort extension

---

## Repository Structure

```
pi-ppo-green-scheduling/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── paper/
│   ├── main.tex                     # Full LaTeX manuscript
│   └── references.bib               # BibTeX references (35 entries)
├── experiments/
│   ├── scripts/
│   │   ├── train_pi_ppo.py          # PI-PPO training script
│   │   ├── baselines.py             # Baseline controllers (GS, PPO, DQN, MPC)
│   │   ├── evaluate.py              # Evaluation and metrics
│   │   └── nn_surrogate.py          # NN surrogate pre-training
│   ├── data/
│   │   ├── thermal_params.py        # Building thermal parameters (5-zone & 20-zone)
│   │   └── jeddah_tmy3.py           # Jeddah TMY3 ambient temperature profiles
│   └── results/
│       ├── table1_villa5z.csv       # 5-zone villa year-round results (Tab. 2)
│       ├── table2_compound20z.csv   # 20-zone compound year-round results (Tab. 3)
│       ├── table3_july_baselines.csv# Full baseline comparison July 5-zone (Tab. 4)
│       ├── table4_comfort_ext.csv   # Comfort extension results (Tab. 5)
│       ├── table5_ablation.csv      # Ablation study 20-zone July (Tab. 6)
│       ├── table6_scalability.csv   # Scalability results (Tab. 7)
│       └── table7_waterfall.csv     # Waterfall decomposition (Tab. 8)
└── notes/
    ├── outline.md                   # Paper outline and section notes
    └── research_log.md              # Research progress log
```

---

## Method Overview

PI-PPO replaces the classical reactive lazy scheduler with a learned stationary policy. The framework has four key features:

1. **Infinite-horizon stationary policy** — No re-solving at each step (unlike MPC)
2. **Physics-informed reward** — Embeds heat balance equations (Eq. 1) directly into the RL reward for near-zero comfort violations
3. **Inter-zone thermal coupling** — Exploits thermal buffering through shared walls (K_ij terms)
4. **Tiered-tariff awareness** — Tracks cumulative consumption relative to the 6,000 kWh threshold

### Reward Components

| Component | Symbol | Formula | Role |
|---|---|---|---|
| Peak penalty | r_peak | −ω₀·(Σ 1.8·mᵢ)² | Penalizes coincident activation (quadratic) |
| Tariff penalty | r_tariff | −ω₁·p(E_cum)·1.8·Σmᵢ·ΔT | Applies marginal rate (0.18 or 0.30 SAR/kWh) |
| Comfort penalty | r_comfort | −ω₂·Σ[max(0,xᵢ−h)+max(0,l−xᵢ)] | Penalizes temperature excursions |
| Physics residual | r_physics | −ω₃·Σ|ẋᵢ_obs−ẋᵢ_pred| | Heat balance consistency (key differentiator) |
| Feasibility penalty | r_feas | −ω₄·max(0,d̂−k+0.5) | Penalizes proximity to infeasibility boundary |
| Switching penalty | r_switch | −ω₅·Σ|mᵢ(t)−mᵢ(t−1)| | Penalizes unnecessary compressor cycling |

### Reward Weights

| Weight | Value | Component |
|---|---|---|
| ω₀ | 1.0 | Peak penalty |
| ω₁ | 0.8 | Tariff penalty |
| ω₂ | 5.0 | Comfort penalty |
| ω₃ | 2.0 | Physics residual |
| ω₄ | 1.5 | Feasibility penalty |
| ω₅ | 0.3 | Switching penalty |

---

## Building Setup

**5-Zone Villa (Saudi Arabia)** — All units: 18,500 BTU, 1.8 kW input, 5.3 kW cooling, EER 10.25

| Zone | Area (m²) | Kᵢ (kW/K) | Cᵢ (kJ/K) | Kᵢⱼ (kW/K) | Adjacent to |
|---|---|---|---|---|---|
| Dining Room | 30 | 0.28 | 3,600 | 0.05 | Living |
| Living Room | 25 | 0.25 | 3,000 | 0.05 | Majlis, Master |
| Master Bedroom | 25 | 0.24 | 3,000 | 0.04 | Living, Bed. 2 |
| Boys Bedroom 2 | 20 | 0.22 | 2,400 | 0.04 | Master, Bed. 3 |
| Girls Bedroom 3 | 20 | 0.22 | 2,400 | 0.04 | Bed. 2 |

---

## Key Results

### Year-Round Performance — 5-Zone Villa, Strict [23–25°C]

| Month | Method | Bill (SAR) | Red. (%) | E_tot (kWh) | P_peak (kW) | Viol. (°C) |
|---|---|---|---|---|---|---|
| Jan | On-Off | 94 | — | 520 | 9.0 | 0.00 |
| Jan | GS (lazy) | 86 | 8.5 | 480 | 5.4 | 0.00 |
| Jan | **PI-PPO** | **83** | **11.7** | **460** | **3.6** | **0.00** |
| Apr | On-Off | 409 | — | 2,270 | 9.0 | 0.00 |
| Apr | GS (lazy) | 355 | 13.2 | 1,970 | 5.4 | 0.00 |
| Apr | **PI-PPO** | **328** | **19.8** | **1,820** | **5.4** | **0.00** |
| Jul | On-Off | 711 | — | 3,950 | 9.0 | 0.00 |
| Jul | GS (lazy) | 621 | 12.7 | 3,450 | 5.4 | 0.00 |
| Jul | **PI-PPO** | **551** | **22.5** | **3,060** | **5.4** | **0.00** |
| Oct | On-Off | 490 | — | 2,720 | 9.0 | 0.00 |
| Oct | GS (lazy) | 425 | 13.3 | 2,360 | 5.4 | 0.00 |
| Oct | **PI-PPO** | **392** | **20.0** | **2,180** | **5.4** | **0.00** |

### Full Baseline Comparison — 5-Zone Villa, July, Strict [23–25°C]

| Method | Bill (SAR) | Red. (%) | E_tot (kWh) | P_peak (kW) | Viol. (°C) |
|---|---|---|---|---|---|
| On-Off | 711 | — | 3,950 | 9.0 | 0.00 |
| MPC-1 | 632 | 11.1 | 3,510 | 7.2 | 0.00 |
| MPC-3 | 601 | 15.5 | 3,340 | 5.4 | 0.00 |
| GS (lazy) | 621 | 12.7 | 3,450 | 5.4 | 0.00 |
| DQN | 666 | 6.3 | 3,700 | 7.2 | 0.14 |
| PPO (standard) | 590 | 17.0 | 3,280 | 5.4 | 0.05 |
| PI-PPO (no coupling) | 567 | 20.3 | 3,150 | 5.4 | 0.00 |
| **PI-PPO (coupled)** | **551** | **22.5** | **3,060** | **5.4** | **0.00** |

### Ablation — 20-Zone Compound, July, Strict [23–25°C]

| Variant | Bill (SAR) | Red. (%) | P_peak (kW) | Viol. (°C) |
|---|---|---|---|---|
| **PI-PPO (full, coupled)** | **2,693** | **33.0** | **18.0** | **0.02** |
| w/o r_physics | 2,935 | 27.0 | 21.6 | 0.10 |
| w/o r_peak | 2,854 | 29.0 | 25.2 | 0.02 |
| w/o r_feas | 2,794 | 30.5 | 19.8 | 0.05 |
| w/o coupling | 2,774 | 31.0 | 19.8 | 0.02 |
| w/o tariff awareness | 2,874 | 28.5 | 18.0 | 0.02 |
| w/o physics & feas | 3,276 | 18.5 | 25.2 | 0.18 |

### Waterfall — 20-Zone Compound, July

| Source | Bill (SAR) | P_peak (kW) | Cumul. pp |
|---|---|---|---|
| On-Off (strict) | 4,020 | 36.0 | 0 |
| + Scheduling coordination (GS) | 3,176 | 21.6 | +21.0 |
| + PPO policy learning | 3,015 | 21.6 | +25.0 |
| + Tiered-tariff awareness | 2,854 | 21.6 | +29.0 |
| + Physics-informed reward | 2,774 | 19.8 | +31.0 |
| + Inter-zone coupling | **2,693** | **18.0** | **+33.0** |
| + Comfort extension [22–26°C] | **2,131** | **12.6** | **+47.0** |

---

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

EnergyPlus 23.2 must be installed separately: https://energyplus.net/

### Train PI-PPO

```bash
cd experiments/scripts
python train_pi_ppo.py --zones 5 --months july --comfort strict
python train_pi_ppo.py --zones 20 --months jul --comfort extended
```

### Evaluate All Baselines

```bash
python evaluate.py --zones 5 --months jan apr jul oct --output ../results/
```

### Reproduce Scalability Table

```bash
python evaluate.py --zones 5 20 50 100 --months july --output ../results/table6_scalability.csv
```

### Pre-train NN Surrogate

```bash
python nn_surrogate.py --samples 50000 --save surrogate.pt
```

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{faraj2026pippo,
  author  = {Hamzah Faraj},
  title   = {Cost-Optimal Coordination for Peak Demand Reduction in {Saudi} Residential Buildings
             Using Physics-Informed Deep Reinforcement Learning},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Under Review}
}
```

---

## Acknowledgments

The author acknowledges the Deanship of Graduate Studies and Scientific Research, Taif University, for funding this work.

---

## License

This repository is for academic reproducibility. Code is released under the MIT License. The manuscript is copyright of the author.
