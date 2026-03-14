# PI-PPO: Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings

> **Physics-Informed Proximal Policy Optimization (PI-PPO)** — A deep reinforcement learning framework for coordinating On/Off split AC units in Saudi residential buildings under the kingdom's two-tier electricity tariff.

---

## 📄 Paper

**Title:** Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings Using Physics-Informed Deep Reinforcement Learning  
**Author:** Hamzah Faraj — Department of Science and Technology, Ranyah College, Taif University, Taif 21944, Saudi Arabia  
**Submitted to:** *Applied Energy*  
**Status:** Under Review

---

## 📋 Abstract

When multiple On/Off split air-conditioning units in Saudi residential buildings activate simultaneously, the resulting peak demand spike stresses the electrical grid and inflates monthly bills under the kingdom's two-tier tariff (0.18 SAR/kWh ≤ 6,000 kWh; 0.30 SAR/kWh above). This paper proposes a **Physics-Informed Proximal Policy Optimization (PI-PPO)** framework that learns a *stationary* scheduling policy — applicable over an infinite time horizon without re-solving any optimization — to coordinate 18,500 BTU On/Off split units (1.8 kW input, EER 10.25) across multiple zones.

Key results:
- **Peak demand reduced by 40–60%** across all seasons (Jeddah TMY3 data)
- **July electricity cost reduced by 22.6%** for a 5-zone villa
- **49.2% cost reduction** for a 20-zone compound with comfort extension
- **Near-zero comfort violations** (≤ 0.02°C) guaranteed via physics-informed reward
- **Sub-linear scaling**: inference < 0.04 s at 100 zones

---

## 🗂️ Repository Structure

```
pi-ppo-green-scheduling/
├── README.md                   # This file
├── paper/
│   ├── main.tex                # Main LaTeX manuscript
│   ├── references.bib          # BibTeX references (35 entries)
│   └── figures/                # TikZ figures (embedded in LaTeX)
├── experiments/
│   ├── scripts/
│   │   ├── train_pi_ppo.py     # PI-PPO training script
│   │   ├── baselines.py        # Baseline controllers (GS, PPO, DQN, MPC)
│   │   └── evaluate.py         # Evaluation and metrics
│   ├── data/
│   │   ├── thermal_params.py   # Building thermal parameters
│   │   └── jeddah_tmm3.py      # Jeddah TMY3 ambient temperature data
│   └── results/
│       ├── table2_villa5z.csv  # 5-zone villa year-round results
│       ├── table3_july_baselines.csv  # Full baseline comparison (July)
│       ├── table4_scalability.csv     # Scalability results
│       └── table5_waterfall.csv       # Waterfall decomposition
└── notes/
    ├── outline.md              # Paper outline and section notes
    └── research_log.md         # Research progress log
```

---

## ⚙️ Method Overview

PI-PPO replaces the classical reactive lazy scheduler with a learned stationary policy. The framework has four key features:

1. **Infinite-horizon stationary policy** — No re-solving at each step (unlike MPC)
2. **Physics-informed reward** — Embeds heat balance equations (Eq. 1) directly into the RL reward for near-zero comfort violations
3. **Inter-zone thermal coupling** — Exploits thermal buffering through shared walls
4. **Tiered-tariff awareness** — Tracks cumulative consumption relative to the 6,000 kWh threshold

### Reward Components

| Component | Symbol | Role |
|---|---|---|
| Peak penalty | r_peak | Penalizes coincident compressor activation (quadratic) |
| Tariff penalty | r_tariff | Applies marginal rate (0.18 or 0.30 SAR/kWh) |
| Comfort penalty | r_comfort | Penalizes temperature bound excursions |
| Physics residual | r_physics | Heat balance consistency (key differentiator) |
| Feasibility penalty | r_feas | Penalizes proximity to infeasibility boundary |
| Switching penalty | r_switch | Penalizes unnecessary compressor cycling |

---

## 🏗️ Building Setup

**5-Zone Villa (Saudi Arabia)** — All units: 18,500 BTU, 1.8 kW input, 5.3 kW cooling, EER 10.25

| Zone | Area (m²) | Ki (kW/K) | Ci (kJ/K) | Adjacent |
|---|---|---|---|---|
| Dining Room | 30 | 0.28 | 3,600 | Living |
| Living Room | 25 | 0.25 | 3,000 | Majlis, Master |
| Master Bedroom | 25 | 0.24 | 3,000 | Living, Bed.2 |
| Boys Bedroom | 20 | 0.22 | 2,400 | Master, Bed.3 |
| Girls Bedroom | 20 | 0.22 | 2,400 | Bed.2 |

---

## 📊 Key Results

### Year-Round Performance (5-Zone Villa, Strict [23–25°C])

| Month | Method | Bill (SAR) | Reduction | E_tot (kWh) | P_peak (kW) |
|---|---|---|---|---|---|
| Jan | On-Off | 93 | — | 520 | 9.0 |
| Jan | PI-PPO | 83 | **10.8%** | 460 | 3.6 |
| Apr | On-Off | 408 | — | 2,270 | 9.0 |
| Apr | PI-PPO | 327 | **19.9%** | 1,820 | 5.4 |
| Jul | On-Off | 711 | — | 3,950 | 9.0 |
| Jul | PI-PPO | 550 | **22.6%** | 3,060 | 5.4 |
| Oct | On-Off | 490 | — | 2,720 | 9.0 |
| Oct | PI-PPO | 392 | **20.0%** | 2,180 | 5.4 |

### Ablation (20-Zone, July, Strict Range)

| Configuration | Bill (SAR) | P_peak (kW) | Cumul. Reduction |
|---|---|---|---|
| On-Off (baseline) | 4,020 | 36.0 | 0% |
| + GS coordination | 3,180 | 21.6 | +20.9 pp |
| + PPO learning | 3,018 | 21.6 | +24.9 pp |
| + Tariff awareness | 2,850 | 21.6 | +29.1 pp |
| + Physics reward | 2,772 | 19.8 | +31.0 pp |
| + Inter-zone coupling | 2,676 | 18.0 | **+33.4 pp** |
| + Comfort extension [22–26°C] | 2,042 | 12.6 | **+49.2 pp** |

---

## 🚀 Quickstart

### Requirements

```bash
pip install torch sinergym gymnasium numpy pandas matplotlib
```

EnergyPlus 23.2 must be installed separately: https://energyplus.net/

### Train PI-PPO

```bash
cd experiments/scripts
python train_pi_ppo.py --zones 5 --months july --comfort strict
```

### Evaluate All Baselines

```bash
python evaluate.py --zones 5 --months jan apr jul oct --output ../results/
```

### Reproduce Scalability Table

```bash
python evaluate.py --zones 5 20 50 100 --months july --output ../results/table4_scalability.csv
```

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{faraj2025pippo,
  author  = {Hamzah Faraj},
  title   = {Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings Using Physics-Informed Deep Reinforcement Learning},
  journal = {Applied Energy},
  year    = {2025},
  note    = {Under Review}
}
```

---

## 🏛️ Acknowledgments

The author acknowledges the Deanship of Graduate Studies and Scientific Research, Taif University, for funding this work.

---

## 📜 License

This repository is for academic reproducibility. Code is released under the MIT License. The manuscript is copyright of the author.
