# PI-PPO: Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings

> **Physics-Informed Deep Reinforcement Learning for HVAC Scheduling**

[![Status](https://img.shields.io/badge/status-under%20review-orange)]()
[![Submitted to](https://img.shields.io/badge/submitted%20to-Applied%20Energy-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Abstract

When multiple On/Off split air-conditioning units in Saudi residential buildings activate simultaneously, the resulting peak demand spike stresses the electrical grid and inflates monthly bills under the kingdom's two-tier tariff (0.18 SAR/kWh ≤ 6,000 kWh; 0.30 SAR/kWh above). This paper proposes a **Physics-Informed Proximal Policy Optimization (PI-PPO)** framework that learns a *stationary* scheduling policy—applicable over an infinite time horizon without re-solving any optimization—to coordinate 18,500 BTU On/Off split units (1.8 kW input, EER 10.25) across multiple zones.

Key results:
- **40–60% peak demand reduction** across four seasonal months (Jeddah TMY3)
- **22.6% July cost reduction** for a 5-zone villa (strict comfort range)
- **49.2% cost reduction** for a 20-zone compound with comfort extension [22–26°C]
- Comfort violations ≤ 0.02°C at all times — a guarantee absent from standard DRL
- Sub-linear scalability: < 0.04 s inference at 100 zones

## Paper Status

| Item | Detail |
|------|--------|
| Title | Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings Using Physics-Informed Deep Reinforcement Learning |
| Author | Hamzah Faraj, Taif University |
| Submitted to | *Applied Energy* |
| Status | Under Review |
| DOI | TBD |

## Repository Structure

```
pi-ppo-green-scheduling/
├── README.md                  # This file
├── paper/
│   ├── main.tex               # Main LaTeX manuscript
│   ├── references.bib         # BibTeX references (35 entries)
│   └── figures/               # TikZ figures (embedded in LaTeX)
├── experiments/
│   ├── scripts/
│   │   ├── pi_ppo_agent.py    # PI-PPO agent implementation
│   │   ├── thermal_env.py     # EnergyPlus/Sinergym environment wrapper
│   │   └── train.py           # Training entry point
│   ├── data/
│   │   └── preprocess.py      # TMY3 weather data preprocessing
│   └── results/
│       └── summary_tables.md  # Key result tables from the paper
└── notes/
    ├── research_outline.md    # Paper structure and contribution notes
    └── ablation_notes.md      # Ablation study design notes
```

## Method Overview

PI-PPO replaces the reactive lazy thermostat with a learned stationary policy. The key innovation is the **physics-informed reward**:

```
r(t) = r_peak + r_tariff + r_comfort + r_physics + r_feas + r_switch
```

Where `r_physics = -ω₃ Σᵢ |ẋᵢᵒᵇˢ − ẋᵢᵖʳᵉᵈ|` is the heat-balance residual from the RC thermal model — the key differentiator from standard DRL methods.

## Baselines Compared

| Method | Peak Red. | Cost Red. (July) |
|--------|-----------|------------------|
| On-Off (independent) | 0% | 0% |
| Lazy Scheduling (GS) | ~40% | 12.7% |
| PPO (standard) | ~40% | 17.0% |
| **PI-PPO (ours)** | **40–60%** | **22.6–49.2%** |

## Requirements

```bash
pip install sinergym torch stable-baselines3 numpy pandas matplotlib
```

EnergyPlus 23.2 must be installed separately. See [Sinergym docs](https://ugr-sail.github.io/sinergym/).

## Citation

If you use this work, please cite:

```bibtex
@article{faraj2026pippo,
  author  = {Hamzah Faraj},
  title   = {Cost-Optimal Coordination for Peak Demand Reduction in Saudi Residential Buildings Using Physics-Informed Deep Reinforcement Learning},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgments

This work was funded by the Deanship of Graduate Studies and Scientific Research, Taif University.

## License

MIT License — see [LICENSE](LICENSE).
