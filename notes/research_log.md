# Research Log — PI-PPO Project

**PI:** Hamzah Faraj, Taif University

---

## 2025-Q3 — Problem Formulation

- Identified synchronization problem in Saudi residential HVAC: multiple On/Off splits activate simultaneously during afternoon peak (July Ta ≈ 43°C)
- Confirmed 2-tier tariff structure: 0.18 SAR/kWh ≤6,000 kWh / 0.30 SAR/kWh above
- Surveyed existing DRL-HVAC literature: all use flat-rate tariffs, no formal feasibility analysis
- Identified gap: PINN methods achieve 78–90% reductions but only under demand-charge ($/kW) tariffs
- Decision: target tiered per-kWh pricing — more relevant to 60% of Saudi residential market

## 2025-Q3 — System Modelling

- Derived RC thermal model with inter-zone coupling (Eq. 1)
- Calibrated 5-zone villa parameters from Saudi building codes and prior literature
- Established Jeddah TMY3 weather profiles (IWEC 41024) for 4 representative months
- Worked example (Living Room, July): cycle period ≈77 min, duty cycle ≈57%, monthly ≈739 kWh

## 2025-Q3 — Task Model & Feasibility Analysis

- Formalized minimum utilization d_i (Eq. 7): cooling-mode analogue of green scheduling
- Key result: d_i(h=25) = 0.48, d_i(h=26) = 0.30 → 36.9% reduction from ±1°C extension
- Infeasibility theorem: sum(d_i) > k → no safe schedule exists
- Comfort extension from [23,25] to [22,26]°C changes k from 3 to 2 for 5 zones
- Validated against ASHRAE Standard 55 adaptive comfort model

## 2025-Q4 — PI-PPO Design

- Designed 6-component reward (Eq. 9): peak, tariff, comfort, physics, feasibility, switching
- Physics reward r_physics = heat balance residual (key differentiator from standard DRL)
- Architecture: 2×256 MLP, Bernoulli output heads, top-k projection for concurrency constraint
- NN surrogate: 3-layer 128-256-128, pre-trained on 50,000 samples, replaces combinatorial search
- PPO hyperparameters: α=3e-4, γ=0.99, ε=0.2, 10 epochs per update

## 2025-Q4 — Simulation & Results

- EnergyPlus 23.2 via Sinergym, Jeddah TMY3, ΔT=15 min, Hardware: NVIDIA A100, PyTorch 2.1
- 5-zone villa results: peak demand 9.0→5.4 kW (40% reduction), July cost 711→550 SAR (22.6%)
- 20-zone compound with comfort extension: 4020→2042 SAR (49.2% reduction)
- Scalability: <0.04s inference at 100 zones vs 17s for GS, >3600s for MPC
- DQN comfort violation: 0.14°C — PI-PPO: ≤0.02°C at all times

## 2026-Q1 — Paper Writing & Submission

- Full manuscript written targeting Applied Energy
- All 35 references verified (DOIs checked, 2022+ where possible)
- Ablation: 6.3 pp physics, 4.8 pp tariff, 2.4 pp coupling, 16.4 pp comfort extension
- Submitted to Applied Energy — under review
- Repository created: https://github.com/drhamzahfaraj/pi-ppo-green-scheduling

---

## Key Decisions Log

| Decision | Rationale |
|---|---|
| On/Off units only (not inverter) | 60% of Saudi market; discrete action space tractable |
| 4 months (Jan/Apr/Jul/Oct) | Seasonal diversity; Jul is critical (43°C peak) |
| Strict [23-25]°C + Extended [22-26]°C | ASHRAE support; tests comfort flexibility tradeoff |
| k = floor(n/2) for strict | Derived from feasibility theorem (d_i ≈ 0.48 each) |
| Top-k projection | Ensures hard concurrency constraint at inference time |
| PPO over SAC/TD3 | Literature review: PPO best for discrete scheduling |
