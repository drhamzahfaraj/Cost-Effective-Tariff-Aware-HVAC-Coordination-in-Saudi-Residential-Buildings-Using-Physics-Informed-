# Paper Outline — PI-PPO for Saudi HVAC Scheduling

**Target journal:** Applied Energy  
**Status:** Under Review  
**Version:** main-13 (final submission)

---

## Section Map

| § | Title | Key Content |
|---|---|---|
| 1 | Introduction | Problem motivation, two-tier tariff, gap analysis, contributions |
| 1.1 | Why Comfort Extension Reduces Peak | d_i derivation intuition, ASHRAE support |
| 2 | Related Work | Saudi energy, DRL-HVAC, PIML, scheduling |
| 3 | System Model | Heat balance Eq.(1), inter-zone coupling K_ij, tariff Eq.(3) |
| 4 | Task Model & Feasibility | Task abstraction, ODE solution Eq.(4-6), d_i formula Eq.(7) |
| 5 | Scheduling Controller | Lazy baseline, coupled system dynamics Eq.(8) |
| 6 | Proposed Methodology: PI-PPO | State, action, reward Eq.(9), PPO training, NN surrogate |
| 7 | Experimental Setup | EnergyPlus 23.2, Sinergym, 5-zone villa Table 1, worked example |
| 8 | Results & Discussion | Year-round, full baseline, comfort ext., ablation, lit. compare, scalability, waterfall |
| 9 | Conclusion | Summary, future work |

---

## Key Claims & Supporting Evidence

### Claim 1: PI-PPO reduces peak demand by 40–60%
- Table 2 (5-zone): P_peak 9.0 → 3.6–5.4 kW
- Table 3 (20-zone): P_peak 36.0 → 7.2–18.0 kW

### Claim 2: Near-zero comfort violations guaranteed
- Table 4: PI-PPO Viol. = 0.00°C vs DQN 0.14°C, PPO 0.05°C
- Physics reward r_physics enforces heat balance consistency

### Claim 3: Comfort extension adds 14.0 pp (36.9% d_i reduction)
- d_i(h=25) = 0.48 → d_i(h=26) = 0.30 → Δ = 36.9%
- k: 3→2 (5-zone), 10→7 (20-zone)
- Table 5: 22.5% → 31.9% (5-zone July), 33.0% → 47.0% (20-zone July)

### Claim 4: Sub-linear scaling (<0.04 s at 100 zones)
- Table 7: GS=17 s, PI-PPO=0.04 s at 100 zones

### Claim 5: Ablation attribution
- 6.0 pp: physics reward r_physics
- 4.5 pp: tiered-tariff awareness r_tariff
- 4.0 pp: PPO policy learning
- 2.0 pp: inter-zone coupling K_ij
- 14.0 pp: comfort extension

---

## Equations Checklist

- Eq.(1): Heat balance with inter-zone coupling
- Eq.(2): Single-zone simplified dynamics
- Eq.(3): Saudi two-tier tariff bill formula
- Eq.(4): Task dynamics (ON/OFF ODE)
- Eq.(5): Temperature rise solution (OFF phase)
- Eq.(6): Temperature fall solution (ON phase)
- Eq.(7): Minimum utilization d_i
- Eq.(8): Coupled system dynamics
- Eq.(9): Physics-informed reward r(t)

---

## Figures Checklist

- Fig. 1: Jeddah 24-h ambient temperature profiles (4 months, TikZ)
- Fig. 2: Cooling task dynamics — strict vs extended comfort ranges (TikZ)
- Fig. 3: PI-PPO architecture diagram (TikZ)

---

## Tables Checklist

- Table 1: 5-zone villa specifications
- Table 2: Year-round results, 5-zone villa
- Table 3: Year-round results, 20-zone compound
- Table 4: Full baseline comparison, 5-zone July
- Table 5: Comfort extension comparison
- Table 6: Ablation study
- Table 7: Literature comparison
- Table 8: Scalability
- Table 9: Waterfall decomposition
