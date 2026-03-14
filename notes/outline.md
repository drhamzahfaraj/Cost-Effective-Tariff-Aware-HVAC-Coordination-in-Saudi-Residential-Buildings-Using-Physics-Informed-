# Paper Outline — PI-PPO for Saudi HVAC Peak Demand Reduction

**Target journal:** Applied Energy  
**Author:** Hamzah Faraj, Taif University  
**Status:** Under Review

---

## Section Map

| # | Section | Key Contribution | Status |
|---|---|---|---|
| 1 | Introduction | Problem framing: 2-tier tariff + On/Off units | ✅ Done |
| 1.1 | Why Wider Comfort Reduces Peak | 36.9% utilization reduction theorem | ✅ Done |
| 2 | Related Work | Saudi energy, DRL-HVAC, PIML, scheduling | ✅ Done |
| 3 | System Model | Heat balance Eq.1, tariff Eq.3, inter-zone coupling | ✅ Done |
| 4 | Task Model & Feasibility | Min utilization Eq.7, infeasibility condition | ✅ Done |
| 5 | Scheduling Controller | Lazy scheduling baseline formulation | ✅ Done |
| 6 | PI-PPO Methodology | Reward Eq.9, network arch, top-k projection | ✅ Done |
| 7 | Experimental Setup | EnergyPlus 23.2, Sinergym, Jeddah TMY3, 5-zone villa | ✅ Done |
| 8 | Results & Discussion | Tables 2–5, ablation, scalability | ✅ Done |
| 9 | Conclusion | Summary + 5 future directions | ✅ Done |

---

## Core Arguments Chain

1. **Problem:** On/Off split ACs in Saudi villas synchronize → peak spike → 2-tier tariff breach
2. **Gap:** MPC doesn't scale; DRL ignores physics + tiered tariff; PINN targets demand-charge tariffs
3. **Solution:** PI-PPO = stationary policy + physics reward + inter-zone coupling + tariff tracking
4. **Key insight:** Comfort extension [23-25] → [22-26]°C reduces min utilization by 36.9%, enabling k=2 vs k=3
5. **Validation:** EnergyPlus simulation, 4 seasons, 5→100 zones, full ablation

---

## Novelty Claims (vs. Related Work)

- First to target **Saudi two-tier (kWh) tariff** (not demand-charge $/kW)
- First **stationary infinite-horizon** policy for multi-zone On/Off coordination
- Formal **feasibility analysis** with minimum utilization theorem
- **Inter-zone coupling** exploited for thermal buffering (2.4 pp from coupling alone)
- **Near-zero comfort violations** (≤0.02°C) — absent from standard DRL

---

## Ablation Attribution (20-zone, July)

| Component | Contribution |
|---|---|
| Scheduling coordination (GS) | 20.9 pp |
| PPO learning | 4.0 pp |
| Tariff awareness | 4.2 pp |
| Physics reward | 1.9 pp |
| Inter-zone coupling | 2.4 pp |
| Comfort extension | 15.8 pp |
| **Total PI-PPO** | **49.2 pp** |

---

## Open Items / Future Work

- [ ] Sim-to-real deployment in Saudi villas (online C_i, K_i, K_ij estimation)
- [ ] Time-of-use pricing under Vision 2030 smart-grid
- [ ] Rooftop PV + battery integration (0.07 SAR/kWh feed-in tariff)
- [ ] Occupant field study for 22–26°C acceptance in Saudi households
- [ ] Extension to inverter-driven variable-speed units (continuous action space)
