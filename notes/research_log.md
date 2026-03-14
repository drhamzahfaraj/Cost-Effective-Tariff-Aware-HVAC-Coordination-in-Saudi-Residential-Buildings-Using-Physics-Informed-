# Research Log — PI-PPO for Saudi HVAC Scheduling

---

## 2026-03-15 — Final manuscript sync (main-13)

- Pushed full `paper/main.tex` (44 KB, all 9 sections, 3 TikZ figures, 9 tables)
- Updated `paper/references.bib` (35 verified BibTeX entries, ref1–ref35)
- Added `experiments/data/thermal_params.py` — 5-zone and 20-zone building parameters
- Added `experiments/data/jeddah_tmy3.py` — TMY3 temperature profiles (4 months)
- Added `experiments/scripts/nn_surrogate.py` — NN surrogate pre-training (50k samples)
- Added `experiments/results/table1_villa5z.csv` through `table7_waterfall.csv`
- Updated `README.md` — corrected all numbers to match paper final values
- Added `requirements.txt` and `.gitignore`

### Final key numbers (verified against main-13.tex)

| Result | Value |
|---|---|
| 5-zone July PI-PPO bill reduction | 22.5% |
| 20-zone July PI-PPO bill reduction (strict) | 33.0% |
| 20-zone July PI-PPO bill reduction (extended) | 47.0% |
| Peak demand reduction range | 40–60% |
| d_i reduction from comfort extension | 36.9% |
| Physics reward contribution (ablation) | 6.0 pp |
| Tariff awareness contribution (ablation) | 4.5 pp |
| Coupling contribution (ablation) | 2.0 pp |
| Comfort extension contribution (waterfall) | 14.0 pp |
| Inference time at 100 zones | <0.04 s |

---

## 2026-03-14 — Manuscript revision to main-13

- Refined abstract numbers: peak 40–60%, July 22.5% (5z), 47.0% (20z extended)
- Added worked numerical example (Living room, July) in Section 6.2
- Completed ablation table with 7 variants
- Added waterfall decomposition table (Table 8)
- Added comfort range extension table (Table 5) — all months, both scales
- Added scalability results to 100 zones
- Finalized feasibility analysis: d_i(h=25)=0.48, d_i(h=26)=0.30, Δ=36.9%
- Added limitations subsection
- Finalized acknowledgments and data availability statement

---

## 2026-03-10 — Initial framework implementation

- Implemented PI-PPO training loop (PPOTrainer, PIPPOPolicy)
- Implemented physics-informed reward with 6 components
- Implemented lazy scheduling (GS) baseline
- Implemented MPC-1 and MPC-3 baselines
- Integrated EnergyPlus 23.2 via Sinergym for 5-zone villa
- Ran initial experiments: 5-zone July, strict comfort
- Confirmed zero comfort violations with physics reward enabled

---

## 2026-03-05 — Task model and feasibility analysis

- Derived minimum utilization formula Eq.(7) for cooling mode
- Computed d_i = 0.48 (strict) and 0.30 (extended) for Jeddah July
- Confirmed 36.9% reduction in d_i when extending comfort band
- Verified feasibility conditions: d_sum < k for both ranges
- Implemented inter-zone coupling (K_ij terms) in thermal dynamics
