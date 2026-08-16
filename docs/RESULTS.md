# Results (with interpretation)

All values are means across 3 seeds from `python src/simulator.py --full`, Jeddah weather.
The reference machine-readable copy is `results/converged_results.json`.

## Cost reduction vs On/Off baseline

| Scenario | Band | On/Off (SAR) | PI-PPO (SAR) | Cost red. | Mean viol. (°C) | Max viol. (°C) |
|---|---|---|---|---|---|---|
| Villa 5-zone, Jul | strict  | 704  | 674 ± 1.8  | 4.3% | 0.165 | 1.4 |
| Villa 5-zone, Jul | extended| 711  | 616 ± 6.1  | 13.4%| 0.154 | 1.29 |
| Villa 5-zone, Jan | strict  | 107  | 97  ± 0.1  | 9.1% | 0.726 | 2.75 |
| Villa 5-zone, Apr | strict  | 335  | 309 ± 0.1  | 7.9% | 0.044 | 0.73 |
| Villa 5-zone, Oct | strict  | 453  | 421 ± 0.1  | 6.9% | 0.083 | 0.91 |
| Building 20-zone, Jul | strict  | 4166 | 4124 ± 30.5 | 1.0% | 0.175 | 2.67 |
| Building 20-zone, Jul | extended| 4190 | 3862 ± 172  | 7.8% | 0.172 | 2.07 |

## Interpretation

**1. Strict-band savings are modest and near the physical optimum.**
The 20-zone strict-band saving (~1.0%) is essentially the dynamic-programming optimum for a
thermally saturated building: when cooling demand is near the units' capacity, there is almost no
scheduling slack, so coordination cannot do much. This is an honest, useful boundary result.

**2. Comfort extension is the primary lever.**
Widening the band from [23–25] to [22–26] roughly triples the villa saving (4.3% → 13.4%) and
lifts the building from 1.0% to 7.8%, because the wider band creates legitimate slack.

**3. Tariff-crossing amplifies bill savings at scale.**
For the 20-zone extended case, the bill reduction (7.8%) exceeds the energy reduction (6.7%):
coordination removes consumption billed at the higher Tier-2 rate, so SAR saved > kWh saved.

**4. Peak demand is NOT reduced under a strict band.**
On/Off and PI-PPO peak identically (villa 9.0 kW, building 36.0 kW). Under a tight band in extreme
heat the comfort constraint forces compressors on at peak, so coordination redistributes off-peak
load rather than shaving the peak.

**5. Seasonal pattern.**
Savings are larger in cooler/shoulder months (Jan 9.1%, Apr 7.9%, Oct 6.9%) than in peak July
(4.3%): coordination helps most where cooling demand is substantial but non-saturating. This
suggests better performance in cooler Saudi locations — a hypothesis for future testing, not a
measured result here.

## Honest reporting notes

- **Comfort:** we report violation *magnitude*, not just frequency. Mean violations are small
  (~0.04–0.18 °C) in cooling months; January is looser (0.73 °C mean) because there is little
  cooling to optimise. PI-PPO's mean violation (~0.16 °C) exceeds the On/Off baseline's (~0.03 °C):
  it trades a small comfort margin for savings. This is stated in the paper.
- **Seed variance:** the 20-zone extended case has a large standard deviation (±172 SAR) and is
  reported as approximate. It is the scenario most in need of additional seeds.
- **No best-seed selection:** all figures are seed means.
