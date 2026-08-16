# Results

Reference outputs of the paper's `--full` run (3 seeds × 1000 episodes, Jeddah weather).
These are the values the manuscript's tables and figures are built from. All bills are computed
from measured kWh via the Saudi two-tier tariff, so energy and cost are consistent by construction.

## Files

| File | Contents |
|---|---|
| `converged_results.json` | Full machine-readable results: per-scenario On/Off and PI-PPO kWh, bills, standard deviations, peak kW, comfort-violation magnitude (mean/max °C), plus run configuration and key findings. **This is the authoritative reference.** |
| `all_results.csv`        | The same results flattened to a single CSV (one row per scenario) for easy spreadsheet import. |
| `seasonal_villa.csv`     | Villa seasonal breakdown (Jan/Apr/Jul/Oct) as CSV. |
| `results_table.md`       | Human-readable table of all scenarios. |
| `KEY_FINDINGS.md`        | The study's main honest findings in plain language. |

## Headline numbers

| Scenario | Band | Cost reduction |
|---|---|---|
| Villa 5-zone, July | strict [23–25 °C] | ~4.3% |
| Villa 5-zone, July | extended [22–26 °C] | ~13.4% |
| Building 20-zone, July | strict | ~1.0% (near physical optimum) |
| Building 20-zone, July | extended | ~7.8% (high seed variance) |
| Villa seasonal (strict) | Jan / Apr / Oct | 9.1% / 7.9% / 6.9% |

## How to regenerate

```
cd ../src
python simulator.py --full
```

This writes `results_generated.json` and `results_generated.md` next to the script. Compare
against `converged_results.json`. Small differences (a few tenths of a percent) are expected from
stochastic training; the qualitative findings are stable.

## Honest reporting notes

- **Peak demand is not reduced** under a strict band (On/Off and PI-PPO peak identically); savings
  come from energy and tariff-tier effects, not peak shaving.
- The **20-zone strict-band saving (~1%)** is near the dynamic-programming physical optimum for a
  thermally saturated building.
- The **20-zone extended case has high seed variance** (±172 SAR) and is reported as approximate.
- **Comfort** is reported by magnitude (mean/max °C), not just violation frequency; PI-PPO trades a
  small comfort margin (mean ~0.16 °C) for its savings versus the On/Off baseline (~0.03 °C).
- All figures are **seed means with no best-seed selection.**
