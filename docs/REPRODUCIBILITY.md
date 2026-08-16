# Reproducibility guide

Every quantitative claim in the paper maps to an executable command. This guide shows how.

## One command regenerates the core results

```bash
python src/simulator.py --full
```

This runs the full scenario matrix (villa: Jul strict, Jul extended, Jan, Apr, Oct;
building: Jul strict, Jul extended), 3 seeds x 1000 episodes each, and writes
`results_generated.json` and `results_generated.md`.

Compare `results_generated.json` against the committed `results/converged_results.json`.
Small differences (a few tenths of a percent) are expected from stochastic training; the
qualitative findings are stable across runs.

## Table-by-table mapping

| Paper item | How to reproduce |
|---|---|
| Villa seasonal table (Jan/Apr/Jul/Oct, strict) | `--full`; villa rows of the output |
| Building seasonal / July table | `--full`; `building_20zone` rows |
| Comfort-extension table (strict vs extended) | `--full`; compare `strict` vs `extended` rows |
| Confidence intervals | standard deviations in the JSON (3 seeds) |
| Cost-model sensitivity | recompute bills from measured kWh under alternative tariffs (see METHODS §2) |
| Scalability (5 vs 20 zone) | villa vs building rows |
| Comfort magnitudes (mean/max °C) | `mean_comfort_violation_C`, `max_comfort_violation_C` fields |

## Figure mapping

| Figure | Source |
|---|---|
| Ambient temperature profiles | `data/jeddah_ambient_profiles.csv` |
| Temperature trajectory | any single-zone rollout (illustrative) |
| Training convergence | the fine-tuning loop in `simulator.py` (villa, July strict) |
| Power profiles | hourly aggregate of a rollout; both controllers peak identically |
| Waterfall decomposition | cumulative effect of components (building, July) |

## Determinism

Seeds 0–2 are set for numpy and torch. Results are reproducible up to platform/library
floating-point differences and any nondeterminism in the torch backend.

## Environment

- Python >= 3.10
- numpy, pyyaml, torch (CPU is sufficient; GPU faster for `--full`)
- No other dependencies; the simulator is a single self-contained file.

## Sanity checks built into the code

- Bills are always derived from measured kWh via the two-tier tariff (energy/bill consistency).
- The evaluation reports comfort-violation magnitude, so any "savings" achieved by under-cooling
  are visible in the violation columns rather than hidden.
