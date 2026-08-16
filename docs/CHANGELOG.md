# Changelog / revision history

This repository accompanies the revised manuscript. This file records the substantive changes to
the reported results, in the interest of transparency and reproducibility.

## Current version (revised)

- All quantitative results are produced by the released simulation code (`src/simulator.py`) and
  are reported as measured values across 3 independent training seeds.
- Results supersede figures reported in earlier manuscript versions. During this revision the full
  pipeline was re-implemented and re-run rigorously; the corrected results are lower in magnitude
  than earlier figures and are accompanied by characterisation of the method's limits
  (thermal saturation, no peak-demand reduction under a strict band, comfort–cost trade-off, and
  seed variance in the 20-zone extended case).
- The simulation is a lumped-parameter RC thermal model (not EnergyPlus); the manuscript describes
  it accurately as such, with detailed-simulator validation listed as future work.
- The full code and data are released so that every table and figure can be independently
  regenerated.

## Notes on interpretation

- Reported savings are modest. The paper's contribution is framed as a realistic-
  conditions characterisation of when tariff-aware coordination helps (and when it does not),
  rather than a maximal-savings claim.
- Bills are computed from measured kWh via the Saudi two-tier tariff, so energy and cost are
  internally consistent throughout.

## Reproducibility commitment

Anyone can run `python src/simulator.py --full` and obtain results consistent with
`results/converged_results.json` and the manuscript tables (up to expected stochastic variation).
