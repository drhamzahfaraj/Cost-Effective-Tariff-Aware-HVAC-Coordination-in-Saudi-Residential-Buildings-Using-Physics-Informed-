# Changelog

## Revision (results verified against released code)

In this revision, every reported number was verified to trace to an executed run in this
repository. The following changes were made:

- **GS (green-scheduling) baseline** set to its measured value at the documented
  concurrency limit k=n (villa July: 0.6% cost reduction; ~0% at 20-zone scale). This
  supersedes earlier GS figures, which did not match the implemented controller's output.
- **MPC-1, MPC-3, and DQN baselines removed.** No implementation of these controllers
  exists in the codebase; their previously reported values were not reproducible and have
  been removed rather than asserted.
- **20-zone results restricted to July.** Under the strict band the 20-zone building is
  thermally saturated; measured runs show coordination does not improve on uncoordinated
  On/Off control outside the peak-cooling month, so non-July 20-zone rows were removed and
  the boundary is stated in the findings (see Section: "Where the strategy does and does not apply").
- See `results/provenance_audit.json` for the measured values behind these changes.

These changes reduce the headline scope but ensure the released code reproduces every
reported figure.

/ revision history

This repository accompanies the revised manuscript. This file records the substantive changes to
the reported results, in the interest of transparency and reproducibility.

## Current version (revised)

- All quantitative results are produced by the released simulation code (`src/simulator.py`) and
  are reported as measured values across 3 independent training seeds.
- Results supersede figures reported in earlier manuscript versions. During this revision the full
  pipeline was re-implemented and re-run rigorously; the revised results are lower in magnitude
  than earlier figures and are accompanied by a full characterisation of the method's limits
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
