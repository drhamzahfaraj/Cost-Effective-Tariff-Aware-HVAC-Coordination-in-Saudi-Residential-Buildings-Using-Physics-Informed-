# Data and Code Availability Statement

All code and data required to reproduce every result, table, and figure in the accompanying
manuscript are contained in this repository.

- **Simulation code:** `src/` (both a self-contained `simulator.py` and a modular implementation).
- **Building configurations:** `configs/`.
- **Weather data:** `data/` (Jeddah profiles used for all reported results; Riyadh and Taif
  included as optional extras not used in the paper).
- **Reference results:** `results/converged_results.json` and derived CSV/Markdown files.
- **Verification:** `python tests/verify.py` confirms the pipeline reproduces the baseline.
- **Figure regeneration:** `python scripts/make_figures.py`.

To reproduce the paper's headline numbers:

```
pip install -r requirements.txt
python src/simulator.py --full
```

and compare the generated `results_generated.json` with `results/converged_results.json`.

No proprietary data or licensed software is required. The simulation is a self-contained
lumped-parameter RC thermal model implemented in NumPy/PyTorch; it does not depend on EnergyPlus,
Sinergym, or any external building-physics engine.

The weather profiles are representative diurnal temperature series for reproducible relative
comparison of control strategies; users wishing to reproduce results for a specific location with
calibrated meteorological data may substitute a CSV in the documented format (see `data/README.md`).

**Suggested archival:** for a citable DOI, this repository can be deposited on Zenodo or a
comparable archive; a `CITATION.cff` is provided.
