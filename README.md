# PI-PPO: Tariff-Aware HVAC Coordination for Saudi Residential Buildings

Reproducible code and data for the paper:

> **Tariff-Aware HVAC Coordination in Saudi Residential Buildings Using Physics-Informed Deep Reinforcement Learning**
> Hamzah Faraj, Department of Science and Technology, Ranyah College, Taif University.

This repository contains everything needed to independently regenerate **every quantitative
result and figure** in the paper. Nothing is hardcoded: all reported numbers are measured by
executing the simulation and the reinforcement-learning training in `src/`.

---

## 1. What this is (and is not)

- It **is** a lumped-parameter (RC) thermal simulation of multi-zone residential cooling under
  the Saudi two-tier electricity tariff, plus a Physics-Informed Proximal Policy Optimization
  (PI-PPO) scheduler and baselines (On/Off thermostat, greedy/lazy scheduling).
- It **is not** EnergyPlus or a detailed building-physics model. The RC model is a reduced-order
  abstraction chosen for analytical tractability. The paper states this plainly and lists
  higher-fidelity validation as future work.

The headline results are deliberately modest and are reported honestly (see §5).

---

## 2. Quick start

```bash
# 1. install (Python >= 3.10)
pip install numpy pyyaml torch

# 2. quick smoke test (2 seeds x 150 episodes, ~10 min CPU)
python src/simulator.py

# 3. fast run (2 seeds x 400 episodes, ~15-25 min CPU)
python src/simulator.py --fast

# 4. full run used for the paper (3 seeds x 1000 episodes)
python src/simulator.py --full
```

Outputs are written next to the script:
- `results_generated.json` — machine-readable results (means, standard deviations, comfort magnitudes)
- `results_generated.md` — a human-readable table

The committed `results/converged_results.json` is the reference output of the `--full` run that
the paper's tables are built from.

Optional flags:
- `--city jeddah|riyadh` — weather profile (paper uses Jeddah)
- `--comfort <float>` — override the comfort-penalty weight (default 45)

---

## 3. Repository layout

```
.
├── README.md                     # this file
├── src/
│   ├── README.md                 # explains single-file vs modular versions
│   ├── simulator.py              # self-contained: RC model, tariff, baselines,
│   │                             #   DP expert, behaviour cloning, PPO, evaluation
│   ├── rc_sim.py                 # modular: RC environment, tariff, baselines
│   ├── train_real.py             # modular: PPO training loop
│   ├── best_case.py              # modular: DP expert + behaviour-cloning warm-start
│   ├── agent.py, reward.py, scheduler.py, surrogate.py, utils.py
├── configs/
│   ├── villa_5zone.yaml          # 5-zone villa (per-zone On/Off split units)
│   └── building_20zone_multifloor.yaml   # 20-zone multi-floor building
├── data/
│   ├── README.md                         # documents format, provenance, per-city ranges
│   ├── jeddah_ambient_profiles.csv       # Jeddah diurnal profiles (paper scope)
│   ├── riyadh_ambient_profiles.csv       # optional: hotter, larger diurnal swing
│   └── taif_ambient_profiles.csv         # optional: cooler; not used in the paper
├── results/
│   └── converged_results.json    # reference output the paper's tables use
├── paper/
│   ├── main_honest.tex           # manuscript source (LaTeX)
│   ├── main_honest.pdf           # compiled manuscript
│   └── references.bib            # bibliography
└── docs/
    ├── METHODS.md                # detailed method description
    ├── RESULTS.md                # the numbers, with interpretation
    ├── REPRODUCIBILITY.md        # how each table/figure maps to a command
    └── CHANGELOG.md              # honest record of the correction history
```

---

## 4. How the method works (short version)

Each thermal zone is served by its **own** On/Off split air-conditioning unit, as is standard in
Saudi housing. The controller must decide, at each 15-minute step, which units run — respecting a
comfort band and a concurrency limit — to minimise the monthly bill under the two-tier tariff.

PI-PPO combines:
1. A **physics-informed reward** that embeds the zone heat-balance so the agent gets a dense,
   physically meaningful learning signal.
2. A **behaviour-cloning warm-start** from a per-zone dynamic-programming expert (PPO from scratch
   converges poorly in this binary, high-dimensional action space).
3. A **feasibility / concurrency analysis** that bounds how many units may run at once.

Baselines: an On/Off hysteresis thermostat (competent, not naive) and a greedy/lazy scheduler.

Full detail: `docs/METHODS.md`.

---

## 5. Headline results (honest summary)

| Scenario | Band | Cost reduction vs On/Off |
|---|---|---|
| 5-zone villa, July | strict [23–25 °C] | ~4.3% |
| 5-zone villa, July | extended [22–26 °C] | ~13.4% |
| 20-zone building, July | strict | ~1.0% (near physical optimum) |
| 20-zone building, July | extended | ~7.8% (high seed variance) |
| 5-zone villa, seasonal (strict) | Jan/Apr/Oct | 9.1% / 7.9% / 6.9% |

**Honest caveats, stated up front:**
- **Peak demand is not reduced under a strict band.** Both On/Off and PI-PPO peak identically,
  because the comfort constraint forces compressors on at peak. Savings come from energy and
  tariff-tier effects, not peak shaving.
- The **strict-band building saving (~1%) is near the dynamic-programming physical optimum** — in
  a thermally saturated building there is little scheduling slack, so coordination helps little.
- The **20-zone extended case has high seed variance** (±172 SAR) and is reported as approximate.
- PI-PPO trades a small comfort margin for savings (mean violation ~0.16 °C vs the On/Off
  baseline's ~0.03 °C); most excursions are fractional-degree, with occasional transient peaks of
  1–3 °C under load.

See `docs/RESULTS.md` for the full breakdown and interpretation.

---

## 6. Verify, reproduce, and regenerate figures

Quick check the pipeline works (~2 min):
```
python tests/verify.py        # or: make verify
```

Full run producing the paper's numbers:
```
python src/simulator.py --full   # or: make reproduce
```

Regenerate figures from the measured results:
```
python scripts/make_figures.py   # or: make figures  -> writes figures/
```

See `results/sample_output_full.md` for the expected output, and
`DATA_AVAILABILITY.md` for the data/code availability statement.

## 7. Reproducing specific paper items

Every table and figure maps to a command; see `docs/REPRODUCIBILITY.md`. In brief:
run `python src/simulator.py --full`, then compare `results_generated.json` to
`results/converged_results.json`.

---

## 8. Citation

If you use this code or data, please cite the paper (see `paper/`). A `CITATION.cff` can be added
on request.

## 9. License

Code released for reproducibility. See repository license (add your preferred license, e.g. MIT
for code and CC-BY for data/paper, before public release).
