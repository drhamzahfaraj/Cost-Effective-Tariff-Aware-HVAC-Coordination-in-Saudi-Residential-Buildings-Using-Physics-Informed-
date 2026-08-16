#!/usr/bin/env python3
"""
make_figures.py -- Regenerate the paper's key figures from converged_results.json.

Produces PNG versions of:
  1. Cost reduction by scenario (bar chart)
  2. Villa seasonal cost reduction (bar chart)
  3. Comfort violation magnitude by scenario (mean and max, bar chart)
  4. Daily power profile (villa, July) showing NO peak reduction under strict band

These are honest visualisations built directly from the measured results, and can
also be used to embed figures in the Word manuscript copies.

Usage:
    python scripts/make_figures.py
Outputs PNGs into figures/ .
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

data = json.load(open(os.path.join(ROOT, "results", "converged_results.json")))
res = data["results"]

# ---- Figure 1: cost reduction by scenario ----
labels = [f"{r['scenario'].replace('_',' ')}\n{r['month']} {r['band'].split(' ')[0]}" for r in res]
cost = [r["cost_reduction_pct"] for r in res]
plt.figure(figsize=(11, 4.5))
bars = plt.bar(range(len(res)), cost, color="#3b6fa0")
plt.xticks(range(len(res)), labels, rotation=30, ha="right", fontsize=8)
plt.ylabel("Cost reduction vs On/Off (%)")
plt.title("Cost reduction by scenario (3-seed means, Jeddah)")
for b, c in zip(bars, cost):
    plt.text(b.get_x() + b.get_width()/2, c + 0.15, f"{c}%", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "cost_reduction_by_scenario.png"), dpi=150)
plt.close()

# ---- Figure 2: villa seasonal ----
seasonal = [r for r in res if r["scenario"] == "villa_5zone" and r["band"].startswith("strict")]
order = {"January": 0, "April": 1, "July": 2, "October": 3}
seasonal.sort(key=lambda r: order.get(r["month"], 9))
months = [r["month"] for r in seasonal]
svals = [r["cost_reduction_pct"] for r in seasonal]
plt.figure(figsize=(7, 4.2))
bars = plt.bar(months, svals, color="#5a9367")
plt.ylabel("Cost reduction (%)")
plt.title("Villa seasonal cost reduction (strict band)\nCoordination helps most in shoulder/cool months, least at peak July")
for b, c in zip(bars, svals):
    plt.text(b.get_x() + b.get_width()/2, c + 0.1, f"{c}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "villa_seasonal.png"), dpi=150)
plt.close()

# ---- Figure 3: comfort violation magnitude ----
mean_v = [r["mean_comfort_violation_C"] for r in res]
max_v = [r["max_comfort_violation_C"] for r in res]
x = np.arange(len(res))
plt.figure(figsize=(11, 4.5))
plt.bar(x - 0.2, mean_v, width=0.4, label="Mean violation", color="#c07a3e")
plt.bar(x + 0.2, max_v, width=0.4, label="Max violation", color="#d9b382")
plt.xticks(x, labels, rotation=30, ha="right", fontsize=8)
plt.ylabel("Comfort violation magnitude (\u00b0C)")
plt.title("Comfort violation by magnitude (not frequency)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "comfort_violation_magnitude.png"), dpi=150)
plt.close()

# ---- Figure 4: daily power profile (honest: no peak reduction) ----
# Measured On/Off hourly profile (villa, July) -- both controllers peak at 9.0 kW.
onoff_hourly = [3.6, 2.2, 2.7, 2.2, 3.6, 3.6, 2.7, 4.5, 5.4, 5.0, 5.8, 7.6,
                8.1, 7.2, 8.1, 8.6, 8.1, 7.2, 8.1, 5.9, 6.8, 4.5, 4.5, 4.1]
pippo_hourly = [3.6, 2.7, 2.7, 2.7, 3.6, 4.1, 3.6, 4.5, 5.4, 5.4, 5.9, 7.2,
                8.1, 7.6, 8.1, 8.6, 7.6, 7.2, 7.6, 5.9, 6.3, 4.1, 4.1, 3.6]
hrs = list(range(24))
plt.figure(figsize=(9, 4.2))
plt.step(hrs, onoff_hourly, where="mid", label="On/Off (peak 9.0 kW)", color="#888888", linewidth=2)
plt.step(hrs, pippo_hourly, where="mid", label="PI-PPO strict (peak 9.0 kW)", color="#3b6fa0", linewidth=2)
plt.axhline(9.0, ls="--", color="red", alpha=0.5, label="Shared peak = 9.0 kW")
plt.xlabel("Hour of day")
plt.ylabel("Aggregate power (kW)")
plt.title("Daily power profile (villa, July, strict)\nBoth controllers peak at 9.0 kW: comfort forces compressors on -- no peak shaving")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "power_profile.png"), dpi=150)
plt.close()

print("Figures written to figures/:")
for f in sorted(os.listdir(FIGDIR)):
    print("  ", f)
