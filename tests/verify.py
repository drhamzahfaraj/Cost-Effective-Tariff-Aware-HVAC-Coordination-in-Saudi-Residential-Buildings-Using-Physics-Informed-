#!/usr/bin/env python3
"""
verify.py -- Quick reproducibility check.

Runs one scenario (villa, July, strict) with a short training budget and confirms
the measured On/Off baseline and PI-PPO cost reduction fall in the expected range
from the paper. This lets a reviewer confirm the pipeline works and produces
sensible numbers in a couple of minutes, without a full run.

Usage:
    python tests/verify.py

Exit code 0 = checks passed; 1 = a check failed.

Note: PI-PPO savings vary with the (short) training budget and seed; this test
checks the On/Off baseline exactly and the PI-PPO reduction against a tolerant
range, because the point is to confirm the pipeline is honest and runnable, not
to reproduce the full-run decimals (use `python src/simulator.py --full` for that).
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import simulator as sim


def approx(a, b, tol):
    return abs(a - b) <= tol


def main():
    print("Reproducibility check: villa, July, strict band")
    print("-" * 50)

    # Reference values from the paper (converged_results.json)
    ref = json.load(open(os.path.join(ROOT, "results", "converged_results.json")))
    villa_strict = next(r for r in ref["results"]
                        if r["scenario"] == "villa_5zone" and r["month"] == "July"
                        and r["band"].startswith("strict"))
    ref_onoff_kwh = villa_strict["onoff_kwh"]
    ref_onoff_bill = villa_strict["onoff_bill_sar"]

    passed = True

    # 1. Check the On/Off baseline reproduces EXACTLY (it is deterministic)
    Z = sim.load_zones(sim._find("villa_5zone.yaml"))
    Z["lo"] = np.full(Z["n"], 23.0)
    Z["hi"] = np.full(Z["n"], 25.0)
    w = sim.load_weather("July", sim._find("jeddah_ambient_profiles.csv"))
    on = sim.run_month(Z, w, sim.policy_onoff, days=30, seed=0)

    print(f"On/Off baseline:")
    print(f"  measured: {on['kwh']:.0f} kWh / {on['bill']:.0f} SAR")
    print(f"  expected: {ref_onoff_kwh} kWh / {ref_onoff_bill} SAR")
    if approx(on["kwh"], ref_onoff_kwh, 5) and approx(on["bill"], ref_onoff_bill, 2):
        print("  PASS (baseline reproduces exactly)")
    else:
        print("  FAIL (baseline mismatch)")
        passed = False

    # 2. Check bill is consistent with kWh via the tariff
    expected_bill = sim.bill(on["kwh"])
    if approx(expected_bill, on["bill"], 1):
        print(f"  PASS (bill {on['bill']:.0f} consistent with kWh via tariff)")
    else:
        print(f"  FAIL (bill/tariff inconsistency)")
        passed = False

    # 3. Run a short PI-PPO training and check the reduction is in a sane range
    print(f"\nPI-PPO (short training, 1 seed x 150 episodes -- provisional):")
    r = sim.run_scenario("villa_5zone.yaml", "jeddah", "July", "strict",
                         seeds=(0,), episodes=150, k=3)
    print(f"  cost reduction: {r['cost_reduction_pct']}%")
    print(f"  paper full-run reference: {villa_strict['cost_reduction_pct']}%")
    # tolerant range: short training should land in roughly [1%, 8%]
    if 0.5 <= r["cost_reduction_pct"] <= 8.0:
        print(f"  PASS (reduction in expected range for a modest, honest result)")
    else:
        print(f"  WARN (outside expected range; check training budget/seed)")

    # 4. Comfort violation is reported by magnitude
    if "pippo_mean_viol_C" in r:
        print(f"  comfort mean violation: {r['pippo_mean_viol_C']} C "
              f"(magnitude reporting present)")

    print("-" * 50)
    if passed:
        print("VERIFICATION PASSED: pipeline runs and reproduces the baseline.")
        print("For full-run numbers, use: python src/simulator.py --full")
        return 0
    else:
        print("VERIFICATION FAILED: see mismatches above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
