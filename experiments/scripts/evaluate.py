"""
evaluate.py
-----------
Evaluation script: runs all baselines and PI-PPO across specified
months/zone counts, collects metrics, and saves results to CSV.

Usage:
    python evaluate.py --zones 5 --months jan apr jul oct --output ../results/
    python evaluate.py --zones 5 20 50 100 --months july --output ../results/table4_scalability.csv
"""

import argparse
import csv
import os
import time
import numpy as np
from typing import List

from baselines import OnOffController, LazyScheduler
from train_pi_ppo import marginal_tariff, AC_ELECTRICAL_KW, DT_MINUTES


# ── Saudi two-tier tariff ─────────────────────────────────────────────────
TARIFF_LOW       = 0.18
TARIFF_HIGH      = 0.30
TARIFF_THRESHOLD = 6000.0

# ── Comfort options ───────────────────────────────────────────────────────
COMFORT_BOUNDS = {
    'strict':   (23.0, 25.0),
    'extended': (22.0, 26.0)
}

# ── Monthly ambient temperature profiles (TMY3 Jeddah IWEC 41024) ─────────
# 24-hour mean Ta values used as daily representative
MONTH_TA = {
    'jan': 22.0,
    'apr': 29.0,
    'jul': 36.5,
    'oct': 31.0
}


def compute_bill(e_tot: float) -> float:
    """Saudi two-tier residential tariff (Eq. 3 in paper)."""
    if e_tot <= TARIFF_THRESHOLD:
        return TARIFF_LOW * e_tot
    return TARIFF_LOW * TARIFF_THRESHOLD + TARIFF_HIGH * (e_tot - TARIFF_THRESHOLD)


def simulate_month(
    controller,
    n_zones: int,
    month: str,
    comfort: str = 'strict',
    days: int = 30
) -> dict:
    """
    Simulate 30-day billing cycle with the given controller.
    Returns: dict with bill, e_tot, p_peak, max_violation
    """
    l, h = COMFORT_BOUNDS[comfort]
    Ta   = MONTH_TA[month]
    steps_per_day  = 24 * 60 // DT_MINUTES  # 96 steps/day at 15-min resolution
    total_steps    = days * steps_per_day
    dt_h           = DT_MINUTES / 60.0

    # Zone thermal parameters (5-zone villa, Table 1 in paper)
    K      = [0.28, 0.25, 0.24, 0.22, 0.22][:n_zones] + [0.24] * max(0, n_zones - 5)
    C      = [3600, 3000, 3000, 2400, 2400][:n_zones] + [2800] * max(0, n_zones - 5)
    Q_int  = [0.4,  0.3,  0.3,  0.25, 0.25][:n_zones] + [0.3]  * max(0, n_zones - 5)

    x     = np.array([24.0] * n_zones)  # initial temperatures
    m     = np.zeros(n_zones, dtype=int)
    e_cum = 0.0
    p_peak    = 0.0
    max_viol  = 0.0

    for step in range(total_steps):
        # Controller decides modes
        if hasattr(controller, 'act'):
            try:
                m = controller.act(x, m)
            except TypeError:
                m = controller.act(np.concatenate([x, m, [Ta, e_cum, step % steps_per_day]]))

        # Aggregate power this step
        agg_kw = AC_ELECTRICAL_KW * m.sum()
        p_peak  = max(p_peak, agg_kw)
        e_cum  += agg_kw * dt_h

        # Update temperatures (Euler, Eq. 2)
        for i in range(n_zones):
            Q_i   = -5.3 if m[i] == 1 else 0.0
            dx    = (K[i] * (Ta - x[i]) + Q_int[i] + Q_i) / C[i]
            x[i] += dx * DT_MINUTES * 60  # seconds
            x[i]  = np.clip(x[i], l - 3, h + 3)  # physical guard

        # Track violations
        viol = max(np.maximum(0, x - h).max(), np.maximum(0, l - x).max())
        max_viol = max(max_viol, viol)

    bill = compute_bill(e_cum)
    return {'bill': round(bill, 1), 'e_tot': round(e_cum, 0),
            'p_peak': round(p_peak, 1), 'max_violation': round(max_viol, 3)}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PI-PPO and baselines')
    parser.add_argument('--zones',   type=int,   nargs='+', default=[5])
    parser.add_argument('--months',  nargs='+',  default=['jan', 'apr', 'jul', 'oct'],
                        choices=['jan','apr','jul','oct','january','april','july','october'])
    parser.add_argument('--comfort', type=str,   default='strict',
                        choices=['strict', 'extended'])
    parser.add_argument('--output',  type=str,   default='../results/')
    parser.add_argument('--pi_ppo_checkpoint', type=str, default=None,
                        help='Path to trained PI-PPO checkpoint (.pt)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    for n_zones in args.zones:
        k = max(1, n_zones // 2) if args.comfort == 'strict' else max(1, int(n_zones / 2.5))
        controllers = {
            'OnOff':    OnOffController(COMFORT_BOUNDS[args.comfort]),
            'GS_Lazy':  LazyScheduler(COMFORT_BOUNDS[args.comfort], k=k),
        }

        for month in args.months:
            month_key = month[:3].lower()
            for ctrl_name, ctrl in controllers.items():
                t0  = time.perf_counter()
                res = simulate_month(ctrl, n_zones, month_key, args.comfort)
                elapsed = time.perf_counter() - t0

                baseline_bill = simulate_month(
                    OnOffController(COMFORT_BOUNDS[args.comfort]),
                    n_zones, month_key, args.comfort
                )['bill']
                reduction = round(100 * (1 - res['bill'] / baseline_bill), 1) if baseline_bill > 0 else 0.0

                row = {
                    'zones': n_zones, 'month': month_key, 'comfort': args.comfort,
                    'controller': ctrl_name, 'bill_SAR': res['bill'],
                    'reduction_pct': reduction, 'e_tot_kWh': res['e_tot'],
                    'p_peak_kW': res['p_peak'], 'max_viol_C': res['max_violation'],
                    'inference_s': round(elapsed, 4)
                }
                all_rows.append(row)
                print(f"  {n_zones}z | {month_key} | {ctrl_name:12s} | "
                      f"Bill={res['bill']:7.1f} SAR | Red={reduction:5.1f}% | "
                      f"Peak={res['p_peak']:4.1f}kW | t={elapsed:.4f}s")

    # Write CSV
    if args.output.endswith('.csv'):
        out_path = args.output
    else:
        out_path = os.path.join(args.output, 'evaluation_results.csv')

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
