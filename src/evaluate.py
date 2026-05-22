"""
Evaluation and Billing Computation
Computes monthly bills using Saudi two-tier tariff formula.
Supports ablation study and sensitivity analysis modes.
"""

import numpy as np
import argparse
from reward import compute_bill, TIER1_RATE, TIER2_RATE, TIER_THRESHOLD


def compute_monthly_stats(
    power_trace: np.ndarray,     # Power draw at each step (kW) [n_steps]
    temp_trace: np.ndarray,      # Zone temperatures [n_steps, n_zones]
    comfort_low: float,
    comfort_high: float,
    dt_hours: float = 0.25,      # 15 min
) -> dict:
    """Compute monthly performance statistics."""
    # Total energy (kWh)
    e_total = np.sum(power_trace) * dt_hours
    
    # Bill (SAR)
    bill = compute_bill(e_total)
    
    # Peak demand (kW)
    p_peak = np.max(power_trace)
    
    # Comfort violations
    violations_high = np.maximum(0, temp_trace - comfort_high)
    violations_low = np.maximum(0, comfort_low - temp_trace)
    max_violation = float(np.max(violations_high + violations_low))
    
    return {
        'e_total_kwh': round(e_total),
        'bill_sar': round(bill),
        'p_peak_kw': round(p_peak, 1),
        'max_violation_c': round(max_violation, 2),
    }


def run_sensitivity_analysis(base_K: np.ndarray, base_C: np.ndarray):
    """Sensitivity analysis: ±20% perturbation of K_i and C_i."""
    results = {}
    for param_name, base_val, perturb in [
        ('K_i +20%', base_K, 1.20),
        ('K_i -20%', base_K, 0.80),
        ('C_i +20%', base_C, 1.20),
        ('C_i -20%', base_C, 0.80),
    ]:
        perturbed = base_val * perturb
        # Run evaluation with perturbed parameters
        # (placeholder — actual implementation uses EnergyPlus)
        results[param_name] = {'perturbation': perturb}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PI-PPO')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--config', type=str, help='Building config YAML')
    parser.add_argument('--month', type=str, default='july')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    
    print(f"Evaluating: month={args.month}, ablation={args.ablation}")
    
    # Example bill computation verification
    test_cases = [
        (3950, 711),   # 5z On-Off July
        (3060, 551),   # 5z PI-PPO July strict
        (2690, 484),   # 5z PI-PPO July extended
        (15800, 4020), # 20z On-Off July
        (11380, 2693), # 20z PI-PPO July strict
    ]
    print("\nBill verification (kWh → SAR):")
    for kwh, expected in test_cases:
        actual = round(compute_bill(kwh))
        status = "✓" if abs(actual - expected) <= 1 else "✗"
        print(f"  {kwh:>6} kWh → {actual:>5} SAR (expected {expected}) {status}")
