# Results

All results in these CSV files are from EnergyPlus 23.2 simulations.
The On-Off baseline uses EnergyPlus's native thermostatic control.
All other methods (GS, DQN, PPO, PI-PPO) replace the control logic
while EnergyPlus continues to simulate building physics.

Billing computed via Saudi two-tier tariff:
  Bill = 0.18 × min(kWh, 6000) + 0.30 × max(0, kWh - 6000)

## Files
- seasonal_5zone.csv: Table 2 (5-zone, 4 months)
- seasonal_20zone.csv: Table 3 (20-zone, 4 months)
- july_full_comparison.csv: Table 4 (5-zone, July, all 8 baselines)
- comfort_extension.csv: Table 5 (strict vs extended, all months × both scales)
- ablation.csv: Table 6 (component ablation, 20-zone July)
- scalability.csv: Table 8 (5/20/50/100 zones)
- waterfall.csv: Table 9 (cumulative decomposition)
- sensitivity.csv: Sensitivity analysis (±20% K_i, C_i)
- model_validation.csv: Table 10 (zone-level, July)
- model_validation_seasonal.csv: Table 11 (4-month RMSE)
- energy_validation.csv: Table 12 (monthly kWh: model vs EnergyPlus)
