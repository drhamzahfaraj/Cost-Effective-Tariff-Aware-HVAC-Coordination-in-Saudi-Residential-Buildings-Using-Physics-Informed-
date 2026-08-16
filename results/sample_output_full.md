# Sample output — `python src/simulator.py --full`

This is a representative example of the output written to `results_generated.md` after a
full run (3 seeds × 1000 episodes, Jeddah). Your own run should match these closely (small
stochastic variation of a few tenths of a percent is expected).

```
FULL run | city=jeddah | seeds=[0, 1, 2] | episodes=1000
  running villa_5zone.yaml | July | strict ...
  running villa_5zone.yaml | July | extended ...
  running villa_5zone.yaml | January | strict ...
  running villa_5zone.yaml | April | strict ...
  running villa_5zone.yaml | October | strict ...
  running building_20zone_multifloor.yaml | July | strict ...
  running building_20zone_multifloor.yaml | July | extended ...
Done.
```

| Scenario | Month | Band | OnOff kWh | OnOff SAR | PI-PPO SAR | Cost% | Energy% | Viol% | MeanViol C | MaxViol C |
|---|---|---|---|---|---|---|---|---|---|---|
| villa_5zone | July | strict | 3912 | 704 | 674+/-1.8 | 4.3 | 4.3 | 84.0 | 0.165 | 1.4 |
| villa_5zone | July | extended | 3951 | 711 | 616+/-6.1 | 13.4 | 13.4 | 81.8 | 0.154 | 1.29 |
| villa_5zone | January | strict | 594 | 107 | 97+/-0.1 | 9.1 | 9.1 | 52.1 | 0.726 | 2.75 |
| villa_5zone | April | strict | 1863 | 335 | 309+/-0.1 | 7.9 | 7.9 | 49.0 | 0.044 | 0.73 |
| villa_5zone | October | strict | 2516 | 453 | 421+/-0.1 | 6.9 | 6.9 | 59.6 | 0.083 | 0.91 |
| building_20zone_multifloor | July | strict | 16285 | 4166 | 4124+/-30.5 | 1.0 | 0.9 | 100.0 | 0.175 | 2.67 |
| building_20zone_multifloor | July | extended | 16368 | 4190 | 3862+/-172.1 | 7.8 | 6.7 | 99.9 | 0.172 | 2.07 |

Notes:
- The `+/-` values are standard deviations across the 3 seeds.
- The 20-zone extended case has high variance (+/-172 SAR) and is reported as approximate.
- `Viol%` is the fraction of timesteps with any violation; the meaningful comfort metric is
  the magnitude (`MeanViol C`, `MaxViol C`), which is small in the cooling-relevant months.
