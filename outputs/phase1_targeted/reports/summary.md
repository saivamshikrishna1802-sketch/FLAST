# Phase 1 Summary

## Dataset and sample

- Sampled households: 200 total (60 Std, 140 ToU)
- Sampling strategy: drift_targeted
- Source dataset window: 2011-11-23 through 2014-02-28
- Post-drift note: the 2014 window is limited to Jan-Feb because the source dataset ends on 2014-02-28

## Drift evidence

- Overall ADF failure rate: 5.50% in 2011-2012, 2.50% in 2013
- ToU mean shift from pre-drift to 2013: 30.76%
- Std mean shift from pre-drift to 2013: 13.82%

## Baseline failure

- 2012 Q4 baseline RMSE / MAE: 0.3600 / 0.1803
- 2013 drift-year RMSE / MAE: 0.3731 / 0.1744
- RMSE increase vs 2012 Q4: 3.66%

## Acceptance status

- All acceptance checks passed: False

Generated artifacts:

- Figure 1: figures/figure_1_distribution_shift.png
- Figure 2: figures/figure_2_adf_failure_rates.png
- Figure 3: figures/figure_3_acf_shift.png
- Table 1: tables/table_1_baseline_metrics.csv
- Acceptance report: reports/acceptance_report.json
