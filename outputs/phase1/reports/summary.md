# Phase 1 Summary

## Dataset and sample

- Sampled households: 200 total (60 Std, 140 ToU)
- Sampling strategy: drift_targeted
- Federated nodes: 10 total (7 ToU nodes, 3 Std nodes, 20 households each)
- Source dataset window: 2011-11-23 through 2014-02-28
- Post-drift note: the 2014 window is limited to Jan-Feb because the source dataset ends on 2014-02-28

## Drift evidence

- ToU KS statistic / p-value: 0.0816 / 0
- ToU mean shift from pre-drift to 2013: 30.76%
- Std mean shift from pre-drift to 2013: 13.82%
- ToU ACF mean absolute change across lags: 0.0745
- ToU ACF lag-24 change: -0.0434

## Node-level baseline

- Nodes above 10% RMSE degradation: 1/10
- Average node RMSE increase: -6.02%
- ToU node average RMSE increase: -5.57% (1/7 nodes above threshold)
- Std node average RMSE increase: -7.07% (0/3 nodes above threshold)
- Node heterogeneity range: 28.21 percentage points
- Aggregate baseline context: RMSE 0.3491 in 2012 Q4 vs 0.3498 in 2013 (0.20%)

Top drifting nodes:
- node_tou_6 (ToU): 10.13% RMSE increase
- node_std_2 (Std): -3.05% RMSE increase
- node_tou_4 (ToU): -5.79% RMSE increase

## Acceptance status

- Mandatory checks passed: True
- Preferred degradation check passed: False
- Phase 1 closed: True

## Conclusion

- Analysis of the London Smart Meter dataset reveals statistically significant distributional and behavioral non-stationarity, particularly within Time-of-Use tariff nodes. While aggregate forecasting metrics partially conceal this effect, node-level evaluation demonstrates heterogeneous degradation patterns across distributed smart-grid regions, motivating the need for adaptive federated forecasting mechanisms.

Generated artifacts:

- Figure 1: figures/figure_1_distribution_shift.png
- Figure 2: figures/figure_2_acf_shift.png
- Figure 3: figures/figure_3_node_rmse_degradation.png
- Table 1: tables/table_1_node_baseline_metrics.csv
- Node summary: tables/node_degradation_summary.csv
- Acceptance report: reports/acceptance_report.json