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
- ADF status: kept only as a diagnostic artifact, not an acceptance gate

## Node-level baseline failure

- Nodes above 10% RMSE degradation: 0/10
- Average node RMSE increase: -6.22%
- ToU node average RMSE increase: -6.07% (0/7 nodes above threshold)
- Std node average RMSE increase: -6.57% (0/3 nodes above threshold)
- Aggregate baseline context: RMSE 0.3491 in 2012 Q4 vs 0.3498 in 2013 (0.20%)

Top drifting nodes:
- node_tou_6 (ToU): 7.40% RMSE increase
- node_tou_4 (ToU): 3.76% RMSE increase
- node_tou_2 (ToU): 0.19% RMSE increase

## Acceptance status

- All acceptance checks passed: False

Generated artifacts:

- Figure 1: figures/figure_1_distribution_shift.png
- Figure 2: figures/figure_2_node_rmse_degradation.png
- Figure 3: figures/figure_3_acf_shift.png
- Table 1: tables/table_1_node_baseline_metrics.csv
- Node summary: tables/node_degradation_summary.csv
- Aggregate baseline: tables/aggregate_baseline_metrics.csv
- Diagnostic ADF figure: figures/diagnostic_adf_failure_rates.png
- Acceptance report: reports/acceptance_report.json