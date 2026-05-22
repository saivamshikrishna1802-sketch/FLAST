# Phase 1 Summary

## Dataset and sample

- Sampled households: 8 total (4 Std, 4 ToU)
- Sampling strategy: drift_targeted
- Federated nodes: 4 total (2 ToU nodes, 2 Std nodes, 2 households each)
- Source dataset window: 2011-11-23 through 2014-02-28
- Post-drift note: the 2014 window is limited to Jan-Feb because the source dataset ends on 2014-02-28

## Drift evidence

- ToU KS statistic / p-value: 0.1357 / 2.909e-296
- ToU mean shift from pre-drift to 2013: 208.98%
- Std mean shift from pre-drift to 2013: 3.18%
- ADF status: kept only as a diagnostic artifact, not an acceptance gate

## Node-level baseline failure

- Nodes above 15% RMSE degradation: 3/4
- Average node RMSE increase: 188.92%
- ToU node average RMSE increase: 350.58% (2/2 nodes above threshold)
- Std node average RMSE increase: 27.26% (1/2 nodes above threshold)
- Aggregate baseline context: RMSE 0.1593 in 2012 Q4 vs 0.7944 in 2013 (398.75%)

Top drifting nodes:
- node_tou_0 (ToU): 551.97% RMSE increase
- node_tou_1 (ToU): 149.19% RMSE increase
- node_std_1 (Std): 58.06% RMSE increase

## Acceptance status

- All acceptance checks passed: True

Generated artifacts:

- Figure 1: figures/figure_1_distribution_shift.png
- Figure 2: figures/figure_2_node_rmse_degradation.png
- Figure 3: figures/figure_3_acf_shift.png
- Table 1: tables/table_1_node_baseline_metrics.csv
- Node summary: tables/node_degradation_summary.csv
- Aggregate baseline: tables/aggregate_baseline_metrics.csv
- Diagnostic ADF figure: figures/diagnostic_adf_failure_rates.png
- Acceptance report: reports/acceptance_report.json