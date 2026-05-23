# Phase 3 Summary

## Setup

- Phase 1 root frozen at: C:\Users\DELL\OneDrive\画像\Desktop\researchpaper_code\outputs\phase1
- Phase 2 root frozen at: C:\Users\DELL\OneDrive\画像\Desktop\researchpaper_code\outputs\phase2
- Communication rounds: 10
- Local epochs per round: 1
- Extra fine-tuning epochs: 1
- Validation window: 2012-09-01 00:00:00 to 2012-10-01 00:00:00

## Drift scoring

- node_tou_3 (ToU) KS=0.3092, mean shift=26.71%
- node_tou_4 (ToU) KS=0.2834, mean shift=20.72%
- node_tou_2 (ToU) KS=0.1895, mean shift=20.81%

## Federated results

- FedAvg overall RMSE: 0.0927 (2012 Q4), 0.0853 (2013)
- FLAST overall RMSE: 0.0922 (2012 Q4), 0.0851 (2013)
- Reference node node_tou_6: centralized 0.1041, FedAvg 0.1017, FLAST 0.1011
- Best FLAST vs FedAvg node gain: node_tou_4 at 2.55%
- Weakest FLAST vs FedAvg node gain: node_std_2 at -0.53%

## Acceptance status

- Required checks passed: False
- Phase 3 closed: False

Generated artifacts:

- Figure 7: figures/figure_7_federated_convergence.png
- Figure 8: figures/figure_8_drift_aware_weights.png
- Figure 9: figures/figure_9_node_rmse_comparison_phase3.png
- Figure 10: figures/figure_10_trigger_activity.png
- Table 4: tables/table_4_node_model_comparison.csv
- Table 5: tables/table_5_group_model_summary.csv
- Acceptance report: reports/acceptance_report.json