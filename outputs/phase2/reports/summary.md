# Phase 2 Summary

## Setup

- Phase 1 root frozen at: C:\Users\DELL\OneDrive\画像\Desktop\researchpaper_code\outputs\phase1
- Sequence length: 48 hours
- Hidden size: 64
- Dropout: 0.2
- Epochs: 20

## Centralized benchmark

- Plain LSTM RMSE: 0.3613 (2012 Q4), 0.3560 (2013)
- Attention-LSTM RMSE: 0.3471 (2012 Q4), 0.3410 (2013)
- RMSE improvement vs Plain LSTM: 3.92% (2012 Q4), 4.21% (2013)

## Node-level comparison

- Best node-level 2013 improvement: node_tou_3 (ToU) at 5.71%
- Weakest node-level 2013 improvement: node_tou_0 (ToU) at -5.56%

## Attention behavior

- node_std_2 peaks at lookback hour 1 with average attention weight 0.2289
- node_tou_6 peaks at lookback hour 1 with average attention weight 0.2339

## Acceptance status

- Required checks passed: True
- Optional checks passed: False
- Phase 2 closed: True

Generated artifacts:

- Figure 4: figures/figure_4_training_curves.png
- Figure 5: figures/figure_5_attention_visualization.png
- Figure 6: figures/figure_6_node_rmse_comparison.png
- Table 2: tables/table_2_centralized_model_comparison.csv
- Table 3: tables/table_3_node_performance.csv
- Acceptance report: reports/acceptance_report.json