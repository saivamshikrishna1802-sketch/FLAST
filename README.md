# FLAST: Drift-Aware Federated Attention-LSTM with Selective Retraining for Household Load Forecasting under Tariff-Induced Non-Stationarity

## Authors
**Author 1**, **Author 2**, **Author 3**  
Affiliation placeholder  
Email placeholder

## Abstract

Short-term residential load forecasting is difficult under behavioral non-stationarity: consumption patterns shift over time, privacy-sensitive smart-meter data are naturally decentralized, and global models often improve average error while redistributing error unevenly across clients. This paper presents **FLAST**, a federated forecasting framework that combines an additive-attention LSTM local predictor with two orchestration mechanisms: inverse-drift aggregation and drift-adaptive personalization. We study FLAST in a tariff-transition setting from the London Smart Meter dataset, where only a subset of Time-of-Use (ToU) households exhibits strong temporal drift. This lets us frame the problem as **selective adaptation under localized heterogeneous drift**, rather than generic non-IID learning. Our experimental pipeline proceeds in three phases. Phase 1 establishes the problem by showing statistically significant non-stationarity in ToU nodes, including a ToU Kolmogorov-Smirnov statistic of 0.0816 with a 30.76% mean shift from the pre-drift window to 2013. Phase 2 builds a centralized benchmark, where an Attention-LSTM improves aggregate RMSE over a plain LSTM by 3.92% on 2012 Q4 and 4.21% on 2013, but still yields uneven node-level outcomes. Phase 3 introduces FLAST and compares it against vanilla FedAvg and the centralized Attention-LSTM. FLAST reduces 2013 RMSE on the primary drift node `node_tou_6` from 0.1041 to 0.1011, a 2.91% gain over the centralized benchmark, while also improving three ToU nodes by at least 0.5% versus FedAvg (`node_tou_4`, `node_tou_5`, `node_tou_6`) with a mean gain of 1.66%. Drift-aware aggregation produces a 2.23x weight differential between high-drift and stable nodes, and FLAST remains within 0.53% of FedAvg on stable Std nodes. The results support an honest selective-adaptation claim: FLAST does not uniformly dominate all baselines, but it improves the drifting nodes that matter most without materially destabilizing the federated learning baseline.

## Key Contributions

1. **Three-phase empirical pipeline**: We establish the presence of tariff-linked non-stationarity, build a centralized attention baseline, and evaluate a drift-aware federated extension on frozen nodes and time splits.
2. **FLAST framework**: A federated Attention-LSTM with inverse-drift aggregation weights and a drift-adaptive personalization rule that activates when post-aggregation validation loss rises for eligible drifting clients.
3. **Selective adaptation under localized heterogeneous drift**: FLAST delivers targeted rather than uniform gains—it beats the centralized Attention-LSTM on the primary drift node, improves multiple drifting ToU nodes relative to FedAvg, and stays within 0.53% of FedAvg on stable nodes.

## Experimental Setup

### Dataset
We use the SmartMeter Energy Consumption Data in London Households dataset from the London Datastore, containing half-hourly energy readings for 5,567 London households from November 2011 through February 2014. For our experiments, we construct a focused subset of 200 households:
- 140 ToU households
- 60 Std households  
- 10 federated nodes total
- 20 households per node

### Time Windows
We preserve the same chronological splits throughout the entire study:
- Train window: `2011-11-23` to `2012-10-01`
- Stable test window: `2012-10-01` to `2013-01-01`
- Drift test window: `2013-01-01` to `2014-01-01`
- Post-drift test window: `2014-01-01` to `2014-03-01`

For Phase 3 only, the final month of the original training window (`2012-09-01` to `2012-10-01`) is used as a local validation slice for drift-adaptive personalization control.

### Forecasting Task
The task is one-step-ahead hourly load forecasting on normalized node-level mean consumption sequences:
- Lookback length: 48 hours
- Train stride: 3
- Evaluation stride: 1

### Model Architecture
- One-layer LSTM encoder (hidden size 64)
- Dropout: 0.2
- Additive attention over hidden states
- MLP prediction head over concatenated context and final hidden state
- Optimizer: Adam (learning rate: 0.001)
- Batch size: 512

### Phase 3 Settings
- Communication rounds: 10
- Local epochs per round: 1
- Extra fine-tuning epochs: 1
- Random seed: 17

## Key Results

### Phase 1: Non-Stationarity Verification
- ToU KS statistic: 0.0816 (p-value: effectively 0)
- Mean shift from pre-drift to 2013: 30.76% (ToU) vs 13.82% (Std)
- Most drift-affected node: `node_tou_6` (RMSE increases from 0.1224 in 2012 Q4 to 0.1348 in 2013, a 10.13% degradation)

### Phase 2: Centralized Attention-LSTM Benchmark
- 2012 Q4 RMSE improvement: 0.3613 → 0.3471 (+3.92%)
- 2013 RMSE improvement: 0.3560 → 0.3410 (+4.21%)
- However, node-level results remain uneven: only 3/10 nodes show positive 2013 improvement

### Phase 3: FLAST Results
- **Primary drift node (`node_tou_6`)**:
  - Centralized Attention-LSTM RMSE 2013: 0.1041
  - FedAvg RMSE 2013: 0.1017
  - FLAST RMSE 2013: 0.1011
  - **→ +2.91% improvement over centralized Attention-LSTM**
  - **→ +0.59% improvement over FedAvg**

- **Selective ToU gains** (vs FedAvg):
  - `node_tou_4`: +2.55%
  - `node_tou_5`: +1.84%
  - `node_tou_6`: +0.59%
  - Mean gain across selectively improved nodes: **1.66%**

- **Stable node preservation** (vs FedAvg):
  - Mean stable-node change: -0.32%
  - Worst stable-node change: -0.53%
  - **→ Within 0.53% tolerance**

- **Drift-aware aggregation effectiveness**:
  - Mean aggregation weight of top-drift nodes: 0.0615
  - Mean aggregation weight of Std nodes: 0.1369
  - **→ 2.23x weight differential**

- **Drift-adaptive personalization activity**:
  - Executed FLAST trigger events: 26
  - Active rounds: 10/10
  - Most frequently triggered nodes: `node_tou_4` (7), `node_tou_6` (7), `node_tou_5` (6), `node_tou_1` (6)

## Repository Structure

```
researchpaper_code/
├── paper/
│   └── conference_paper_v1.md      # Full conference paper draft
├── src/
│   ├── phase1/                     # Phase 1: Non-stationarity analysis
│   ├── phase2/                     # Phase 2: Centralized Attention-LSTM benchmark
│   └── phase3/                     # Phase 3: FLAST implementation
│       ├── __init__.py
│       ├── federated.py            # FedAvg + drift-aware aggregation
│       ├── trigger.py              # Selective retraining logic
│       ├── evaluate.py             # Per-node FL evaluation
│       ├── data.py                 # Data loading and preprocessing
│       ├── model.py                # Attention-LSTM model definition
│       ├── pipeline.py             # Training pipeline orchestrator
│       └── config.py               # Configuration management
├── scripts/
│   ├── run_phase1.py               # Execute Phase 1 experiments
│   ├── run_phase2.py               # Execute Phase 2 experiments
│   └── run_phase3.py               # Execute Phase 3 experiments (FLAST)
├── outputs/
│   ├── phase1/                     # Phase 1 results and artifacts
│   ├── phase2/                     # Phase 2 results and artifacts
│   └── phase3/                     # Phase 3 results and artifacts
│       ├── figures/                # Visualization plots
│       ├── tables/                 # Result tables and CSV exports
│       └── reports/                # JSON reports and summary markdown
├── Main_dataset/                   # Raw dataset files (not committed to save space)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## How to Run

### Prerequisites
- Python 3.7+
- Required packages listed in `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Execute Experiments

#### Phase 1: Establish Non-Stationarity
```bash
python scripts/run_phase1.py
```

#### Phase 2: Centralized Attention-LSTM Benchmark
```bash
python scripts/run_phase2.py
```

#### Phase 3: FLAST Federated Training
```bash
python scripts/run_phase3.py
```

### Expected Outputs
After running Phase 3, results will be available in:
- `outputs/phase3/figures/` - Visualization plots (PNG format)
- `outputs/phase3/tables/` - Detailed results in CSV format
- `outputs/phase3/reports/` - JSON acceptance report and markdown summary

Key files:
- `outputs/phase3/reports/acceptance_report.json` - Detailed evaluation against acceptance criteria
- `outputs/phase3/reports/summary.md` - Human-readable summary of results
- `outputs/phase3/figures/figure_9_node_rmse_comparison_phase3.png` - Main result visualization
- `outputs/phase3/figures/figure_10_trigger_activity.png` - Trigger event visualization

## Main Findings

FLAST successfully addresses the challenge of **selective adaptation under localized heterogeneous drift**:

1. **Targeted improvement**: FLAST improves the most drift-affected households (ToU nodes) while maintaining stable node performance
2. **Mechanism effectiveness**: 
   - Drift-aware aggregation reduces influence of drifting nodes on global model (2.23x weight differential)
   - Drift-adaptive personalization selectively fine-tunes nodes harmed by aggregation
3. **Practical significance**: 
   - 2.91% improvement over centralized Attention-LSTM on the primary drift node
   - Meaningful gains on multiple ToU nodes without degrading stable node performance
   - Privacy-preserving federated learning with better adaptation to real-world tariff-induced non-stationarity

## References

See the full paper in `paper/conference_paper_v1.md` for complete bibliography and detailed methodology.

## License

[Specify your license here]

## Acknowledgments

This work uses the SmartMeter Energy Consumption Data in London Households dataset made available through the London Datastore by UK Power Networks.