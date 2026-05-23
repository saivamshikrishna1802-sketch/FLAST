# Phase 3 Implementation Plan

This document turns the direction in `phase3_pln.md` into an implementation-ready plan for the current repository.

The main rule for Phase 3 is simple: build federated learning directly on top of the frozen Phase 1 data artifacts and the existing Phase 2 Attention-LSTM benchmark. We should not redesign the dataset logic, temporal splits, or model family again.

## 1. What is already present

### Frozen data and split artifacts

Phase 1 is already frozen and gives us the exact data contract Phase 3 must reuse:

- `outputs/phase1/data/hourly_subset.csv.gz`
  - Hourly household-level load series for the sampled households.
- `outputs/phase1/data/sampled_households.csv`
  - The frozen sampled cohort.
- `outputs/phase1/data/node_assignments.csv`
  - The simulated federated node membership map.
- `outputs/phase1/reports/config.json`
  - The authoritative split definition:
    - train: `2011-11-23` to `2012-10-01`
    - test_2012q4: `2012-10-01` to `2013-01-01`
    - test_2013: `2013-01-01` to `2014-01-01`
    - test_2014: `2014-01-01` to `2014-03-01`

### Frozen Phase 1 findings that motivate Phase 3

- Phase 1 is closed.
- There are `10` federated nodes total:
  - `7` ToU nodes
  - `3` Std nodes
- ToU drift is real at the group level:
  - KS statistic: `0.0816`
  - mean shift: `30.76%`
- The ACF shift is also real:
  - ToU mean absolute ACF change: `0.0745`
- Node-level heterogeneity is visible:
  - RMSE-change range across nodes: `28.21` percentage points
- Reference drifting node:
  - `node_tou_6` is the strongest Phase 1 drift node in the frozen outputs.

### Centralized benchmark already implemented in Phase 2

The centralized benchmark is already in place and must remain the direct comparison target for Phase 3:

- Model code:
  - `src/attention_model.py`
    - `PlainLSTM`
    - `AttentionLSTM`
- Phase 2 config and orchestration:
  - `src/phase2/config.py`
  - `src/phase2/pipeline.py`
  - `scripts/run_phase2.py`

Current frozen centralized result:

- Overall 2013 RMSE:
  - Plain LSTM: `0.3560`
  - Attention-LSTM: `0.3410`
- Reference drift node result:
  - `node_tou_6` Attention-LSTM RMSE in 2013: `0.1041`

That `0.1041` number is the concrete Phase 3 node target to beat.

## 2. What is reusable right now

We already have almost everything needed for Phase 3 except the federated orchestration itself.

### Reusable model components

From `src/attention_model.py`:

- `PlainLSTM`
- `AttentionLSTM`
- additive attention layer

Phase 3 should import `AttentionLSTM` directly from this file. Do not create a second attention model implementation.

### Reusable preprocessing logic

The exact sequence-building behavior already exists and should be reused without changing the math:

From `src/phase1/model.py`:

- `_build_window_series`
- `_make_sliding_windows`
- `_build_series_split_arrays`
- `_build_stacked_household_train_arrays`
- `assign_nodes` (already used only for frozen Phase 1, but useful as the node data contract reference)

From `src/phase2/pipeline.py`:

- `_build_household_series_lookup`
- `_build_node_series_lookup`
- `_evaluate_arrays`
- `_predict_arrays`

### Reusable reporting pattern

Both earlier phases already follow the same repo pattern:

- `config.py` defines the phase dataclass
- `pipeline.py` runs the full experiment
- `scripts/run_phaseX.py` is the CLI entrypoint
- `outputs/phaseX/{figures,tables,reports}` stores artifacts

Phase 3 should follow the same structure.

## 3. What is not present yet

These are the Phase 3 pieces that do not exist yet:

- per-node client dataset objects
- server/client federated round orchestration
- vanilla FedAvg baseline
- drift-aware aggregation
- selective retraining trigger
- per-round convergence logging
- node-level drift scores for aggregation weights

One important honesty note:

The current Phase 1 outputs contain tariff-group drift metrics, not per-node KS drift scores. So Phase 3 cannot "reuse node drift weights directly from disk" yet. It must compute node-level drift scores from the frozen Phase 1 data and frozen node assignments as a small preprocessing step.

That is still fully consistent with the frozen-data rule because the source remains:

- `outputs/phase1/data/hourly_subset.csv.gz`
- `outputs/phase1/data/node_assignments.csv`
- `outputs/phase1/reports/config.json`

## 4. How the current centralized training loop works

This is the existing training structure we should build on top of, not replace.

### Phase 2 loop today

1. `scripts/run_phase2.py` builds `Phase2Config` and calls `run_phase2`.
2. `run_phase2` loads frozen Phase 1 artifacts:
   - hourly data
   - sampled households
   - node assignments
   - Phase 1 node table
3. It builds:
   - `household_series_lookup`
   - `node_series_lookup`
4. It creates one global training set by stacking household windows across all households.
5. It trains a single centralized model with `_train_model`.
6. It evaluates that single model:
   - by household group
   - by node
   - on frozen test windows
7. It writes figures, tables, and acceptance reports.

### The key structural insight for Phase 3

Phase 3 should keep steps `1`, `2`, `3`, `6`, and `7`.

The only major change is step `4` and how step `5` is executed:

- Phase 2 step `4`: build one global training array
- Phase 3 step `4`: build one training bundle per node

and

- Phase 2 step `5`: train one centralized model once
- Phase 3 step `5`: run round-based client training plus server aggregation

That means Phase 3 is not a new pipeline from scratch. It is a different orchestration layer on top of the same sequence generation, same model family, same evaluation windows, and same node definitions.

## 5. Phase 3 design principle

Phase 3 should preserve the following invariants:

- same sampled households as frozen Phase 1
- same node assignments as frozen Phase 1
- same time windows as frozen Phase 1 / Phase 2
- same sequence length: `48`
- same model family: `AttentionLSTM`
- same node-level evaluation target: node-average signal
- no future information in normalization or validation

What changes in Phase 3 is only this:

- training becomes local per node
- aggregation happens at the server
- some nodes are treated differently based on drift and local harm

## 6. Proposed Phase 3 file structure

Recommended Phase 3 module layout:

```text
src/
  phase3/
    __init__.py
    config.py
    data.py
    federated.py
    trigger.py
    evaluate.py
    pipeline.py

scripts/
  run_phase3.py
```

### Responsibilities by file

`src/phase3/config.py`

- `Phase3Config` dataclass
- output paths
- FL hyperparameters
- validation-window settings
- aggregation mode settings
- trigger settings

`src/phase3/data.py`

- load frozen Phase 1 artifacts
- build household and node series lookups
- build per-node local train arrays
- build per-node evaluation arrays
- build per-node drift scores
- define client dataclasses

`src/phase3/federated.py`

- local client training step
- model broadcast / state cloning
- vanilla FedAvg aggregation
- drift-aware aggregation
- per-round bookkeeping

`src/phase3/trigger.py`

- compare current and previous local validation losses
- decide which nodes receive extra fine-tuning
- store trigger flags and deltas

`src/phase3/evaluate.py`

- per-node evaluation on 2012 Q4 / 2013 / 2014
- overall summaries
- baseline comparison tables
- figures and report tables

`src/phase3/pipeline.py`

- end-to-end experiment orchestration
- run vanilla FedAvg baseline
- run FLAST
- write outputs and acceptance report

`scripts/run_phase3.py`

- CLI entrypoint aligned with `run_phase1.py` and `run_phase2.py`

## 7. Data structure that Phase 3 should introduce

The most useful new abstraction is a single node-client bundle.

Suggested dataclass:

```python
@dataclass
class NodeClientData:
    node_id: str
    node_type: str
    household_ids: list[str]
    household_count: int
    train_features: np.ndarray
    train_targets: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray
    eval_arrays: dict[str, dict[str, object]]
    train_mean: float
    train_std: float
    sample_count: int
    drift_score: float
```

Why this matters:

- Phase 2 hides all samples inside one centralized array.
- Phase 3 needs a persistent per-node object because each node becomes a client.

## 8. How to build client datasets directly from existing code

### Local train data

Local train data should reuse the same logic as the final Phase 1 node baseline:

- train on stacked household-level sequences inside each node
- keep the target as next-hour household load during training

This comes directly from the behavior in `run_node_lstm_analysis` and `_build_stacked_household_train_arrays`.

### Local evaluation data

Local evaluation should remain exactly what Phase 1 and Phase 2 already use:

- evaluate on the node-average time series
- not on individual household series

This keeps the comparisons fair:

- Phase 1 node baseline
- Phase 2 centralized benchmark
- Phase 3 federated methods

all end up being judged on the same node-average forecasting task.

### Local validation data for round-level trigger logic

Phase 3 needs one additional internal split that earlier phases did not need:

- a small client validation window cut from the tail of the training period only

Recommended default:

- client training window: `2011-11-23` to `2012-09-01`
- client validation window: `2012-09-01` to `2012-10-01`

Reason:

- no leakage into `2012 Q4`, `2013`, or `2014`
- gives each node a stable pre-drift validation slice
- supports the selective retraining trigger after each communication round

This validation window is for Phase 3 round control only. It does not change the frozen external evaluation windows.

## 9. Node drift scores: what to reuse and what to compute

### Already available

Available now:

- tariff-group KS metrics from `outputs/phase1/tables/distribution_shift_metrics.csv`

Not available yet:

- per-node KS drift scores

### Required Phase 3 preprocessing step

Before training, Phase 3 should compute a node-level drift table from the frozen Phase 1 data:

- build each node's mean series using `hourly_subset.csv.gz` plus `node_assignments.csv`
- compare:
  - pre-drift window: `2011-11-23` to `2013-01-01`
  - drift window: `2013-01-01` to `2014-01-01`
- compute at least:
  - KS statistic
  - mean shift percent
  - std shift percent

Write this as:

- `outputs/phase3/tables/node_drift_scores.csv`

This file becomes the source of truth for drift-aware weighting.

## 10. Training loop structure for Phase 3

## 10.1 Common setup for both methods

Both vanilla FedAvg and FLAST should use the same shared setup:

1. Load frozen Phase 1 artifacts.
2. Build `NodeClientData` for all 10 nodes.
3. Initialize one global `AttentionLSTM`.
4. Run `R` communication rounds.
5. After each round:
   - store global round metrics
   - store node-level metrics
   - store local validation losses

### Recommended Phase 3 defaults

- communication rounds: `10`
- local epochs per round: `1` to `2`
- batch size: reuse `512` unless memory forces smaller
- learning rate: start with `1e-3`
- sequence length: keep `48`
- hidden size: keep `64`

## 10.2 Vanilla FedAvg baseline

Vanilla FedAvg is the first required federated baseline and should be implemented before FLAST.

Round structure:

1. Server sends global weights to all clients.
2. Each client trains locally on its own stacked household train arrays.
3. Each client returns:
   - updated model state
   - local sample count
   - local validation loss
4. Server aggregates with standard FedAvg:
   - weight by local sample count
5. Server evaluates the aggregated model by node.

Suggested aggregation rule:

```python
global_param = sum(client_samples_i / total_samples * client_param_i)
```

This is the correct "plain FL" benchmark that FLAST must beat.

## 10.3 FLAST round structure

FLAST should reuse the exact same client update code as vanilla FedAvg and only change two things:

- aggregation weighting
- post-aggregation selective retraining

### A. Drift-aware aggregation

Recommended weighting:

```python
raw_weight_i = sample_count_i / (drift_score_i + eps)
norm_weight_i = raw_weight_i / sum(raw_weight_j)
```

Why this form is better than pure inverse-KS only:

- preserves the FedAvg idea of sample-size weighting
- still downweights drifting nodes
- reduces to near-FedAvg behavior when drift scores are similar

### B. Selective retraining trigger

After aggregation, evaluate the new global model on each client's local validation slice.

Trigger condition:

```python
relative_increase_i = (val_loss_i_now - val_loss_i_prev) / val_loss_i_prev
trigger_i = relative_increase_i > threshold
```

Recommended default threshold:

- `5%`

Recommended first implementation behavior:

1. Server broadcasts the new aggregated model.
2. Triggered nodes perform `extra_finetune_epochs` on their own local train arrays.
3. The extra fine-tuned copy is used for that node's personalized evaluation artifact.
4. The next federated round still starts from the canonical shared global model.

This keeps the first implementation clean:

- FedAvg and FLAST are comparable
- server state remains unambiguous
- selective retraining is clearly measurable as a local adaptation mechanism

If we later want trigger-driven client states to persist into the next round, that can be a second pass. It should not be part of the first implementation.

## 11. Minimal implementation sequence

This is the order we should follow in code.

### Step 1 - Build data layer

Create `src/phase3/data.py` with:

- frozen artifact loading
- node client bundle construction
- node drift score computation

Deliverable:

- `NodeClientData` objects for all 10 nodes
- `node_drift_scores.csv`

### Step 2 - Build vanilla FedAvg baseline

Create `src/phase3/federated.py` with:

- local client update function
- FedAvg aggregation
- round history logging

Deliverable:

- a complete federated baseline that converges

### Step 3 - Add Phase 3 evaluation layer

Create `src/phase3/evaluate.py` with:

- per-node RMSE tables
- overall tables
- comparison against centralized Attention-LSTM

Deliverable:

- node-by-node 2013 evaluation from federated baseline

### Step 4 - Add drift-aware aggregation

Extend `src/phase3/federated.py` with:

- drift-aware weighted aggregation
- weight-history logging per round

Deliverable:

- visible difference between FedAvg weights and FLAST weights

### Step 5 - Add selective retraining trigger

Create `src/phase3/trigger.py` and integrate into `pipeline.py`.

Deliverable:

- trigger events per round
- personalized node evaluation for triggered nodes

### Step 6 - Reporting and acceptance outputs

Add:

- `scripts/run_phase3.py`
- `outputs/phase3/figures`
- `outputs/phase3/tables`
- `outputs/phase3/reports`

## 12. Proposed Phase 3 outputs

Recommended outputs:

### Tables

- `node_drift_scores.csv`
- `federated_round_history.csv`
- `federated_validation_history.csv`
- `aggregation_weights_by_round.csv`
- `table_4_federated_model_comparison.csv`
- `table_5_node_federated_performance.csv`
- `trigger_events.csv`

### Figures

- `figure_7_federated_convergence.png`
- `figure_8_drift_aware_weights.png`
- `figure_9_node_rmse_comparison_phase3.png`
- `figure_10_trigger_activity.png`

### Reports

- `reports/config.json`
- `reports/acceptance_report.json`
- `reports/summary.md`

## 13. Acceptance targets for implementation

Phase 3 implementation is complete when all of the following are true:

1. Vanilla FedAvg runs end to end across all 10 frozen nodes.
2. FLAST runs end to end across the same nodes and same splits.
3. Drift-aware aggregation visibly differs from vanilla FedAvg.
4. The selective trigger fires for at least one node in at least one round.
5. Results are reported against the centralized Attention-LSTM benchmark.
6. `node_tou_6` is explicitly tracked as the reference drift node.
7. No data leakage is introduced.

Performance acceptance can still follow the targets in `phase3_pln.md`, but the implementation itself should be considered structurally complete once the pipeline above exists and produces stable outputs.

## 14. Guardrails

To avoid breaking the paper story, Phase 3 should not do any of the following:

- resample households again
- change node assignments
- change the external train/test windows
- introduce a different model family
- evaluate on household-level targets instead of node-average targets
- use 2012 Q4 or 2013 as a tuning window for local trigger control

## 15. Bottom line

The repo is already in a very good place for Phase 3.

What exists today:

- frozen node definitions
- frozen hourly data
- frozen centralized benchmark
- reusable sequence-building code
- reusable Attention-LSTM model

What Phase 3 must add:

- client-wise dataset packaging
- federated round orchestration
- drift-aware aggregation
- selective retraining

So the right implementation strategy is not "build a new forecasting stack." It is:

build a federated wrapper around the exact training and evaluation logic that already proved out in Phase 1 and Phase 2.
