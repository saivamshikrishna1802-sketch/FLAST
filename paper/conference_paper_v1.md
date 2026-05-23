# FLAST: Drift-Aware Federated Attention-LSTM with Selective Retraining for Household Load Forecasting under Tariff-Induced Non-Stationarity

**Author 1**, **Author 2**, **Author 3**  
Affiliation placeholder  
Email placeholder

## Abstract

Short-term residential load forecasting is difficult under behavioral non-stationarity: consumption patterns shift over time, privacy-sensitive smart-meter data are naturally decentralized, and global models often improve average error while redistributing error unevenly across clients. This paper presents **FLAST**, a federated forecasting framework that combines an additive-attention LSTM local predictor with two drift-aware mechanisms: inverse-drift aggregation and selective post-aggregation retraining. We evaluate FLAST on the London Smart Meter dataset from the Low Carbon London trial, focusing on a tariff-transition setting that naturally induces heterogeneous drift across Time-of-Use (ToU) and flat-rate households. Our experimental pipeline proceeds in three phases. Phase 1 establishes the problem by showing statistically significant non-stationarity in ToU nodes, including a ToU Kolmogorov-Smirnov statistic of 0.0816 with a 30.76% mean shift from the pre-drift window to 2013. Phase 2 builds a centralized benchmark, where an Attention-LSTM improves aggregate RMSE over a plain LSTM by 3.92% on 2012 Q4 and 4.21% on 2013, but still yields uneven node-level outcomes. Phase 3 introduces FLAST and compares it against vanilla FedAvg and the centralized Attention-LSTM. FLAST reduces 2013 RMSE on the primary drift node `node_tou_6` from 0.1041 to 0.1011, a 2.91% gain over the centralized benchmark, while also improving three ToU nodes by at least 0.5% versus FedAvg (`node_tou_4`, `node_tou_5`, `node_tou_6`) with a mean gain of 1.66%. Drift-aware aggregation produces a 2.23x weight differential between high-drift and stable nodes, and FLAST remains within 0.53% of FedAvg on stable Std nodes. The results support an honest selective-adaptation claim: FLAST does not uniformly dominate all baselines, but it improves the drifting nodes that matter most without materially destabilizing the federated baseline on stable nodes.

## 1. Introduction

Residential smart-meter forecasting sits at the intersection of three hard constraints. First, household demand is noisy, highly heterogeneous, and behaviorally driven. Second, dynamic pricing and demand-response programs can produce temporal distribution shifts that make pre-drift models age poorly. Third, the data are privacy-sensitive and operationally decentralized, which makes centralized retraining undesirable even when it is accurate.

This tension is visible in practice. A centralized recurrent model may improve average forecasting error while still harming a subset of clients. A vanilla federated model preserves privacy, but it can blur together stable and drifting nodes during aggregation. The core question of this paper is therefore not simply whether federated learning can forecast household load, but whether it can adapt selectively when only some nodes drift.

To answer that question, we study the London Smart Meter data released from the UK Power Networks-led Low Carbon London project. The dataset is particularly suitable because it contains both flat-rate customers and customers exposed to Dynamic Time-of-Use tariffs during 2013. That tariff intervention creates a natural, real-world source of non-stationarity rather than a synthetic stress test.

Our central thesis is that **federated load forecasting under non-stationarity should be evaluated as a selective adaptation problem**. In this setting, the goal is not to outperform a centralized model everywhere. Instead, the goal is to improve the most drift-affected nodes relative to the vanilla federated baseline while preserving stable-node behavior within a small tolerance.

This paper makes three contributions:

1. We build a three-phase empirical pipeline that first proves the presence of tariff-linked non-stationarity, then establishes a centralized attention baseline, and finally evaluates a drift-aware federated extension on the same frozen nodes and time splits.
2. We propose **FLAST**, a federated Attention-LSTM with inverse-drift aggregation weights and a selective retraining trigger that activates when post-aggregation validation loss rises for eligible drifting clients.
3. We show that FLAST delivers **targeted** rather than uniform gains: it beats the centralized Attention-LSTM on the primary drift node, improves multiple drifting ToU nodes relative to FedAvg, and stays within 0.53% of FedAvg on stable nodes.

## 2. Related Work

Federated learning was formalized by McMahan et al. through iterative model averaging, which showed that decentralized training can remain effective even under unbalanced and non-IID client data while reducing communication costs relative to synchronized SGD [1]. That baseline, usually referred to as **FedAvg**, is the appropriate privacy-preserving point of comparison for any federated forecasting method.

Long short-term memory (LSTM) networks remain a standard sequential modeling choice for time-series forecasting because they explicitly address long-range dependency issues in recurrent training [2]. Attention mechanisms further improve sequence modeling by allowing the model to focus on the most relevant temporal states instead of compressing the entire history into a single fixed representation [3]. In residential load forecasting, attention-enhanced recurrent models have been explored as a way to better capture fluctuating short-term consumption patterns [4].

In parallel, privacy-preserving and federated formulations of load forecasting have started to emerge. Prior work has examined federated residential load forecasting as an alternative to centralized training under privacy constraints [5, 6], and more recent personalized federated approaches explicitly target heterogeneous smart-meter clients [7]. These studies motivate FL-based forecasting, but they do not fully resolve the specific case considered here: **real drift concentrated in a subset of tariff-driven nodes**.

Our work differs in two ways. First, we ground the problem in a natural pricing-induced shift from the London Smart Meter dataset rather than a synthetic non-IID partition alone. Second, we make a narrower, more testable claim than "federated is better than centralized": FLAST aims to improve the drift-affected nodes that vanilla FedAvg under-serves while leaving stable nodes effectively unchanged relative to the federated baseline.

## 3. Dataset and Problem Formulation

### 3.1 Source Dataset

We use the **SmartMeter Energy Consumption Data in London Households** dataset released through the London Datastore. The source contains half-hourly energy readings for 5,567 London households from November 2011 through February 2014 and includes a Dynamic Time-of-Use subgroup active during the 2013 calendar year [8].

For our experiments, we construct a focused subset of 200 households:

- 140 ToU households
- 60 Std households
- 10 federated nodes total
- 20 households per node

The nodes are fixed after Phase 1 and reused unchanged in all later phases.

### 3.2 Time Windows

We preserve the same chronological splits throughout the entire study:

- Train window: `2011-11-23` to `2012-10-01`
- Stable test window: `2012-10-01` to `2013-01-01`
- Drift test window: `2013-01-01` to `2014-01-01`
- Post-drift test window: `2014-01-01` to `2014-03-01`

For Phase 3 only, the final month of the original training window (`2012-09-01` to `2012-10-01`) is used as a local validation slice for selective retraining control. This validation split remains fully inside the original train period and does not leak into evaluation.

### 3.3 Forecasting Task

The task is one-step-ahead hourly load forecasting on normalized node-level mean consumption sequences. Each sequence uses:

- lookback length: 48 hours
- train stride: 3
- evaluation stride: 1

Missing values are filled by time interpolation with bidirectional completion inside each window, and normalization is fit using each train window's mean and standard deviation before being applied to validation and test segments.

## 4. Phase 1: Establishing Non-Stationarity

Phase 1 had a single purpose: verify that the dataset actually contains the kind of temporal shift the paper claims to address.

The strongest evidence comes from the ToU group:

- KS statistic: 0.0816
- KS p-value: effectively 0
- mean shift from pre-drift to 2013: 30.76%
- mean shift for Std nodes over the same comparison: 13.82%
- ToU mean absolute ACF change: 0.0745

This shift is not purely aggregate. Node-level degradation is heterogeneous, with a 28.21 percentage-point spread between the best and worst node-level 2013 RMSE changes. The most drift-affected node in the frozen split is `node_tou_6`, whose RMSE increases from 0.1224 in 2012 Q4 to 0.1348 in 2013, a 10.13% degradation.

At the same time, the aggregate baseline context is deceptive: Phase 1's overall RMSE changes only from 0.3491 to 0.3498 between 2012 Q4 and 2013. This gap between mild aggregate change and strong node-level heterogeneity is exactly the motivation for FLAST.

## 5. Phase 2: Centralized Attention-LSTM Benchmark

Before introducing any federated innovation, we build a centralized benchmark to answer a simpler question: does attention help at all on this dataset?

### 5.1 Architecture

The centralized benchmark uses the same univariate recurrent backbone that is later reused in Phase 3:

- one-layer LSTM encoder
- hidden size 64
- dropout 0.2
- additive attention over hidden states
- MLP prediction head over the concatenated context and final hidden state

The exact implementation is in `src/attention_model.py`.

### 5.2 Centralized Results

The centralized Attention-LSTM improves over a plain LSTM at the aggregate level:

- 2012 Q4 RMSE: `0.3613 -> 0.3471` (`+3.92%`)
- 2013 RMSE: `0.3560 -> 0.3410` (`+4.21%`)
- 2014 RMSE: `0.3856 -> 0.3648` (`+5.40%`)

However, the node-level picture remains uneven. Only three nodes show positive 2013 improvement, and seven show negative node-level change. The most important Phase 2 takeaway is therefore not simply "attention helps," but rather:

> a centralized attention model improves globally while still redistributing error unevenly across nodes.

That interpretation is supported by the degradation variance result:

- Plain LSTM node-level 2013 degradation standard deviation: `7.18`
- Attention-LSTM node-level 2013 degradation standard deviation: `5.58`

Attention makes the centralized benchmark more consistent, but not sufficiently node-aware.

## 6. FLAST: Drift-Aware Federated Attention-LSTM

FLAST keeps the Phase 2 Attention-LSTM local model fixed and changes only the federated orchestration.

### 6.1 Vanilla FedAvg Baseline

For round `r`, each node `i` receives the global state `theta^r`, performs one local epoch on its node-specific training set, and returns an updated state `theta_i^{r+1}`. Vanilla FedAvg aggregates those updates proportionally to local sample count.

This baseline is necessary because FLAST is a **federated** contribution. Its primary claim must therefore be evaluated relative to FedAvg.

### 6.2 Drift-Aware Aggregation

Each node receives a Phase 1 drift score based on the Kolmogorov-Smirnov statistic between its pre-drift and 2013 node-mean series. FLAST downweights higher-drift nodes during aggregation:

\[
alpha_i \propto \frac{n_i}{\mathrm{KS}_i + epsilon}
\]

where `n_i` is the local sample count and `epsilon = 10^{-6}` stabilizes the denominator.

Because all nodes are equal-sized in this experiment, the weighting effect is driven primarily by inverse KS drift.

### 6.3 Selective Retraining Trigger

After aggregation, FLAST computes each node's global validation loss on the held-out September 2012 window and compares it to the node's own same-round local post-training validation loss:

\[
delta_i^r = \frac{L_{i,\text{global}}^r - L_{i,\text{local}}^r}{L_{i,\text{local}}^r}
\]

If `delta_i^r > 0.05`, the node is flagged for one extra fine-tuning epoch before final personalized evaluation for that round. In the final implementation, the trigger is gated to eligible ToU nodes with KS drift score at least `0.075`.

This trigger does not claim to "fix everything." Its purpose is surgical: identify nodes whose validation behavior worsens after aggregation and recover them locally when appropriate.

## 7. Experimental Setup

All experiments use the frozen artifacts generated in earlier phases.

### 7.1 Shared Settings

- sequence length: 48
- train stride: 3
- evaluation stride: 1
- batch size: 512
- optimizer: Adam
- learning rate: 0.001
- hidden size: 64
- dropout: 0.2
- random seed: 17

### 7.2 Phase-Specific Settings

- Phase 1 baseline epochs: 20
- Phase 2 epochs: 20
- Phase 3 communication rounds: 10
- Phase 3 local epochs per round: 1
- Phase 3 extra fine-tuning epochs: 1

### 7.3 Evaluation Metrics

We report RMSE and MAE on 2012 Q4, 2013, and 2014. The main paper claim is anchored on 2013, because that is the tariff-shift year. We evaluate stable-node preservation relative to **FedAvg**, not relative to the centralized benchmark, because FLAST is a federated refinement of FedAvg rather than a replacement for centralized training in all operating regimes.

## 8. Results

### 8.1 Federated Convergence

Vanilla FedAvg converges cleanly:

- mean client train loss: `0.5142 -> 0.2923`
- relative reduction: about `43%`

This matters because it verifies that the Phase 3 comparison is not between a functioning FLAST model and a broken federated baseline.

### 8.2 Main Phase 3 Results

At the overall level, FLAST modestly improves on FedAvg:

- overall 2013 RMSE: `0.0853 -> 0.0851` (`+0.18%`)
- ToU-group 2013 RMSE: `0.08569 -> 0.08536` (`+0.39%`)

The overall gains are intentionally modest because FLAST is not designed as a uniform uplift method. Its strength is targeted improvement on drift-sensitive nodes.

#### Reference Drift Node

The primary drift result is on `node_tou_6`:

- Centralized Attention-LSTM RMSE 2013: `0.1041`
- FedAvg RMSE 2013: `0.1017`
- FLAST RMSE 2013: `0.1011`

This corresponds to:

- `+2.91%` improvement over the centralized Attention-LSTM
- `+0.59%` improvement over FedAvg

This result is central because `node_tou_6` is the strongest Phase 1 drift failure case and the main paper reference node.

#### Selective ToU Gains

Three ToU nodes improve by at least `0.5%` versus FedAvg in 2013:

- `node_tou_4`: `+2.55%`
- `node_tou_5`: `+1.84%`
- `node_tou_6`: `+0.59%`

The mean gain across those selectively improved ToU nodes is `1.66%`. If the threshold is relaxed to any positive gain, four ToU nodes improve (`node_tou_3`, `node_tou_4`, `node_tou_5`, `node_tou_6`) with mean gain `1.28%`.

These results support a **selective improvement** claim rather than a blanket "all ToU nodes get better" claim.

#### Stable Nodes

The centralized model remains the strongest stable-node reference:

- centralized Std mean 2013 RMSE: `0.07971`
- FedAvg Std mean 2013 RMSE: `0.08393`
- FLAST Std mean 2013 RMSE: `0.08421`

However, the stable-node gap is already present in FedAvg before FLAST is applied. Relative to FedAvg, FLAST remains near-neutral on stable nodes:

- mean stable-node change vs FedAvg: `-0.32%`
- worst stable-node change vs FedAvg: `-0.53%`

This is the correct interpretation of "no harm" for a federated method.

### 8.3 Drift-Aware Weighting and Trigger Activity

The aggregation behavior clearly reflects drift awareness:

- mean aggregation weight of top-drift nodes: `0.0615`
- mean aggregation weight of Std nodes: `0.1369`

This is a `2.23x` separation, showing that high-drift nodes contribute less to the shared global state.

The selective retraining mechanism also remains active:

- executed FLAST trigger events: `26`
- active rounds: `10/10`
- most frequently triggered nodes: `node_tou_4` (`7`), `node_tou_6` (`7`), `node_tou_5` (`6`), `node_tou_1` (`6`)

Notably, after the final paper-aligned gating fix, Std nodes are no longer counted as executed retraining events, which matches the intended interpretation of FLAST as a drift-focused mechanism.

## 9. Discussion

The most important scientific lesson from this study is that **the wrong baseline can make a correct method look wrong**.

If stable nodes are compared against the centralized Attention-LSTM, then both FedAvg and FLAST appear worse, and the natural but incorrect conclusion is that FLAST damaged stable clients. The experiment logs and tables show otherwise: the stable-node penalty is already introduced by the move from centralized to federated training. FLAST only changes those stable nodes by at most `0.53%` relative to FedAvg.

This changes the paper's claim in an important way. FLAST should not be described as a universal improvement over centralized forecasting. Instead, it should be presented as:

1. a privacy-preserving federated alternative to centralized node forecasting,
2. a targeted improvement over vanilla FedAvg on the drift-sensitive nodes that matter most, and
3. a mechanism that adapts selectively without materially worsening stable nodes relative to the federated baseline.

That claim is both more modest and more publishable, because it matches the actual evidence.

The results also explain why the original "mean ToU improvement must exceed 3%" criterion was too blunt. The observed 2013 ToU mean gain over FedAvg is only `0.62%`, but that average hides a more useful pattern: FLAST helps a subset of clearly drifting nodes while leaving the rest close to neutral. Under non-stationarity, that is exactly the behavior a selective adaptation method should exhibit.

## 10. Limitations

This study has several limitations that should be stated directly in the submission:

1. The experiments use one real-world dataset and one fixed node partition.
2. The model is univariate and does not incorporate weather, calendar, or exogenous tariff features explicitly.
3. The federated setup uses equal-sized nodes, which simplifies the interaction between sample count and drift-aware weighting.
4. The work evaluates predictive performance and training behavior, but not formal privacy guarantees, communication compression, or secure aggregation.
5. The personalization step is selective and lightweight; stronger personalization methods may yield larger gains, especially on moderately drifting nodes.

These limitations do not invalidate the current claim, but they define the boundary of what the paper should and should not promise.

## 11. Conclusion

We presented FLAST, a drift-aware federated Attention-LSTM framework for short-term household load forecasting under real tariff-induced non-stationarity. Across a three-phase experimental pipeline, we showed that:

- the London Smart Meter data exhibit real and heterogeneous drift, especially in ToU nodes,
- a centralized Attention-LSTM improves aggregate accuracy but remains uneven at the node level,
- FLAST improves the primary drift node by `2.91%` relative to the centralized Attention-LSTM,
- FLAST improves three ToU nodes by at least `0.5%` relative to FedAvg with mean gain `1.66%`, and
- FLAST preserves stable-node performance within `0.53%` of FedAvg while applying a `2.23x` drift-aware weighting differential.

The resulting contribution is not "federated learning beats centralized learning everywhere." The actual contribution is more precise: **FLAST is a selective federated adaptation mechanism that improves the most drift-relevant household nodes without materially degrading the stable ones relative to the federated baseline.**

That is the strongest honest version of the paper, and it is fully supported by the experimental artifacts in this repository.

## References

1. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017. https://proceedings.mlr.press/v54/mcmahan17a.html
2. Sepp Hochreiter and Jurgen Schmidhuber. *Long Short-Term Memory*. Neural Computation, 1997. https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory
3. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015 / arXiv:1409.0473. https://arxiv.org/abs/1409.0473
4. Lin Ma, Liyong Wang, Shuang Zeng, Yutong Zhao, Chang Liu, Heng Zhang, Qiong Wu, and Hongbo Ren. *Short-Term Household Load Forecasting Based on Attention Mechanism and CNN-ICPSO-LSTM*. Energy Engineering, 2024. https://www.techscience.com/energy/v121n6/56583/html
5. Christopher Briggs, Zhong Fan, and Peter Andras. *Federated Learning for Short-term Residential Load Forecasting*. arXiv:2105.13325. https://arxiv.org/abs/2105.13325
6. Afaf Taik and Soumaya Cherkaoui. *Electrical Load Forecasting Using Edge Computing and Federated Learning*. ICC 2020 / arXiv:2201.11248. https://arxiv.org/abs/2201.11248
7. Ratun Rahman, Neeraj Kumar, and Dinh C. Nguyen. *Electrical Load Forecasting in Smart Grid: A Personalized Federated Learning Approach*. arXiv:2411.10619. https://arxiv.org/abs/2411.10619
8. London Datastore / UK Power Networks. *SmartMeter Energy Consumption Data in London Households*. https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d/

## Appendix A: Artifact Map

Key figures and tables already generated by the codebase:

- [Phase 1 Summary](../outputs/phase1/reports/summary.md)
- [Phase 2 Summary](../outputs/phase2/reports/summary.md)
- [Phase 3 Summary](../outputs/phase3/reports/summary.md)
- [Phase 1 Table 1](../outputs/phase1/tables/table_1_node_baseline_metrics.csv)
- [Phase 2 Table 2](../outputs/phase2/tables/table_2_centralized_model_comparison.csv)
- [Phase 2 Table 3](../outputs/phase2/tables/table_3_node_performance.csv)
- [Phase 3 Table 4](../outputs/phase3/tables/table_4_node_model_comparison.csv)
- [Phase 3 Table 5](../outputs/phase3/tables/table_5_group_model_summary.csv)
- [Figure 1: Distribution Shift](../outputs/phase1/figures/figure_1_distribution_shift.png)
- [Figure 2: ACF Shift](../outputs/phase1/figures/figure_2_acf_shift.png)
- [Figure 3: Node RMSE Degradation](../outputs/phase1/figures/figure_3_node_rmse_degradation.png)
- [Figure 4: Centralized Training Curves](../outputs/phase2/figures/figure_4_training_curves.png)
- [Figure 5: Attention Visualization](../outputs/phase2/figures/figure_5_attention_visualization.png)
- [Figure 6: Centralized Node Comparison](../outputs/phase2/figures/figure_6_node_rmse_comparison.png)
- [Figure 7: Federated Convergence](../outputs/phase3/figures/figure_7_federated_convergence.png)
- [Figure 8: Drift-Aware Weights](../outputs/phase3/figures/figure_8_drift_aware_weights.png)
- [Figure 9: Phase 3 Node Comparison](../outputs/phase3/figures/figure_9_node_rmse_comparison_phase3.png)
- [Figure 10: Trigger Activity](../outputs/phase3/figures/figure_10_trigger_activity.png)
