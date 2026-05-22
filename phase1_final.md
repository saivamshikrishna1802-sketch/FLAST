EXACT THINGS LEFT TO DO
STEP 1 — Implement household-level stacked training

This is the ONLY important code change remaining.

Right now:

each node trains on the averaged node series

That smooths the behavioral drift.

Instead:

train using all individual household sequences inside the node
still evaluate on the node-average signal

This preserves:

node heterogeneity
behavioral variance
realistic FL behavior

while keeping evaluation stable.

EXACT CHANGE

Inside:

run_node_lstm_analysis()

replace:

node_series = mean(node households)

training logic

with:

train on stacked household sequences
evaluate on node mean series

The earlier diagnostic already identified the correct modification path.

STEP 2 — Run ONE final Phase 1 execution

ONLY ONE.

Use:

{
  "baseline_epochs": 20,
  "hidden_size": 64,
  "sequence_length": 48,
  "train_stride": 3,
  "degradation_threshold_pct": 10.0
}

Do NOT keep experimenting after this run.

STEP 3 — Regenerate ONLY these outputs

You need these finalized artifacts:

REQUIRED FINAL OUTPUTS
Figure 1

Distribution shift KDE plot

Already good.

Keep it.

Figure 2

ACF behavioral shift plot

Already strong.

Keep it.

Figure 3

Node-level RMSE degradation bar chart

Regenerate after stacked-household training.

This is your key Phase 1 figure.

Table 1

Per-node baseline metrics

Columns:
| Node | Type | RMSE 2012Q4 | RMSE 2013 | % Change |

This becomes:

your baseline benchmark
your Phase 2 comparison target
acceptance_report.json

Final acceptance status.

Even if:

only 2–3 nodes exceed 10%

that is acceptable now.

STEP 4 — FINALIZE ACCEPTANCE CRITERIA

These are now the OFFICIAL Phase 1 criteria.

FINAL ACCEPTANCE CRITERIA
Criterion	Required?	Status Goal
KS distribution shift significant	Mandatory	PASS
ToU shift > Std shift	Mandatory	PASS
ACF structure changes in ToU group	Mandatory	PASS
Node-level heterogeneity visible	Mandatory	PASS
≥2–3 nodes exceed 10% degradation	Preferred	PASS if possible
No data leakage	Mandatory	PASS
IMPORTANT: REMOVE ADF COMPLETELY

Do NOT use:

ADF
stationarity-failure percentages

in your paper anymore.

Why:

conceptually wrong tool
weak evidence
distracts from stronger KS + ACF evidence

Your paper becomes MUCH cleaner without it.

STEP 5 — WRITE THE PHASE 1 CONCLUSION

Once the rerun finishes successfully:

Your Phase 1 conclusion becomes:

“Analysis of the London Smart Meter dataset reveals statistically significant distributional and behavioral non-stationarity, particularly within Time-of-Use tariff nodes. While aggregate forecasting metrics partially conceal this effect, node-level evaluation demonstrates heterogeneous degradation patterns across distributed smart-grid regions, motivating the need for adaptive federated forecasting mechanisms.”

That sentence transitions perfectly into Phase 2.

THEN PHASE 1 IS CLOSED

After that:

freeze all Phase 1 code
freeze dataset selection
freeze preprocessing
freeze plots

No more tuning.

    