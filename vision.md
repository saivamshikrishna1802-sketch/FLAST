Phase Plan
Phase 1 — Dataset Selection & Validation (do this first, before any code)
Find a publicly available dataset that actually exhibits non-stationarity — not just any load data. If the data doesn't show the problem, your solution has nothing to prove. We'll audit 2-3 candidate datasets and pick the one that best demonstrates temporal, behavioral, and structural shifts.
Phase 2 — Baseline Implementation
Code a standard Attention-LSTM first, run it, get a benchmark accuracy. This becomes your "existing method" comparison. Without this, reviewers have nothing to compare your framework against.
Phase 3 — FLAST Framework Implementation
Build the three layers we designed — local Attention-LSTM nodes, drift-aware FL aggregation, selective retraining trigger. Run it on the same dataset.
Phase 4 — Results & Analysis
Compare FLAST vs baseline across normal conditions and deliberately injected non-stationarity scenarios. Generate the tables and plots that go into the paper.
Phase 5 — Paper Draft
Write around the results, not before them. Most students do this backwards.

