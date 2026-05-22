Phase 1 — What we need to do in code
Phase 1 has one job: prove the problem exists in this dataset before we claim to solve it. If you can't show non-stationarity clearly, the paper has no motivation.
Three tasks in order
Task 1 — Data Loading & Cleaning

Load the dataset, parse timestamps, handle the ~1.25% missing values
Sample a working subset — 200 households is enough, stratified so you have both tariff groups represented
Resample to hourly (from half-hourly) — cleaner for modeling
Split into three temporal windows: 2011–2012 (pre-drift), 2013 (drift period, tariff change), 2014 (post-drift)

Task 2 — Non-Stationarity Analysis
This is the core of Phase 1. Three specific tests:

Statistical drift test — Run the Augmented Dickey-Fuller (ADF) test on consumption time series per household. Non-stationary series will fail stationarity. Plot the proportion of households failing ADF across the three time windows — if it jumps in 2013, you have your evidence.
Distribution shift visualization — Plot load distribution (histogram/KDE) for the flat-rate group vs the tariff-change group across all three windows. The tariff-change group should show a visible shift in 2013 that the flat-rate group doesn't.
Temporal autocorrelation change — Plot ACF (autocorrelation function) for both groups across windows. If behavioral patterns shift, the autocorrelation structure changes. This directly motivates why a static model trained on 2011–2012 fails in 2013.

Task 3 — Baseline Model Failure Demonstration
Train a simple LSTM (not even Attention yet — keep it simple for Phase 1) on 2011–2012 data. Test it on 2013 data. Measure RMSE and MAE. This number becomes your "existing method fails" evidence in the paper.

Acceptance Criteria to Move to Phase 2
Be strict here. If these aren't met, Phase 2 results mean nothing.
CriteriaWhat you need to seeWhyADF test shows drift>30% more households failing stationarity in 2013 vs 2011–2012Proves non-stationarity is real, not noiseDistribution shift is visibleKDE plots show measurable mean/variance shift in tariff group in 2013Visual evidence for the paper figuresFlat-rate group stays stableFlat-rate KDE plots don't shift significantlyProves the shift is behavioral, not seasonalBaseline LSTM degradesRMSE increases by at least 15–20% from 2012 test to 2013 testQuantifies the problem your framework solvesNo data leakageTrain/test split is strictly temporal, no shufflingIf violated, all results are invalid
If all five are met — move to Phase 2. If the LSTM degradation is less than 15%, it doesn't mean your framework won't work, it means the dataset subset you sampled isn't showing the effect strongly enough — resample with more tariff-change households and rerun.

What Phase 1 produces for the paper
By the end of Phase 1 you will have:

Figure 1 — KDE distribution shift plots (goes in Section II, Motivation)
Figure 2 — ADF failure rate across time windows (goes in Section II)
Table 1 — Baseline LSTM RMSE/MAE across time windows (goes in Section IV, as your comparison benchmark)

These three outputs directly write your motivation section. You're not describing the problem in words — you're showing it with numbers from real data.@