# Mini Loto (ミニロト) AI Research — Final Report

Draws analyzed: round 1 – 1404 (1999-04-13 to 2026-09-15). Target: DRAW 1405.
SEED = 20260920 (fixed throughout). All code in `scripts/`, all intermediate
and final data in `data/` and `results/`.

## 1. Data sources

Two independent full-history sources, both public GitHub-hosted mirrors of
the official Mizuho Bank mini loto results (JSON):

- **Source A** (primary): `tk030-lotto/lotto-data-hub` — `data/miniloto.json`, all 1404 rounds.
- **Source B** (cross-check + prize data): `ddsky0728/Japanese-lottery-data` — `loto/miniloto_history.json`, all 1404 rounds, includes 1st-4th prize winner counts and per-winner amounts.
- **Source C** (partial cross-check): `tank1159jhs/jp-lottery-api` — `data/miniloto/all.json`, rounds ~1305-1404 only.

**Network constraint (disclosed):** this session runs behind an organization
egress proxy that allow-lists only GitHub and standard package-registry
domains. Every site named in the task brief as a preferred source
(mizuhobank.co.jp, mk-mode.com, miniloto.thekyo.jp, sougaku.com, loto-life.net)
returned `EGRESS_BLOCKED`/403 on every attempted fetch method (curl, and
Anthropic's own WebFetch tool). GitHub-hosted mirrors of the same official
results were the only reachable source and were used instead, after
confirming they agree with each other exactly (see §2). This is a real
access limitation of this sandboxed session, not a silent substitution.

Set-ball (A-J) history and drawing-venue history (Steps 16-17) could not be
obtained for the same reason — no GitHub-hosted machine-readable mirror of
either exists, and every site that compiles them by hand is blocked. Per the
"never fabricate missing data" rule, these features are simply **excluded**
from every model; see `reports/setball_venue_note.md`.

## 2. Completeness audit (round 1-1404)

Full result in `reports/data_audit.md`. **OVERALL: PASS.**

- Rows: 1404/1404. draw_no range 1-1404, 0 missing, 0 duplicates.
- Every draw: exactly 5 main numbers, all in 1-31, no duplicates within a draw.
- Bonus: 1-31, never equal to a main number.
- Total main numbers = 7020 (1404×5) ✓. Total main+bonus = 8424 ✓.
- Dates strictly increasing with draw_no.

## 3. Source cross-validation

- Source A vs Source B: **0 mismatches** across all 1404 overlapping rounds (numbers + bonus).
- Source A vs Source C: **0 mismatches** across the 100-round overlap.
- `data/source_comparison.csv` logs every compared round; it is empty because nothing disagreed.
- Limitation disclosed: rounds 355-1301 have full A-vs-B cross-validation but only a single source (no third mirror) reachable for those rounds beyond A and B — C only covers the most recent ~100 rounds. Given A and B agree on literally every round including that range, this is treated as adequately verified, but the gap in independent 3-way coverage is disclosed rather than hidden.

## 4. Theoretical (fair-draw) baseline

`scripts/theoretical_probabilities.py`, cross-checked independently in
`scripts/audit_final.py`.

- C(31,5) = **169,911** (confirmed by scipy comb, manual factorial, and itertools enumeration — 3 independent computations agree).
- P(a fixed number is a main number) = 5/31 = 0.16129.
- P(fixed number is bonus, given not main) = 1/26; P(main or bonus overall) ≈ 0.1935.
- Ticket-vs-draw match distribution (Hypergeometric(31,5,5)): P(0)=0.3871, P(1)=0.4399, P(2)=0.1530, P(3)=0.01913, P(4)=0.000765, P(5)=0.00000589 (1 in 169,911, exactly).
- Prize-tier probabilities: 1st 5.885e-6, 2nd 2.943e-5, 3rd 7.357e-4, 4th 0.01913.

All downstream "is there a signal" questions are framed as deviations from
this exact baseline.

## 5. Single-number analysis (Step 5)

`results/number_statistics.csv`. Over 1404 draws, main-number counts range
196-266 (theoretical mean 226.45, sd≈13.8). Only 3/31 numbers show
uncorrected p<0.05 (chance expectation ≈1.55) — and **0/31 survive
Bonferroni, Holm, or Benjamini-Hochberg FDR correction**. Same pattern for
bonus counts and main+bonus counts. **No individual number departs from a
fair draw once multiple comparisons are accounted for.**

## 6. Rolling / EWMA frequency (Step 6)

`results/rolling_statistics.csv`: windows {10,20,30,50,75,100,125,150,200,300,500}
and EWMA half-lives {10,20,30,50,75,100,150,200} computed causally (using
only prior draws) for every number at every draw_no. Used as walk-forward
model features, not judged for "best window" on full-period hindsight — the
window/half-life actually used by the C/D baselines in walk-forward is
selected fold-by-fold from training data only (`results/walk_forward_window_choices.csv`).

## 7. Gap / overdue analysis (Step 7)

`results/gap_statistics.csv`, `results/gap_reappearance_curve.csv`,
`results/gap_gamblers_fallacy_test.csv`. Pooled logistic regression of
"appears next draw" on "current gap length" across all 43,319 (number, draw)
observations: coefficient = -0.00357, **p = 0.113 (not significant)**,
Pearson r = -0.0076. No evidence for either the "overdue" or "hot-hand"
hypothesis; consistent with gambler's-fallacy-type patterns being absent
(as they should be for a physically fair draw).

## 8. Previous-draw transition analysis (Step 8)

`results/transition_overlap_statistics.csv`, `results/transition_statistics.csv`.
Distribution of how many of a draw's 5 numbers repeat from the immediately
preceding draw matches the theoretical Hypergeometric(31,5,5) distribution
closely (χ²=4.06, df=5, p=0.54). Per-number "does appearing last draw change
next-draw probability" test: **0/31 numbers FDR-significant.** One
borderline result: the previous draw's bonus number reappearing as next
draw's main number occurred at rate 0.1418 vs theoretical 0.1613 (binomial
p=0.0499) — a single raw p right at the 0.05 boundary out of the large
family of tests run in this project; not treated as a finding.

## 9. All 465 pairs (Step 9)

`results/pair_statistics.csv`, period-split stability in
`results/pair_statistics_by_period.csv` / `results/pair_stability_check.csv`.
23/465 pairs show uncorrected p<0.05 (chance expectation ≈23.25 at α=0.05).
**0/465 survive Bonferroni, Holm, or BH-FDR correction.**

## 10. All 4,495 triples (Step 10)

`results/triple_statistics.csv`, with a pre-specified (not outcome-tuned)
Beta-Binomial shrinkage estimator (prior strength = 50 pseudo-draws) given
the low expected count (~3.1 per triple over 1404 draws). 181/4495
uncorrected p<0.05 (chance expectation ≈225 — actually *below* chance).
**0/4495 survive any correction.**

## 11-12. 4-5 number co-occurrence, combination shape (Steps 11-12)

Extending to 4- and 5-number co-occurrence was judged too sparse to test
reliably (expected count per specific 5-set over 1404 draws ≈ 0.008) and was
not pursued beyond the shape-feature analysis below, to avoid the exact
kind of overfitting-on-noise the brief warns against.

`results/structural_shape_features.csv` / `_summary.csv`: observed draws'
sum (mean 80.91), odd-count (mean 2.60), low-count (mean 2.36) all match the
theoretical distribution computed directly from all 169,911 combinations
(80.00, 2.58, 2.42 respectively) within sampling noise. Chi-square
goodness-of-fit on the odd-count distribution: χ²=4.79, p=0.44 (no
departure from theory).

## 13. Autocorrelation / periodicity (Step 13)

`results/timeseries_autocorrelation.csv`: per-number ACF(lag 1), Ljung-Box
(lag 10), and Wald-Wolfowitz runs test on each number's 0/1 occurrence
series. **0/31 numbers FDR-significant on Ljung-Box.** No further FFT/
periodogram analysis was pursued given this null result at the simpler test
stage (would only add more comparisons to correct for, for no expected gain).

## 14. Structural change (Step 14)

`results/structural_statistics.csv`: CUSUM probe on the draw-sum series,
candidate change point near draw 217, **permutation p = 0.618 (n=2000
permutations)** — not distinguishable from a random fluctuation in a
stationary series. No adopted change point.

## 15. Date features (Step 15)

Every one of the 1404 draws falls on a **Tuesday** (`results/date_weekday_distribution.csv`)
— the draw schedule is fixed, so weekday carries zero information by
construction, exactly as anticipated in the task brief, and was excluded.
Season/month features were not pursued further given the null results at
every simpler stage above (avoiding unnecessary additional comparisons).

## 16-17. Set ball / venue (Steps 16-17)

**Not obtainable** — see §1 and `reports/setball_venue_note.md`. No
fabricated data was substituted; these features are absent from every model
in this project.

## 18-21. Baselines, ML models, walk-forward (Steps 18-21)

`scripts/feature_engineering.py` builds a strictly causal
(draw_no, number) feature table — every feature at row *t* uses only draws
before *t* (verified: `y.mean()` over the whole table equals exactly
5/31, confirming no leakage of the target into features).

`scripts/walk_forward.py`: **45 expanding-window folds**, retrained every 25
draws from draw 300 onward (compute-driven simplification, disclosed in the
script's docstring — see §"limitations"), 34,224 total (draw, number)
out-of-sample predictions. 10 baselines (A random, B full-period frequency,
C best rolling window *selected inside the training fold only*, D best EWMA
half-life *ditto*, E gap-based, F hot, G cold, H pair co-occurrence lift,
J Bayesian-shrinkage frequency, K previous-draw transition rate) and 8 ML
models (Logistic Regression L1/L2/ElasticNet, Random Forest, Extra Trees,
Gradient Boosting, XGBoost, LightGBM), all evaluated by Brier score, log
loss, AUC, and mean matches from a top-5-by-score "predicted ticket" per
draw. Full table: `results/model_performance.csv`.

**Headline result: the random/uniform baseline (A_random) has the lowest
(best) Brier score of all 18 models tested (0.135276).** Every other model's
Brier score is equal to or worse. AUC for every model sits at essentially
0.50 (range 0.4965-0.5095). A paired Wilcoxon signed-rank test (per-draw
Brier score, model vs. random, BH-FDR corrected across all 17 comparisons)
finds: **`sig_better_than_random_fdr` is False for every single model.**
Mean matches-per-draw for every model cluster around the theoretical random
value (0.8065), with no model reliably above it.

## 22. Combination scoring of all 169,911 (Step 22)

Pre-registered adoption rule (decided in `walk_forward.py`, before this
step ran): a model enters the final ensemble only if it beat random with
FDR-significant OOS Brier improvement. **Zero models qualified.** Per this
rule, `scripts/combination_rank.py` scores every combination identically
(1/169911 — the exact fair-draw probability) and ranks them only by a
fixed, pre-registered seeded tie-break (`np.random.default_rng(20260920)`),
applied after scoring so it cannot be tuned to any particular outcome. This
*is* the correct, honest output of the walk-forward result, not a
placeholder. `results/top169911.csv`, `results/top100.csv`.

An **exploratory, explicitly non-adopted** composite score (mean predicted
probability across the 8 tested ML models, fit once on the full 1-1404
history) is saved separately for research transparency in
`results/exploratory_unvalidated_number_scores_draw1405.csv` — it is not
used anywhere in the official ranking or ticket selection.

## 23-25. 10-ticket strategies and backtest (Steps 23-25)

`scripts/portfolio_10.py`, `results/strategy10_performance.csv` /
`_summary.csv`. Four strategies backtested over the full 1404-draw history
with real historical prize amounts (`data/miniloto_1_1404_clean.csv`
rank1-4 columns from Source B):

| Strategy | Mean max match/draw | Rate ≥3 match | Net profit (¥) | ROI |
|---|---|---|---|---|
| A Pure random | 2.069 | 0.209 | -2,433,400 | -86.66% |
| B Diversified (overlap≤2) | 2.080 | 0.209 | -2,433,600 | -86.67% |
| C Coverage-optimized | 2.130 | 0.197 | -2,445,600 | -87.09% |
| D Prize/ROI-optimized (naive, full-history fit) | 1.985 | 0.199 | -2,115,800 | -75.35% |

Strategy D was built on a real-data regression (not an assumed popularity
model): does a combination's "typical/human-picked" shape (low sum
deviation, consecutive pairs, low-number bias, birthday-range bias, number
of distinct decades used) predict a higher ratio of 1st-prize winners to
4th-prize winners (a volume-normalized popularity proxy, both from real
prize data)? Fit on the full 1404-draw history: `decade_spread` was
FDR-significant (coef +0.000053, p_fdr=0.025) — spreading numbers across
more decades correlates with *more* co-winners, i.e., is a "popular" shape.
Selecting against this (favoring low decade_spread) is what drove Strategy
D's better-looking backtest ROI above.

**Self-audit caught a leakage problem here (Step 40, see §"self-audit"
below) and the finding did not survive a proper causal test:** refitting the
same regression on the *first half only* (draws 1-702) and backtesting the
resulting strategy on the *untouched second half* (draws 703-1404) gives:
`decade_spread` p_fdr = 0.102 (**not significant**), and — because the code's
pre-registered fallback rule is "if no predictor survives FDR, revert to
pure random" — the OOS backtest of Strategy D on the second half is
**bit-for-bit identical** to Strategy A (mean match 2.084, profit
-¥1,200,500, ROI -85.51%, both strategies, to the decimal). See
`results/strategy_d_regression_firsthalf_only.csv` and
`results/strategy_d_oos_secondhalf_backtest.csv`.

**Conclusion: the apparent Strategy D edge was an artifact of fitting and
testing on the same (full-history) data. Under a genuine train/test split it
vanishes.** This is reported prominently rather than kept as the flashier
full-history number, per the task's explicit instruction not to keep the
version that looks better after the fact.

## 26. Monte Carlo (Step 26)

`scripts/monte_carlo.py`, `results/monte_carlo_results.csv` /
`_comparison.csv`. 100,000 simulated random 10-ticket-per-draw players,
replayed over the real 1404-draw history with real prize amounts (ticket
match counts drawn from the exact Hypergeometric(31,5,5) distribution,
which is provably the correct distribution for any draw-blind ticket
regardless of which specific numbers it holds).

| Strategy | Profit percentile vs. 100k random players | Max-drawdown percentile |
|---|---|---|
| A Pure random | 34.1 | 35.3 |
| B Diversified | 34.0 | 35.1 |
| C Coverage | 26.4 | 27.4 |
| D (naive, full-history-fit) | 87.9 | 90.5 |

D's high percentile here is the **same in-sample-fit artifact** flagged in
§23-25 — consistent with, not independent evidence against, the "no edge"
conclusion once the leakage is accounted for. A, B, C all land close to the
40-60th-percentile band expected of an ordinary random player, exactly as
theory predicts.

## 27. Permutation test (Step 27)

Covered inside the Step 14 structural-change probe (2000 permutations of
the draw-sum series; observed CUSUM statistic not distinguishable from
permuted-order statistics, p=0.618) and inside the Wilcoxon-based model
comparisons (rank-based, non-parametric, robust to the exact distributional
shape of Brier-score differences). Both are consistent with "no detectable
non-random temporal structure."

## 28. Negative controls (Step 28)

`results/negative_control_feature_importance.csv`. Two pure-noise features
(Gaussian, Uniform) and one deliberately fabricated "lucky number 7"
indicator were added to the full feature set and a Random Forest was fit on
the complete 37,324-row causal feature table. **The two noise features
ranked #2 and #5 of 15 features by impurity-based importance — higher than
`freq_last_20`, `prev_appeared`, and `gap_since_seen`.** This is a strong
confirmatory negative control: the model cannot separate genuine features
from pure noise by importance, consistent with there being no real signal
in *any* of the tested features, not just the synthetic ones. (The
fabricated "lucky number 7" flag correctly ranked dead last, 15/15 — at
least that one behaved as expected.)

## 29. Multiple testing (Step 29)

Applied throughout: Bonferroni, Holm, and Benjamini-Hochberg FDR correction
on every large hypothesis family (31 numbers ×3 target types, 465 pairs,
4495 triples, 31 transition tests, 31 Ljung-Box tests, 17 model-vs-random
Wilcoxon tests, 5 Strategy-D shape predictors). Uncorrected p<0.05 counts
were consistently close to or below chance-level expectation everywhere,
and **nothing survived correction anywhere in the project** except the
Strategy-D `decade_spread` result, which itself failed to replicate
out-of-sample (§23-25) — the single case where a "significant" result
appeared is also the one case flagged and retracted after a proper causal
check, which is the multiple-testing discipline working as intended.

## 30. Discovery / Validation / Holdout (Step 30)

Implemented as: (a) the entire walk-forward evaluation (§18-21) is itself a
continuous rolling holdout — every one of the 34,224 predictions is made
with zero information from its own or any later draw; (b) the Strategy-D
regression was explicitly re-run as a first-half-discovery /
second-half-holdout split (§23-25) after the naive full-history version was
flagged as leaky; (c) window/half-life selection for baselines C/D is
chosen inside each training fold only (`results/walk_forward_window_choices.csv`),
never from full-period hindsight.

## 31. Model adoption criteria (Step 31)

Pre-registered rule: OOS Brier improvement over random, FDR-significant
across all models tested, in `walk_forward.py`, before combination scoring
ran. **Result: 0 of 18 models met the bar.** No model or strategy is in the
final ensemble as a result — this is not a failure of the pipeline, it is
the pipeline correctly reporting a true negative.

## 32. Robustness (Step 32)

- 4 rolling windows (20/50/100/200) and 3 EWMA half-lives (20/50/100) tested; window/half-life selection itself internally cross-validated per fold — no single arbitrary choice drives any baseline's result.
- 8 different ML model families (linear, tree-ensemble bagging, tree-ensemble boosting ×3 implementations) all converge to the same null result — the finding is not an artifact of one algorithm's inductive bias.
- Pair/triple significance checked for period-split stability (`results/pair_stability_check.csv`); the earlier `decade_spread` finding was explicitly re-tested chronologically (discovery/holdout) and did not survive — the one place robustness testing actually changed the reported conclusion.
- Negative controls (§28) directly test robustness to spurious feature "importance."

## 33-37. Final ensemble, strategy, DRAW 1405 picks

Final ensemble = **empty** (§31); official combination score = uniform
1/169911 for all 169,911 combinations (`results/top169911.csv`).
Final ticket-selection strategy: **B (diversified random)**, chosen purely
for ordinary portfolio hygiene — capping pairwise number overlap between
tickets at 2 avoids buying near-duplicate tickets — **not** for any claimed
predictive or economic edge (§23-25 showed B is statistically
indistinguishable from A, pure random). Generated with the same
`SEED + draw_no` formula used unmodified for all 1404 historical backtest
draws, now applied to `draw_no = 1405`, decided and coded before this
number was ever computed.

**AI FINAL PICKS — DRAW 1405** (`results/final_10.csv`):

| # | Numbers |
|---|---|
| 1 | 02 04 06 19 30 |
| 2 | 02 09 18 30 31 |
| 3 | 12 16 18 19 22 |
| 4 | 03 05 10 17 29 |
| 5 | 01 02 12 17 30 |
| 6 | 07 13 15 18 27 |
| 7 | 01 06 15 24 26 |
| 8 | 03 07 17 18 21 |
| 9 | 18 19 25 28 30 |
| 10 | 02 12 20 28 29 |

Coverage: 27/31 distinct numbers used, 92/100 distinct pairs, 100/100
distinct triples across the 10 tickets. Total cost: ¥2,000 (10 × ¥200).

## 38. Verdict (Step 38)

**B: ランダムを安定して上回る予測優位性は確認できなかった。**
("No statistically reliable predictive edge over random selection was
confirmed.") Zero of 18 models beat random OOS after correction; the one
apparent economic-edge finding (Strategy D) failed a proper causal
discovery/holdout test; negative controls show the feature set cannot even
be distinguished from pure noise by model importance. This is a genuine,
reproducible null result, not a failure to try hard enough — see §18-29.

Note, per the task's own instruction: this verdict describes *statistical*
predictability, not physical possibility — it is not a proof that literally
no signal could ever exist in a physical ball-draw mechanism, only that
none was detected in 1404 draws' worth of publicly available data using the
methods in this project, which is the strongest claim the evidence
supports.

## 39. Comparison with external predictions (Step 39)

**Not performed.** Every external "AI prediction" / "recommended numbers"
site is on the same blocked-domain list as the raw data sources (§1), so no
comparison was possible from this sandboxed session — disclosed rather than
skipped silently. The final 10 tickets above were locked before any attempt
to look for such sites was made, consistent with the task's ordering
requirement regardless.

## 40. Self-audit (Step 40)

Deliberately re-read every script adversarially for leakage. Found and
fixed one real issue: **Strategy D's popularity regression was originally
fit on the full 1404-draw history and then "backtested" against that same
full history — classic train/test contamination.** Caught during self-audit,
fixed by re-running as a first-half-discovery / second-half-holdout split
(§23-25), which reversed the conclusion from "Strategy D has an edge" to
"Strategy D has no edge once tested properly." This is disclosed exactly as
it happened, including the wrong-looking intermediate result, per the task's
explicit instruction not to discard poorly-performing or embarrassing
findings.

Other leakage checks performed and passed: `feature_engineering.py`'s
per-row target mean equals exactly 5/31 (confirms no target leakage into
features); walk-forward fold boundaries strictly increasing in `t_idx`
(verified no fold's training data includes any row from its own or a later
prediction block); rolling-window and EWMA features constructed with
explicit `occ[:t]` / `occ[lo:t]` slicing (excludes index t itself);
`combination_rank.py`'s exploratory ensemble score for draw 1405 is
explicitly separated from, and never merged into, the official ranking.

## 41. Independent recompute (Step 41)

`scripts/audit_final.py` — a from-scratch re-implementation using only the
Python standard library (no pandas/numpy) re-derives: C(31,5) via manual
factorial (169,911 ✓), row count (1404 ✓), draw_no uniqueness/range,
total main numbers (7020 ✓), main-number value range (1-31 ✓), number 11's
main count (266, matches the pandas-based `number_statistics.csv` exactly),
the 169,911-combination file's row count, and structural validity of
`final_10.csv`. **All checks PASS.**

## 42. Sensitivity (Step 42)

Because the adopted final model is the uniform/null model (§31), "does the
top pick change under small perturbations" is reframed honestly: a
correctly-functioning random draw *should* be unstable under reasonable
seed/parameter changes, and it is. Each of the 10 final tickets was checked
for recurrence across 1,000 alternative seeds of the same diversified
selection process — **recurrence rate = 0/1000 for all 10 tickets**
(`results/final_10.csv`, `selection_stability` column). This is the correct
sensitivity result for a null model: the picks are not "attractors" that
keep reappearing, which would have been a red flag suggesting a bug rather
than a real signal.

## 43. Rank reliability (Step 43)

Every one of the 169,911 combinations carries the identical official score
(1/169911); `results/top169911.csv`'s `rank` column is entirely a
pre-registered random tie-break, not a meaningful ordering. **Rank 1 is not
meaningfully "better" than Rank 169,911** — this is stated explicitly here
per the task's own instruction not to over-interpret near-zero (here:
exactly-zero) score differences.

## Limitations

- Network access in this sandboxed session was restricted to GitHub and
  package-registry domains; several preferred/official data sources named
  in the task brief were unreachable (§1). Set-ball and venue data are
  entirely absent as a result.
- Rounds 355-1301 have two-source (A/B) but not three-source (A/B/C)
  cross-validation (source C only covers the most recent ~100 rounds); A
  and B agree exactly everywhere they overlap, including this range.
- Walk-forward retraining cadence is every 25 draws (expanding window),
  not literally every single draw, for compute tractability — disclosed in
  `walk_forward.py`'s docstring; this does not change the "no leakage"
  guarantee, only the granularity of retraining.
- 4- and 5-number co-occurrence analysis (Step 11) was not pursued beyond
  shape features, given expected counts too low to test reliably without
  inviting the exact kind of overfitting-on-sparse-counts this project is
  designed to avoid.
- A dedicated triple-based ML feature (beyond the descriptive Step 10
  analysis) was not built into the walk-forward feature set, for the same
  sparse-count reason; the pair-lift feature was used as the representative
  co-occurrence feature in the ML models instead.
