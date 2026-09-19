# Statistical Report (Steps 5-15)

Full narrative and numbers are in `reports/final_report.md` §5-15. This file
indexes the underlying data for quick access.

| Analysis | Script | Output |
|---|---|---|
| Single-number (main/bonus/main+bonus), multi-testing corrected | `scripts/statistical_tests.py` | `results/number_statistics.csv` |
| Rolling frequency + EWMA (all windows/half-lives) | `scripts/rolling_analysis.py` | `results/rolling_statistics.csv` |
| Gap/overdue + gambler's-fallacy regression | `scripts/gap_analysis.py` | `results/gap_statistics.csv`, `results/gap_reappearance_curve.csv`, `results/gap_gamblers_fallacy_test.csv` |
| Previous-draw transitions | `scripts/transition_analysis.py` | `results/transition_overlap_statistics.csv`, `results/transition_statistics.csv` |
| All 465 pairs, multi-testing corrected, period-stability | `scripts/pair_analysis.py` | `results/pair_statistics.csv`, `results/pair_statistics_by_period.csv`, `results/pair_stability_check.csv` |
| All 4,495 triples, Bayesian shrinkage | `scripts/triple_analysis.py` | `results/triple_statistics.csv` |
| Combination shape vs. theoretical (169,911-combo) distribution | `scripts/structural_analysis.py` | `results/structural_shape_features.csv`, `results/structural_shape_summary.csv` |
| Autocorrelation / Ljung-Box / runs test | `scripts/structural_analysis.py` | `results/timeseries_autocorrelation.csv` |
| CUSUM structural change + permutation test | `scripts/structural_analysis.py` | `results/structural_statistics.csv` |
| Weekday/date distribution | `scripts/structural_analysis.py` | `results/date_weekday_distribution.csv` |

**Bottom line: after Bonferroni/Holm/BH-FDR correction, zero individual
numbers, zero pairs, zero triples, zero per-number transitions, and zero
autocorrelations remain statistically significant.** Draws are consistent
with the exact fair-draw (Hypergeometric/uniform) model at every level
tested. See `reports/final_report.md` for full numeric detail and
interpretation of each borderline (uncorrected) result.
