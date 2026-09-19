# 10-Ticket Strategy Report (Steps 23-25, 33-37)

Full narrative in `reports/final_report.md` §23-25, §33-37. This file
indexes the underlying data.

- Strategy implementations + full-history backtest: `scripts/portfolio_10.py`
  - `results/strategy10_performance.csv` — per-draw result for every strategy
  - `results/strategy10_summary.csv` — aggregate ROI / hit-rate / drawdown per strategy
  - `results/strategy_d_popularity_regression.csv` — naive full-history fit (later shown to be leaky)
  - `results/strategy_d_regression_firsthalf_only.csv` — corrected discovery-only fit
  - `results/strategy_d_oos_secondhalf_backtest.csv` — corrected holdout backtest
- Monte Carlo comparison: `scripts/monte_carlo.py` → `results/monte_carlo_results.csv`, `results/monte_carlo_comparison.csv`
- Final ticket generation: `scripts/final_prediction.py` → `results/final_10.csv`, `results/final_10_coverage.csv`

## Strategy summary (full-history backtest, 1404 draws, real prize data)

| Strategy | Mean max match | Rate ≥3 match | Net profit (¥) | ROI | MC profit percentile |
|---|---|---|---|---|---|
| A Pure random | 2.069 | 0.209 | -2,433,400 | -86.66% | 34.1 |
| B Diversified | 2.080 | 0.209 | -2,433,600 | -86.67% | 34.0 |
| C Coverage-optimized | 2.130 | 0.197 | -2,445,600 | -87.09% | 26.4 |
| D Prize/ROI (naive fit — LEAKY, see below) | 1.985 | 0.199 | -2,115,800 | -75.35% | 87.9 |

## Critical correction

Strategy D's apparent edge was fit and tested on the *same* full-history
data. A proper discovery(first half)/holdout(second half) split shows the
underlying shape predictor (`decade_spread`) does **not** survive FDR
correction when fit on discovery data only (p_fdr = 0.102), and the
resulting OOS backtest on the holdout half is **identical to pure random**
(both strategies: mean match 2.084, profit -¥1,200,500, ROI -85.51%). This
is disclosed prominently rather than keeping the better-looking but leaky
number — see `reports/final_report.md` §40 for the self-audit narrative.

## Final decision

No strategy has a validated edge. **Strategy B (diversified random, overlap
cap = 2)** was selected as the final construction method purely for
portfolio hygiene (avoids near-duplicate tickets), decided before generating
the draw-1405 picks, using the same `SEED + draw_no` formula applied to all
1404 historical backtest draws.
