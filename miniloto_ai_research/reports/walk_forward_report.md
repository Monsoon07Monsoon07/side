# Walk-Forward Report (Steps 18-22, 26-32)

Full narrative in `reports/final_report.md` §18-22, §26-32. This file
indexes the underlying data.

- Feature construction (strictly causal): `scripts/feature_engineering.py` → `results/feature_table_cache.parquet`
- Model definitions (10 baselines + 8 ML models): `scripts/models.py`
- Walk-forward loop (45 expanding-window folds, retrain every 25 draws from draw 300): `scripts/walk_forward.py`
  - `results/walk_forward_predictions.csv` / `.parquet` — every one of the 34,224 out-of-sample (draw, number) predictions from every model
  - `results/walk_forward_window_choices.csv` — which rolling window / EWMA half-life was picked inside each training fold (never from the held-out block)
  - `results/model_performance.csv` — Brier score, log loss, AUC, mean matches, match-count distribution, and Wilcoxon-vs-random FDR-corrected significance for every model
- Full-history combination scoring: `scripts/combination_rank.py` → `results/top169911.csv`, `results/top100.csv`
- Negative controls: `results/negative_control_feature_importance.csv`

## Headline numbers

- 18 models tested (A_random baseline + 9 other baselines + 8 ML models).
- Best (lowest) Brier score: **A_random, 0.135276** — the theoretical-optimum uniform baseline. No model beats it.
- AUC range across all models: 0.4965 – 0.5095 (chance = 0.50).
- Models with FDR-significant OOS Brier improvement over random: **0 / 17** (excluding random itself).
- Negative-control noise features ranked #2 and #5 of 15 by Random Forest importance — above several real features, confirming the model cannot separate signal from noise (because there is no signal to separate).

## Adoption outcome

Per the pre-registered rule (OOS beat random, FDR-significant), the final
validated ensemble is **empty**. `combination_rank.py` therefore scores all
169,911 combinations identically (uniform 1/169911) — the correct output of
the rule, not a bug.
