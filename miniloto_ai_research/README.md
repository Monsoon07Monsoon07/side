# Mini Loto (ミニロト) AI Research — Round 1405 Final Picks

Full statistical/ML/backtest research project on Japan's Mini Loto lottery
(rounds 1–1404, 1999-04-13 to 2026-09-15), built to answer one question
honestly: **does the public draw history contain any statistically
reliable, reproducible signal that would let an AI beat random ticket
selection for round 1405?**

## Result

**No.** Zero of 18 tested models (10 statistical baselines + 8 ML models)
beat the random/uniform baseline out-of-sample after multiple-testing
correction, across 45 time-ordered walk-forward folds and 34,224
predictions. A secondary "avoid popular-looking combinations to reduce
prize-sharing" strategy looked promising on a full-history fit but **failed
a proper discovery/holdout re-test** (caught during self-audit — see
`reports/final_report.md` §40). Negative-control (pure noise) features
ranked as "important" as real features in model diagnostics, confirming
there is nothing for the models to find. See `reports/final_report.md` for
the complete 43-point walkthrough.

**AI FINAL PICKS — DRAW 1405** (`results/final_10.csv`), 10 tickets / ¥2,000,
selected by a diversified-random construction (chosen for portfolio hygiene,
not predictive edge — no strategy validated one):

```
1. 02 04 06 19 30      6. 07 13 15 18 27
2. 02 09 18 30 31      7. 01 06 15 24 26
3. 12 16 18 19 22      8. 03 07 17 18 21
4. 03 05 10 17 29      9. 18 19 25 28 30
5. 01 02 12 17 30     10. 02 12 20 28 29
```

## Reproducing this project

```
pip install pandas numpy scipy scikit-learn statsmodels xgboost lightgbm pyarrow

python3 scripts/download_data.py        # fetch 3 independent GitHub-hosted sources
python3 scripts/audit_data.py           # cross-validate + completeness audit (must PASS)
python3 scripts/theoretical_probabilities.py
python3 scripts/combination_enum.py     # all 169,911 combinations
python3 scripts/statistical_tests.py    # Step 5
python3 scripts/rolling_analysis.py     # Step 6
python3 scripts/gap_analysis.py         # Step 7
python3 scripts/transition_analysis.py  # Step 8
python3 scripts/pair_analysis.py        # Step 9
python3 scripts/triple_analysis.py      # Step 10
python3 scripts/structural_analysis.py  # Steps 12-15
python3 scripts/feature_engineering.py  # causal feature table
python3 scripts/walk_forward.py         # Steps 18-21 (~9 min)
python3 scripts/combination_rank.py     # Step 22
python3 scripts/portfolio_10.py         # Steps 23-25 (~3 min)
python3 scripts/monte_carlo.py          # Step 26 (~2 min)
python3 scripts/final_prediction.py     # Steps 33-37, 42
python3 scripts/audit_final.py          # Step 41 independent recompute
```

`SEED = 20260920` is fixed everywhere randomness is used. All results are
fully reproducible from these scripts and the raw data in `data/`.

## Directory structure

```
data/       raw + cross-validated draw history, all 169,911 combinations
scripts/    every analysis/model/backtest script (see reproduction steps above)
results/    every intermediate and final CSV/JSON/Parquet output
reports/    data_audit.md, statistical_report.md, walk_forward_report.md,
            strategy10_report.md, final_report.md (the full writeup),
            setball_venue_note.md (data-access limitation disclosure)
```

## Known limitations (see `reports/final_report.md` "Limitations" for full detail)

- This sandboxed session's network access was restricted to GitHub and
  package registries; several preferred official/semi-official data sources
  named in the task brief were unreachable. Two independent full-history
  GitHub mirrors were used instead and cross-validated with zero
  discrepancies across all 1404 rounds.
- Set-ball (A-J) and drawing-venue history could not be sourced at all and
  are excluded from every model (not fabricated).
- Walk-forward retraining occurs every 25 draws (expanding window), not
  every single draw, for compute tractability — disclosed in
  `scripts/walk_forward.py`.
