"""Steps 33-37, 42-43: assemble the final DRAW 1405 10-ticket output.

Given walk_forward.py's result (ZERO models beat random OOS after FDR
correction) and the corrected Strategy D check (portfolio_10.py's naive
full-history fit showed an apparent edge that VANISHED under a proper
first-half-fit / second-half-test causal split -- see
strategy_d_oos_secondhalf_backtest.csv), the pre-registered adoption rule
(Step 31) leaves NO validated model or strategy. The final construction
method is therefore Strategy B (diversified random, seed = SEED + draw_no,
identical formula used for all 1404 historical backtest draws, now applied
un-modified to draw_no=1405) -- chosen for ordinary portfolio hygiene
(avoid buying near-duplicate tickets) only, not for any claimed predictive
or economic edge.

selection_stability here is redefined honestly for a null-result project:
it is the empirical probability that a given final ticket recurs across
1,000 alternative "reasonable" seeds -- i.e. evidence the ticket is NOT a
special attractor, which is exactly what a correctly-functioning random
draw should show.
"""
import pathlib
import itertools
import numpy as np
import pandas as pd

SEED = 20260920
DRAW_NO = 1405
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from portfolio_10 import strategy_diversified_top10


def main():
    rng = np.random.default_rng(SEED + DRAW_NO)
    tickets = [tuple(sorted(t)) for t in strategy_diversified_top10(rng)]

    top169911 = pd.read_csv(RESULTS / "top169911.csv")
    top169911_idx = {tuple(sorted([r.n1, r.n2, r.n3, r.n4, r.n5])): (r.rank, r.official_score) for r in top169911.itertuples()}

    # selection stability: recurrence rate across 1000 alternative reasonable seeds
    n_alt = 1000
    recur_counts = {t: 0 for t in tickets}
    for k in range(1, n_alt + 1):
        alt_rng = np.random.default_rng(SEED + DRAW_NO + k * 97)
        alt_tickets = set(tuple(sorted(x)) for x in strategy_diversified_top10(alt_rng))
        for t in tickets:
            if t in alt_tickets:
                recur_counts[t] += 1

    rows = []
    for i, t in enumerate(tickets, 1):
        rank, score = top169911_idx.get(t, (np.nan, 1 / 169911))
        rows.append({
            "ticket_no": i, "n1": t[0], "n2": t[1], "n3": t[2], "n4": t[3], "n5": t[4],
            "combination_rank": int(rank) if not np.isnan(rank) else None,
            "model_score": score,
            "relative_score": 1.0,  # uniform under the null-adopted model; see report for interpretation
            "selection_stability": recur_counts[t] / n_alt,
            "strategy": "B_diversified_random (no validated predictive/economic edge; chosen for portfolio hygiene only)",
        })
    final_df = pd.DataFrame(rows)
    final_df.to_csv(RESULTS / "final_10.csv", index=False)

    print("=== AI FINAL PICKS -- DRAW 1405 ===")
    print(final_df.to_string(index=False))

    # coverage summary
    all_nums = [n for t in tickets for n in t]
    num_cov = pd.Series(all_nums).value_counts().sort_index()
    pairs_cov = set()
    triples_cov = set()
    for t in tickets:
        for p in itertools.combinations(t, 2):
            pairs_cov.add(p)
        for tr in itertools.combinations(t, 3):
            triples_cov.add(tr)
    print(f"\nDistinct numbers used: {len(num_cov)}/31")
    print(f"Distinct pairs covered: {len(pairs_cov)}/{10*10}")
    print(f"Distinct triples covered: {len(triples_cov)}/{10*10}")
    print(f"\nSelection stability (recurrence over 1000 alt seeds), mean={final_df['selection_stability'].mean():.4f}, "
          f"expected under pure chance if fully non-special ~ {10/169911:.6f} per exact-match baseline (stability measured here is for the whole diversified 10-ticket regeneration process, not single-ticket draws, so a higher baseline is expected)")

    cov_summary = pd.DataFrame({"number": num_cov.index, "count_in_final10": num_cov.values})
    cov_summary.to_csv(RESULTS / "final_10_coverage.csv", index=False)


if __name__ == "__main__":
    main()
