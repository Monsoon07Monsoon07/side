"""
Step 26: Monte Carlo comparison of the AI's chosen 10-ticket strategy against
a large population of purely-random 10-ticket-per-draw players, replayed
over the same real 1404-draw history (real prize amounts used for payout).

Because ticket numbers are chosen without any knowledge of the future draw,
a uniformly-random ticket's match count against ANY draw (real or
hypothetical) is exactly Hypergeometric(M=31, n=5, N=5) -- this is a exact
property of the fair-draw model confirmed in theoretical_probabilities.py,
not an approximation, so it is used directly to make 100,000-player
simulation tractable without looping over explicit number sets.
"""
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

SEED = 20260920
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

N_SIMS = 100000
N_TICKETS = 10
TICKET_COST = 200
TOTAL_COST_PER_DRAW = TICKET_COST * N_TICKETS


def main():
    clean_df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)
    n_draws = len(clean_df)

    r1 = clean_df["rank1_amount"].fillna(0).values
    r2 = clean_df["rank2_amount"].fillna(0).values
    r3 = clean_df["rank3_amount"].fillna(0).values
    r4 = clean_df["rank4_amount"].fillna(0).values

    rng = np.random.default_rng(SEED)

    sim_summary = np.zeros((N_SIMS, 6))  # mean_max_match, n_3plus, n_4plus, n_5, total_profit, max_drawdown
    CHUNK = 2000
    for chunk_start in range(0, N_SIMS, CHUNK):
        chunk_n = min(CHUNK, N_SIMS - chunk_start)
        # matches shape: (chunk_n, n_draws, N_TICKETS)
        matches = hypergeom.rvs(31, 5, 5, size=(chunk_n, n_draws, N_TICKETS), random_state=rng)
        bonus_roll = rng.random((chunk_n, n_draws, N_TICKETS))
        is5 = matches == 5
        is4 = matches == 4
        is3 = matches == 3
        bonus_match = is4 & (bonus_roll < (1 / 26))
        is4_only = is4 & (~bonus_match)

        payout = (
            is5 * r1[None, :, None]
            + bonus_match * r2[None, :, None]
            + is4_only * r3[None, :, None]
            + is3 * r4[None, :, None]
        )
        payout_per_draw = payout.sum(axis=2)  # (chunk_n, n_draws)
        profit_per_draw = payout_per_draw - TOTAL_COST_PER_DRAW
        cum_profit = np.cumsum(profit_per_draw, axis=1)
        running_max = np.maximum.accumulate(cum_profit, axis=1)
        drawdown = (cum_profit - running_max).min(axis=1)

        max_match_per_draw = matches.max(axis=2)
        sim_summary[chunk_start:chunk_start + chunk_n, 0] = max_match_per_draw.mean(axis=1)
        sim_summary[chunk_start:chunk_start + chunk_n, 1] = (matches >= 3).sum(axis=(1, 2)) / n_draws
        sim_summary[chunk_start:chunk_start + chunk_n, 2] = (matches >= 4).sum(axis=(1, 2)) / n_draws
        sim_summary[chunk_start:chunk_start + chunk_n, 3] = (matches == 5).sum(axis=(1, 2))
        sim_summary[chunk_start:chunk_start + chunk_n, 4] = profit_per_draw.sum(axis=1)
        sim_summary[chunk_start:chunk_start + chunk_n, 5] = drawdown
        print(f"MC chunk {chunk_start+chunk_n}/{N_SIMS} done")

    mc_df = pd.DataFrame(sim_summary, columns=["mean_max_match", "rate_3plus", "rate_4plus", "n_5_hits", "total_profit", "max_drawdown"])
    mc_df.to_csv(RESULTS / "monte_carlo_results.csv", index=False)

    strat_summary = pd.read_csv(RESULTS / "strategy10_summary.csv").set_index("strategy")

    def percentile_of(value, dist, higher_is_better=True):
        if higher_is_better:
            return float((dist <= value).mean() * 100)
        else:
            return float((dist >= value).mean() * 100)

    comparison_rows = []
    for strat in strat_summary.index:
        row = strat_summary.loc[strat]
        comparison_rows.append({
            "strategy": strat,
            "ai_mean_max_match": row["mean_max_match"],
            "mc_mean_max_match_pctile": percentile_of(row["mean_max_match"], mc_df["mean_max_match"]),
            "ai_rate_3plus": row["rate_3plus_per_draw"],
            "mc_rate_3plus_pctile": percentile_of(row["rate_3plus_per_draw"], mc_df["rate_3plus"]),
            "ai_total_profit": row["net_profit"],
            "mc_total_profit_pctile": percentile_of(row["net_profit"], mc_df["total_profit"]),
            "ai_max_drawdown": row["max_drawdown"],
            "mc_max_drawdown_pctile": percentile_of(row["max_drawdown"], mc_df["max_drawdown"], higher_is_better=True),
        })
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(RESULTS / "monte_carlo_comparison.csv", index=False)

    print(f"\n=== Monte Carlo ({N_SIMS} simulated random 10-ticket players, replayed over real {n_draws}-draw history) ===")
    print(mc_df.describe().to_string())
    print("\n=== AI strategies vs random-player Monte Carlo distribution (percentile) ===")
    print(comp_df.to_string(index=False))
    print("\nNote: a percentile near 50 means the AI strategy performed like an ordinary random player -- i.e. NO demonstrated edge. "
          "A percentile has no meaning for skill unless it is BOTH high AND stable/replicable; a single-history percentile can look impressive by chance.")


if __name__ == "__main__":
    main()
