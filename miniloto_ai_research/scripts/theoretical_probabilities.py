"""Step 3: theoretical (fair-draw) probability model for Mini Loto."""
import math
import json
import pathlib
from scipy.stats import hypergeom

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N = 31   # pool size
K = 5    # numbers drawn per ticket / per draw main numbers
TICKET_K = 5


def main():
    total_combinations = math.comb(N, K)
    assert total_combinations == 169911, total_combinations

    # P(a specific number is among the 5 main numbers drawn) = K/N
    p_number_main = K / N
    # P(a specific number is the bonus number | not already drawn as main) modeled as
    # uniform over the 26 remaining numbers after 5 are drawn:
    p_bonus_given_number = (1 / (N - K))
    p_number_bonus_overall = (1 - p_number_main) * p_bonus_given_number
    p_number_main_or_bonus = p_number_main + p_number_bonus_overall

    # Match-count distribution for a fixed ticket of 5 numbers against the
    # official draw of 5 main numbers, using the hypergeometric distribution:
    # X = number of matches, X ~ Hypergeom(N, K, K)
    match_dist = {}
    for x in range(0, K + 1):
        match_dist[x] = hypergeom.pmf(x, N, K, TICKET_K)

    # Prize-tier probabilities (mini loto official rules):
    # 1st: 5 match (all 5 main numbers)
    # 2nd: 4 match main + bonus matches the 5th number the ticket holds... actually
    # official mini loto prize structure:
    #  1st: match all 5 main numbers
    #  2nd: match 4 of 5 main numbers AND match the bonus number
    #  3rd: match 4 of 5 main numbers (no bonus match)
    #  4th: match 3 of 5 main numbers
    p_1st = match_dist[5]
    p_4main = match_dist[4]
    # given exactly 4 main matches, the ticket's 5th (non-matching) number equals bonus
    # with probability 1/26 (bonus is uniformly one of the 26 numbers not drawn as main)
    p_2nd = p_4main * (1 / (N - K))
    p_3rd = p_4main * (1 - 1 / (N - K))
    p_4th = match_dist[3]

    out = {
        "N_pool": N,
        "K_drawn": K,
        "total_combinations_C(31,5)": total_combinations,
        "p_number_is_main": p_number_main,
        "p_number_is_bonus_given_not_main": p_bonus_given_number,
        "p_number_is_bonus_overall": p_number_bonus_overall,
        "p_number_is_main_or_bonus": p_number_main_or_bonus,
        "match_count_distribution_ticket_vs_draw": match_dist,
        "prize_tier_probabilities": {
            "1st_5match": p_1st,
            "2nd_4match_plus_bonus": p_2nd,
            "3rd_4match_no_bonus": p_3rd,
            "4th_3match": p_4th,
        },
        "expected_tickets_per_1st_prize": 1 / p_1st,
    }

    (RESULTS / "theoretical_probabilities.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
