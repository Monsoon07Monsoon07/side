"""
Steps 23-25: 10-ticket (2,000 yen) portfolio strategies, backtested over the
full 1404-draw history, plus a real-data (not assumed/fabricated) check of
whether "typical-looking" combinations attract more co-winners (Strategy D).

Because walk_forward.py found NO model with a validated out-of-sample edge,
strategies A-C are score-agnostic constructions drawn from a reproducible
seeded uniform pool (SEED + draw_no as sub-seed) -- this is the only
defensible action once no predictive edge exists, and what's actually being
compared here is portfolio construction (variance / coverage), not skill.
Strategy D additionally uses REAL historical winner-count data (no assumed
popularity model) to test the one leftover legitimate economic angle:
"unpopular-looking" combinations may split the pari-mutuel 1st-3rd prizes
among fewer people.
"""
import pathlib
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

SEED = 20260920
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]
TICKET_COST = 200
N_TICKETS = 10
TOTAL_COST = TICKET_COST * N_TICKETS


def shape_of(nums):
    nums = sorted(nums)
    s = sum(nums)
    diffs = [nums[i + 1] - nums[i] for i in range(4)]
    return {
        "sum": s, "abs_sum_dev": abs(s - 80),
        "consecutive_pairs": sum(1 for d in diffs if d == 1),
        "low_count": sum(1 for x in nums if x <= 15),
        "birthday_count": sum(1 for x in nums if x <= 12),  # months 1-12 proxy for "human-picked" bias
        "has_all_low12": all(x <= 12 for x in nums),
        "decade_spread": len(set(x // 10 for x in nums)),
    }


def strategy_d_regression(clean_df):
    """Real-data check: does a combination's 'typical/human-picked' shape predict
    FEWER 1st-prize winners relative to overall ticket volume that draw (proxied
    by 4th-prize winners, whose win probability is fixed, so 4th-prize winner
    count tracks total tickets sold)?"""
    df = clean_df.dropna(subset=["rank1_winners", "rank4_winners"]).copy()
    df = df[df["rank4_winners"] > 0]
    rows = []
    for _, r in df.iterrows():
        nums = [r[c] for c in MAIN_COLS]
        f = shape_of(nums)
        f["popularity_proxy"] = r["rank1_winners"] / r["rank4_winners"]  # higher = combo was "popular" (more co-winners) relative to sales volume
        rows.append(f)
    feat = pd.DataFrame(rows)
    X = sm.add_constant(feat[["abs_sum_dev", "consecutive_pairs", "low_count", "birthday_count", "decade_spread"]])
    y = feat["popularity_proxy"]
    model = sm.OLS(y, X).fit()
    coefs = model.params.drop("const")
    pvals = model.pvalues.drop("const")
    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    reg_summary = pd.DataFrame({
        "feature": coefs.index, "coef": coefs.values, "p_raw": pvals.values,
        "p_fdr": p_adj, "sig_fdr": reject,
    })
    return reg_summary, model


def gen_uniform_ticket(rng):
    nums = rng.choice(np.arange(1, 32), size=5, replace=False)
    return tuple(sorted(nums.tolist()))


def strategy_pure_top10(rng):
    tickets = set()
    while len(tickets) < N_TICKETS:
        tickets.add(gen_uniform_ticket(rng))
    return list(tickets)


def strategy_diversified_top10(rng, max_pair_overlap=2):
    tickets = []
    attempts = 0
    while len(tickets) < N_TICKETS and attempts < 5000:
        cand = gen_uniform_ticket(rng)
        attempts += 1
        ok = True
        for t in tickets:
            if len(set(cand) & set(t)) > max_pair_overlap:
                ok = False
                break
        if ok and cand not in tickets:
            tickets.append(cand)
    while len(tickets) < N_TICKETS:
        cand = gen_uniform_ticket(rng)
        if cand not in tickets:
            tickets.append(cand)
    return tickets


def strategy_coverage_optimized(rng):
    tickets = []
    used_counts = np.zeros(32)
    pool_size = 400
    pool = set()
    while len(pool) < pool_size:
        pool.add(gen_uniform_ticket(rng))
    pool = list(pool)
    for _ in range(N_TICKETS):
        best, best_score = None, -1
        for cand in pool:
            if cand in tickets:
                continue
            score = sum(1.0 / (1 + used_counts[n]) for n in cand)
            if score > best_score:
                best_score, best = score, cand
        tickets.append(best)
        for n in best:
            used_counts[n] += 1
    return tickets


def strategy_prize_optimized(rng, reg_summary, pool_size=2000):
    """Rank a random pool by predicted popularity_proxy (lower predicted = fewer
    expected co-winners = higher expected value per win) using the real-data
    regression coefficients, then take the 10 lowest-predicted-popularity
    tickets. Falls back to strategy A behavior if no coefficient survives FDR
    (i.e., no real signal -- must not invent one)."""
    sig = reg_summary[reg_summary["sig_fdr"]]
    if len(sig) == 0:
        return strategy_pure_top10(rng), False
    pool = set()
    while len(pool) < pool_size:
        pool.add(gen_uniform_ticket(rng))
    scored = []
    for cand in pool:
        f = shape_of(cand)
        pred = sum(f[feat] * coef for feat, coef in zip(sig["feature"], sig["coef"]))
        scored.append((pred, cand))
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:N_TICKETS]], True


def score_tickets_against_draw(tickets, actual_main, actual_bonus, prize_row):
    results = []
    actual_set = set(actual_main)
    for tk in tickets:
        matches = len(set(tk) & actual_set)
        bonus_match = actual_bonus in tk
        payout = 0
        tier = None
        if matches == 5:
            tier = 1; payout = prize_row.get("rank1_amount", np.nan)
        elif matches == 4 and bonus_match:
            tier = 2; payout = prize_row.get("rank2_amount", np.nan)
        elif matches == 4:
            tier = 3; payout = prize_row.get("rank3_amount", np.nan)
        elif matches == 3:
            tier = 4; payout = prize_row.get("rank4_amount", np.nan)
        results.append({"matches": matches, "bonus_match": bonus_match, "tier": tier, "payout": payout})
    return results


def backtest_strategy(strategy_fn, clean_df, label):
    rows = []
    for _, r in clean_df.iterrows():
        draw_no = int(r["draw_no"])
        rng = np.random.default_rng(SEED + draw_no)  # reproducible, pre-registered per-draw seed
        tickets = strategy_fn(rng)
        actual_main = [r[c] for c in MAIN_COLS]
        res = score_tickets_against_draw(tickets, actual_main, r["bonus"], r.to_dict())
        max_match = max(x["matches"] for x in res)
        n_3plus = sum(1 for x in res if x["matches"] >= 3)
        n_4plus = sum(1 for x in res if x["matches"] >= 4)
        n_5 = sum(1 for x in res if x["matches"] == 5)
        payout = sum(x["payout"] for x in res if not np.isnan(x["payout"] if x["payout"] is not None else np.nan)) if any(x["tier"] for x in res) else 0
        payout = sum((x["payout"] or 0) for x in res)
        rows.append({
            "draw_no": draw_no, "strategy": label, "max_match": max_match,
            "n_3plus": n_3plus, "n_4plus": n_4plus, "n_5": n_5,
            "cost": TOTAL_COST, "payout": payout, "profit": payout - TOTAL_COST,
        })
    return pd.DataFrame(rows)


def main():
    clean_df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)

    reg_summary, reg_model = strategy_d_regression(clean_df)
    reg_summary.to_csv(RESULTS / "strategy_d_popularity_regression.csv", index=False)
    print("=== Strategy D real-data regression: shape features -> relative popularity (rank1_winners/rank4_winners) ===")
    print(reg_summary.to_string(index=False))
    print(f"Any FDR-significant shape predictor of popularity: {reg_summary['sig_fdr'].any()}")

    strategies = {
        "A_pure_random_top10": lambda rng: strategy_pure_top10(rng),
        "B_diversified_top10": lambda rng: strategy_diversified_top10(rng),
        "C_coverage_optimized": lambda rng: strategy_coverage_optimized(rng),
        "D_prize_roi_optimized": lambda rng: strategy_prize_optimized(rng, reg_summary)[0],
    }

    all_bt = []
    for label, fn in strategies.items():
        bt = backtest_strategy(fn, clean_df, label)
        all_bt.append(bt)
        print(f"\n{label}: mean_max_match={bt['max_match'].mean():.3f}, "
              f"mean_n3plus={bt['n_3plus'].mean():.3f}, total_profit={bt['profit'].sum():.0f}, "
              f"ROI={bt['profit'].sum()/bt['cost'].sum()*100:.2f}%, "
              f"max_drawdown={ (bt['profit'].cumsum() - bt['profit'].cumsum().cummax()).min():.0f}")

    bt_all = pd.concat(all_bt, ignore_index=True)
    bt_all.to_csv(RESULTS / "strategy10_performance.csv", index=False)

    summary_rows = []
    for label, g in bt_all.groupby("strategy"):
        cum_profit = g.sort_values("draw_no")["profit"].cumsum()
        summary_rows.append({
            "strategy": label, "n_draws": len(g),
            "mean_max_match": g["max_match"].mean(),
            "rate_3plus_per_draw": g["n_3plus"].mean(),
            "rate_4plus_per_draw": g["n_4plus"].mean(),
            "n_5_hits_total": g["n_5"].sum(),
            "total_cost": g["cost"].sum(), "total_payout": g["payout"].sum(),
            "net_profit": g["payout"].sum() - g["cost"].sum(),
            "roi_pct": (g["payout"].sum() - g["cost"].sum()) / g["cost"].sum() * 100,
            "max_drawdown": (cum_profit - cum_profit.cummax()).min(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS / "strategy10_summary.csv", index=False)
    print("\n=== Strategy summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
