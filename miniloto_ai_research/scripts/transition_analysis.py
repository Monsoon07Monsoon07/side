"""Step 8: previous-draw dependency (transition) analysis."""
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]


def main():
    df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)
    n = len(df)
    occ = np.zeros((n, 32), dtype=np.int8)
    bonus = np.zeros((n, 32), dtype=np.int8)
    for t, row in df.iterrows():
        for c in MAIN_COLS:
            occ[t, int(row[c])] = 1
        bonus[t, int(row["bonus"])] = 1

    # overlap count between draw t-1 main numbers and draw t main numbers
    overlaps = []
    bonus_carry = []  # did draw t-1's bonus number appear as a main number in draw t?
    for t in range(1, n):
        ov = int((occ[t - 1] & occ[t]).sum())
        overlaps.append(ov)
        prev_bonus_num = np.argmax(bonus[t - 1])
        bonus_carry.append(int(occ[t, prev_bonus_num]))

    overlaps = np.array(overlaps)
    obs_dist = pd.Series(overlaps).value_counts().sort_index()

    # theoretical distribution of overlap between two independent random 5-subsets of 31
    from scipy.stats import hypergeom
    theo = {k: hypergeom.pmf(k, 31, 5, 5) for k in range(6)}
    theo_counts = {k: theo[k] * (n - 1) for k in theo}

    overlap_summary = pd.DataFrame({
        "overlap_count": list(range(6)),
        "observed_n": [int(obs_dist.get(k, 0)) for k in range(6)],
        "observed_rate": [obs_dist.get(k, 0) / (n - 1) for k in range(6)],
        "theoretical_rate": [theo[k] for k in range(6)],
        "theoretical_n": [theo_counts[k] for k in range(6)],
    })
    chi2 = ((overlap_summary["observed_n"] - overlap_summary["theoretical_n"]) ** 2 / overlap_summary["theoretical_n"]).sum()
    from scipy.stats import chi2 as chi2dist
    chi2_p = 1 - chi2dist.cdf(chi2, df=5)

    bonus_carry_rate = np.mean(bonus_carry)
    theo_bonus_carry = 5 / 31
    bt = binomtest(sum(bonus_carry), len(bonus_carry), theo_bonus_carry, alternative="two-sided")

    # per-number transition: P(number appears at t | appeared at t-1) vs P(appears at t | did NOT appear at t-1)
    rows = []
    for num in range(1, 32):
        prev = occ[:-1, num]
        curr = occ[1:, num]
        given_prev1 = curr[prev == 1]
        given_prev0 = curr[prev == 0]
        p1 = given_prev1.mean() if len(given_prev1) else np.nan
        p0 = given_prev0.mean() if len(given_prev0) else np.nan
        # simple 2x2 test
        from scipy.stats import fisher_exact
        table = [[given_prev1.sum(), len(given_prev1) - given_prev1.sum()],
                 [given_prev0.sum(), len(given_prev0) - given_prev0.sum()]]
        _, p_fisher = fisher_exact(table)
        rows.append({
            "number": num, "p_appear_given_prev_appear": p1, "n_prev_appear": len(given_prev1),
            "p_appear_given_prev_absent": p0, "n_prev_absent": len(given_prev0),
            "diff": p1 - p0, "fisher_p_raw": p_fisher,
        })
    trans_df = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(trans_df["fisher_p_raw"], alpha=0.05, method="fdr_bh")
    trans_df["fisher_p_fdr"] = p_adj
    trans_df["sig_fdr"] = reject

    overlap_summary.to_csv(RESULTS / "transition_overlap_statistics.csv", index=False)
    trans_df.to_csv(RESULTS / "transition_statistics.csv", index=False)

    print("Overlap (prev-draw main numbers repeated in next draw) distribution:")
    print(overlap_summary.to_string(index=False))
    print(f"chi2={chi2:.3f} p={chi2_p:.4f}")
    print(f"\nBonus-carry (prev bonus number appears as next main number): observed rate={bonus_carry_rate:.4f}, theoretical={theo_bonus_carry:.4f}, binom p={bt.pvalue:.4f}")
    print(f"\nPer-number transition FDR-significant count: {trans_df['sig_fdr'].sum()} / 31")
    print(trans_df.sort_values("fisher_p_raw").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
