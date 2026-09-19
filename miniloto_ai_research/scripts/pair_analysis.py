"""Step 9: all 465 pairs (31 choose 2) - full-period + period-split stability check."""
import pathlib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import binomtest, hypergeom
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]


def build_occ(df):
    n = len(df)
    occ = np.zeros((n, 32), dtype=np.int8)
    for t, row in df.iterrows():
        for c in MAIN_COLS:
            occ[t, int(row[c])] = 1
    return occ


def analyze(occ, label):
    n = occ.shape[0]
    p_pair = hypergeom.pmf(2, 31, 5, 2)  # P(both of 2 fixed numbers are among the 5 drawn)... use proper formula below
    # Correct: P(specific pair both drawn) = C(29,3)/C(31,5)
    from math import comb
    p_pair = comb(29, 3) / comb(31, 5)
    exp_count = n * p_pair

    rows = []
    for a, b in itertools.combinations(range(1, 32), 2):
        both = int(((occ[:, a] == 1) & (occ[:, b] == 1)).sum())
        a_only = int((occ[:, a] == 1).sum())
        b_only = int((occ[:, b] == 1).sum())
        lift = (both / n) / p_pair if p_pair > 0 else np.nan
        bt = binomtest(both, n, p_pair, alternative="two-sided")
        # odds ratio via 2x2 table: both / a_not_b / b_not_a / neither
        a_and_not_b = a_only - both
        b_and_not_a = b_only - both
        neither = n - both - a_and_not_b - b_and_not_a
        or_num = both * neither
        or_den = a_and_not_b * b_and_not_a
        odds_ratio = or_num / or_den if or_den > 0 else np.inf
        rows.append({
            "num_a": a, "num_b": b, "observed_count": both, "expected_count": exp_count,
            "lift": lift, "odds_ratio": odds_ratio, "p_raw": bt.pvalue, "period": label,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)
    occ_full = build_occ(df)
    full = analyze(occ_full, "full")

    for method, name in [("bonferroni", "bonf"), ("holm", "holm"), ("fdr_bh", "fdr")]:
        reject, p_adj, _, _ = multipletests(full["p_raw"], alpha=0.05, method=method)
        full[f"p_{name}"] = p_adj
        full[f"sig_{name}"] = reject

    # period split stability: first half vs second half, and 4 quartiles, and rolling last-300
    mid = len(df) // 2
    first_half = analyze(occ_full[:mid], "first_half")
    second_half = analyze(occ_full[mid:], "second_half")

    quarters = np.array_split(np.arange(len(df)), 4)
    q_frames = []
    for i, idx in enumerate(quarters):
        qdf = analyze(occ_full[idx], f"q{i+1}")
        q_frames.append(qdf)

    # identify FDR-significant pairs in full period, then check sign consistency across halves
    sig_pairs = full[full["sig_fdr"]][["num_a", "num_b"]]
    stability_rows = []
    for _, r in sig_pairs.iterrows():
        a, b = int(r["num_a"]), int(r["num_b"])
        fh_lift = first_half[(first_half.num_a == a) & (first_half.num_b == b)]["lift"].values[0]
        sh_lift = second_half[(second_half.num_a == a) & (second_half.num_b == b)]["lift"].values[0]
        full_lift = full[(full.num_a == a) & (full.num_b == b)]["lift"].values[0]
        stability_rows.append({
            "num_a": a, "num_b": b, "full_lift": full_lift,
            "first_half_lift": fh_lift, "second_half_lift": sh_lift,
            "same_direction_both_halves": (fh_lift > 1) == (sh_lift > 1) == (full_lift > 1),
        })
    stability_df = pd.DataFrame(stability_rows)

    full.to_csv(RESULTS / "pair_statistics.csv", index=False)
    pd.concat([first_half, second_half] + q_frames, ignore_index=True).to_csv(RESULTS / "pair_statistics_by_period.csv", index=False)
    stability_df.to_csv(RESULTS / "pair_stability_check.csv", index=False)

    print(f"Total pairs: {len(full)} (expect 465)")
    print(f"Uncorrected p<0.05: {(full['p_raw']<0.05).sum()}")
    print(f"Bonferroni-significant: {full['sig_bonf'].sum()}")
    print(f"Holm-significant: {full['sig_holm'].sum()}")
    print(f"BH-FDR-significant: {full['sig_fdr'].sum()}")
    if len(stability_df):
        print(f"Of FDR-significant pairs, consistent-direction-across-both-halves: {stability_df['same_direction_both_halves'].sum()} / {len(stability_df)}")
    print(full.sort_values("p_raw").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
