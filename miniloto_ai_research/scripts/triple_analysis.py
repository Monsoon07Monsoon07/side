"""Step 10: all 4495 triples (31 choose 3), with Bayesian (Beta-Binomial) shrinkage
given the low expected counts (~7.4 per triple over 1404 draws)."""
import pathlib
import itertools
from math import comb
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

    counts = {}
    for t in itertools.combinations(range(1, 32), 3):
        counts[t] = 0
    for _, row in df.iterrows():
        nums = sorted(int(row[c]) for c in MAIN_COLS)
        for trip in itertools.combinations(nums, 3):
            counts[trip] += 1

    p_triple = comb(28, 2) / comb(31, 5)
    exp_count = n * p_triple

    rows = []
    for trip, c in counts.items():
        bt = binomtest(c, n, p_triple, alternative="two-sided")
        rows.append({
            "num_a": trip[0], "num_b": trip[1], "num_c": trip[2],
            "observed_count": c, "expected_count": exp_count,
            "lift": (c / n) / p_triple, "p_raw": bt.pvalue,
        })
    out = pd.DataFrame(rows)
    assert len(out) == 4495, len(out)

    for method, name in [("bonferroni", "bonf"), ("holm", "holm"), ("fdr_bh", "fdr")]:
        reject, p_adj, _, _ = multipletests(out["p_raw"], alpha=0.05, method=method)
        out[f"p_{name}"] = p_adj
        out[f"sig_{name}"] = reject

    # Bayesian shrinkage: Beta-Binomial with prior matching the theoretical rate,
    # prior strength chosen (not tuned on outcome) = equivalent to 50 pseudo-draws.
    prior_strength = 50
    alpha0 = p_triple * prior_strength
    beta0 = (1 - p_triple) * prior_strength
    out["shrunk_rate"] = (out["observed_count"] + alpha0) / (n + alpha0 + beta0)
    out["shrunk_lift"] = out["shrunk_rate"] / p_triple

    out.to_csv(RESULTS / "triple_statistics.csv", index=False)

    print(f"Total triples: {len(out)} (expect 4495)")
    print(f"Theoretical p per triple: {p_triple:.6f}, expected count over {n} draws: {exp_count:.3f}")
    print(f"Uncorrected p<0.05: {(out['p_raw']<0.05).sum()} (chance expectation ~{0.05*4495:.0f})")
    print(f"Bonferroni-significant: {out['sig_bonf'].sum()}")
    print(f"Holm-significant: {out['sig_holm'].sum()}")
    print(f"BH-FDR-significant: {out['sig_fdr'].sum()}")
    print(out.sort_values("p_raw").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
