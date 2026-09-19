"""Step 5: single-number full-period analysis (main numbers, bonus, main+bonus)."""
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_POOL = 31
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]


def wilson_ci(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = x / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return ((center - half) / denom, (center + half) / denom)


def main():
    df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv")
    n_draws = len(df)

    main_counts = {i: 0 for i in range(1, N_POOL + 1)}
    bonus_counts = {i: 0 for i in range(1, N_POOL + 1)}
    for _, row in df.iterrows():
        for c in MAIN_COLS:
            main_counts[int(row[c])] += 1
        bonus_counts[int(row["bonus"])] += 1

    p_main = 5 / N_POOL
    p_bonus = (1 - p_main) * (1 / (N_POOL - 5))  # overall unconditional prob a fixed number is that round's bonus
    # simpler: bonus is drawn uniformly among the 26 non-main numbers each round,
    # but from an a-priori (unconditional) perspective each of the 31 numbers has
    # equal 1/31 chance of being that round's single bonus number. Use both.
    p_bonus_uniform = 1 / N_POOL

    rows = []
    for num in range(1, N_POOL + 1):
        x = main_counts[num]
        exp = n_draws * p_main
        se = np.sqrt(n_draws * p_main * (1 - p_main))
        z = (x - exp) / se
        bt = binomtest(x, n_draws, p_main, alternative="two-sided")
        ci_lo, ci_hi = wilson_ci(x, n_draws)

        xb = bonus_counts[num]
        expb = n_draws * p_bonus_uniform
        seb = np.sqrt(n_draws * p_bonus_uniform * (1 - p_bonus_uniform))
        zb = (xb - expb) / seb
        btb = binomtest(xb, n_draws, p_bonus_uniform, alternative="two-sided")

        xmb = x + xb
        expmb = exp + expb
        p_mb = p_main + p_bonus_uniform
        semb = np.sqrt(n_draws * p_mb * (1 - p_mb))
        zmb = (xmb - expmb) / semb
        btmb = binomtest(xmb, n_draws, p_mb, alternative="two-sided")

        rows.append({
            "number": num,
            "main_count": x, "main_rate": x / n_draws, "main_expected": exp,
            "main_std_resid": z, "main_z": z, "main_p_raw": bt.pvalue,
            "main_ci_lo": ci_lo, "main_ci_hi": ci_hi,
            "bonus_count": xb, "bonus_rate": xb / n_draws, "bonus_expected": expb,
            "bonus_z": zb, "bonus_p_raw": btb.pvalue,
            "main_plus_bonus_count": xmb, "main_plus_bonus_rate": xmb / n_draws,
            "main_plus_bonus_expected": expmb, "main_plus_bonus_z": zmb,
            "main_plus_bonus_p_raw": btmb.pvalue,
        })

    out = pd.DataFrame(rows)

    for col in ["main_p_raw", "bonus_p_raw", "main_plus_bonus_p_raw"]:
        base = col.replace("_p_raw", "")
        for method, name in [("bonferroni", "bonf"), ("holm", "holm"), ("fdr_bh", "fdr")]:
            reject, p_adj, _, _ = multipletests(out[col], alpha=0.05, method=method)
            out[f"{base}_p_{name}"] = p_adj
            out[f"{base}_sig_{name}"] = reject

    out.to_csv(RESULTS / "number_statistics.csv", index=False)

    print("=== Single-number analysis (n_draws =", n_draws, ") ===")
    print("Main-number counts: min/max/mean:", out["main_count"].min(), out["main_count"].max(), out["main_count"].mean())
    print("Uncorrected p<0.05 (main):", (out["main_p_raw"] < 0.05).sum(), "/ 31")
    print("Holm-significant (main):", out["main_sig_holm"].sum(), "/ 31")
    print("BH-FDR-significant (main):", out["main_sig_fdr"].sum(), "/ 31")
    print("Uncorrected p<0.05 (bonus):", (out["bonus_p_raw"] < 0.05).sum(), "/ 31")
    print("Holm-significant (bonus):", out["bonus_sig_holm"].sum(), "/ 31")
    print(out[["number", "main_count", "main_rate", "main_z", "main_p_raw", "main_p_holm", "main_sig_holm"]].sort_values("main_p_raw").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
