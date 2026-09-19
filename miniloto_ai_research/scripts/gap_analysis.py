"""Step 7: gap/overdue analysis + re-appearance probability by gap-since-last-seen,
and an explicit gambler's-fallacy-vs-signal check via logistic regression of
P(appear next draw) on current gap length."""
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]


def main():
    df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)
    n = len(df)
    occ = np.zeros((n, 32), dtype=np.int8)
    for t, row in df.iterrows():
        for c in MAIN_COLS:
            occ[t, int(row[c])] = 1

    # For each number, list of draw indices where it appeared
    gap_rows = []
    reappear_rows = []
    current_gap_rows = []  # (gap_before_this_draw, appeared_this_draw) pairs, all numbers, all draws
    for num in range(1, 32):
        appearances = np.where(occ[:, num] == 1)[0]
        gaps = np.diff(appearances)
        gap_rows.append({
            "number": num,
            "n_appearances": len(appearances),
            "current_gap_since_last_seen": (n - 1 - appearances[-1]) if len(appearances) else np.nan,
            "mean_gap": gaps.mean() if len(gaps) else np.nan,
            "median_gap": np.median(gaps) if len(gaps) else np.nan,
            "max_gap": gaps.max() if len(gaps) else np.nan,
            "std_gap": gaps.std() if len(gaps) else np.nan,
        })
        # re-appearance probability k draws after an appearance, k=1..30
        for k in range(1, 31):
            idx = appearances + k
            idx = idx[idx < n]
            if len(idx) == 0:
                continue
            reappear_rate = occ[idx, num].mean()
            reappear_rows.append({"number": num, "k_draws_after": k, "reappear_rate": reappear_rate, "n_obs": len(idx)})

        # build gap-at-time-t series for gambler's fallacy regression (pooled across numbers)
        last_seen = -1
        for t in range(n):
            gap_before = t - last_seen - 1 if last_seen >= 0 else np.nan
            if not np.isnan(gap_before):
                current_gap_rows.append({"number": num, "draw_no_idx": t, "gap_before": gap_before, "appeared": int(occ[t, num])})
            if occ[t, num] == 1:
                last_seen = t

    gap_df = pd.DataFrame(gap_rows)
    reappear_df = pd.DataFrame(reappear_rows)
    cg_df = pd.DataFrame(current_gap_rows)

    # Pooled logistic regression: appeared ~ gap_before (does longer gap predict higher/lower next-draw prob?)
    X = sm.add_constant(cg_df["gap_before"])
    y = cg_df["appeared"]
    model = sm.Logit(y, X).fit(disp=0)
    coef = model.params["gap_before"]
    pval = model.pvalues["gap_before"]

    corr_r, corr_p = pearsonr(cg_df["gap_before"], cg_df["appeared"])

    summary = pd.DataFrame([{
        "test": "pooled_logistic_appeared_on_gap_before",
        "n_obs": len(cg_df), "coef_gap": coef, "p_value": pval,
        "pearson_r": corr_r, "pearson_p": corr_p,
        "interpretation": (
            "coef>0 & sig -> overdue numbers MORE likely next draw (supports overdue hypothesis); "
            "coef<0 & sig -> recently-seen numbers MORE likely next draw (momentum/hot-hand); "
            "not significant -> consistent with gambler's fallacy being FALSE signal, i.e. fair/memoryless draws"
        ),
    }])

    gap_df.to_csv(RESULTS / "gap_statistics.csv", index=False)
    reappear_df.to_csv(RESULTS / "gap_reappearance_curve.csv", index=False)
    summary.to_csv(RESULTS / "gap_gamblers_fallacy_test.csv", index=False)

    print(gap_df.to_string(index=False))
    print("\nPooled logistic regression appeared ~ gap_before:")
    print(f"  n_obs={len(cg_df)}, coef={coef:.6f}, p={pval:.4f}, pearson_r={corr_r:.4f}, pearson_p={corr_p:.4f}")


if __name__ == "__main__":
    main()
