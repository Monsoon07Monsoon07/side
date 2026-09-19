"""Step 12 (shape) + Step 13 (autocorrelation/runs) + Step 14 (change point) +
Step 15 (date features), combined for efficiency."""
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import chisquare, pearsonr
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


def shape_features(row_nums):
    nums = sorted(row_nums)
    s = sum(nums)
    odd = sum(1 for x in nums if x % 2 == 1)
    low = sum(1 for x in nums if x <= 15)
    prime = sum(1 for x in nums if x in PRIMES)
    diffs = [nums[i + 1] - nums[i] for i in range(4)]
    consec_pairs = sum(1 for d in diffs if d == 1)
    max_consec_run = 1
    cur = 1
    for d in diffs:
        if d == 1:
            cur += 1
            max_consec_run = max(max_consec_run, cur)
        else:
            cur = 1
    tens = [x // 10 for x in nums]  # decade bucket: 0(1-9),1(10-19),2(20-29),3(30-31)
    decade_spread = len(set(tens))
    return {
        "sum": s, "mean": s / 5, "range": nums[-1] - nums[0], "std": np.std(nums),
        "odd_count": odd, "even_count": 5 - odd, "low_count": low, "high_count": 5 - low,
        "prime_count": prime, "consecutive_pair_count": consec_pairs, "max_consecutive_run": max_consec_run,
        "d12": diffs[0], "d23": diffs[1], "d34": diffs[2], "d45": diffs[3],
        "decade_spread": decade_spread,
    }


def main():
    df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)
    n = len(df)

    feat_rows = []
    for _, row in df.iterrows():
        nums = [int(row[c]) for c in MAIN_COLS]
        f = shape_features(nums)
        f["draw_no"] = int(row["draw_no"])
        f["date"] = row["date"]
        feat_rows.append(f)
    feat_df = pd.DataFrame(feat_rows)

    # theoretical distribution from full 169,911 enumeration
    combos = pd.read_csv(DATA / "all_169911_combinations.csv")
    theo_feats = []
    for _, r in combos.iterrows():
        pass  # too slow row-wise; vectorize below instead

    arr = combos[["n1", "n2", "n3", "n4", "n5"]].to_numpy()
    theo_sum = arr.sum(axis=1)
    theo_odd = (arr % 2 == 1).sum(axis=1)
    theo_low = (arr <= 15).sum(axis=1)

    obs_sum_mean, theo_sum_mean = feat_df["sum"].mean(), theo_sum.mean()
    obs_odd_mean, theo_odd_mean = feat_df["odd_count"].mean(), theo_odd.mean()
    obs_low_mean, theo_low_mean = feat_df["low_count"].mean(), theo_low.mean()

    shape_summary = pd.DataFrame([
        {"feature": "sum", "observed_mean": obs_sum_mean, "theoretical_mean": theo_sum_mean,
         "observed_std": feat_df["sum"].std(), "theoretical_std": theo_sum.std()},
        {"feature": "odd_count", "observed_mean": obs_odd_mean, "theoretical_mean": theo_odd_mean,
         "observed_std": feat_df["odd_count"].std(), "theoretical_std": theo_odd.std()},
        {"feature": "low_count", "observed_mean": obs_low_mean, "theoretical_mean": theo_low_mean,
         "observed_std": feat_df["low_count"].std(), "theoretical_std": theo_low.std()},
    ])

    # chi-square goodness of fit for odd_count distribution (0-5) observed vs theoretical
    obs_counts = feat_df["odd_count"].value_counts().reindex(range(6), fill_value=0).sort_index()
    theo_props = pd.Series(theo_odd).value_counts(normalize=True).reindex(range(6), fill_value=0).sort_index()
    theo_counts_expected = theo_props * n
    chi2_stat, chi2_p = chisquare(obs_counts, f_exp=theo_counts_expected)

    feat_df.to_csv(RESULTS / "structural_shape_features.csv", index=False)
    shape_summary.to_csv(RESULTS / "structural_shape_summary.csv", index=False)

    print("=== Shape feature check (observed draws vs theoretical 169,911-combo distribution) ===")
    print(shape_summary.to_string(index=False))
    print(f"\nodd_count chi-square goodness of fit: chi2={chi2_stat:.3f}, p={chi2_p:.4f}")

    # ---- Step 13: autocorrelation / runs test per number ----
    occ = np.zeros((n, 32), dtype=np.int8)
    for t, row in df.iterrows():
        for c in MAIN_COLS:
            occ[t, int(row[c])] = 1

    from statsmodels.stats.diagnostic import acorr_ljungbox

    def runs_test(x):
        # Wald-Wolfowitz runs test for binary sequence
        x = np.asarray(x)
        n1 = (x == 1).sum(); n0 = (x == 0).sum()
        runs = 1 + np.sum(x[1:] != x[:-1])
        mean_runs = 2 * n1 * n0 / (n1 + n0) + 1
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
        if var_runs <= 0:
            return np.nan, np.nan
        z = (runs - mean_runs) / np.sqrt(var_runs)
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(abs(z)))
        return z, p

    ts_rows = []
    for num in range(1, 32):
        series = occ[:, num]
        acf1 = pd.Series(series).autocorr(lag=1)
        try:
            lb = acorr_ljungbox(series, lags=[10], return_df=True)
            lb_p = lb["lb_pvalue"].values[0]
        except Exception:
            lb_p = np.nan
        z_runs, p_runs = runs_test(series)
        ts_rows.append({"number": num, "acf_lag1": acf1, "ljungbox_lag10_p": lb_p, "runs_test_z": z_runs, "runs_test_p": p_runs})
    ts_df = pd.DataFrame(ts_rows)
    reject, p_adj, _, _ = multipletests(ts_df["ljungbox_lag10_p"].fillna(1.0), alpha=0.05, method="fdr_bh")
    ts_df["ljungbox_p_fdr"] = p_adj
    ts_df["ljungbox_sig_fdr"] = reject
    ts_df.to_csv(RESULTS / "timeseries_autocorrelation.csv", index=False)

    print(f"\n=== Step 13: autocorrelation/runs test ===")
    print(f"Ljung-Box FDR-significant (autocorrelation present) numbers: {ts_df['ljungbox_sig_fdr'].sum()} / 31")
    print(ts_df.sort_values("ljungbox_lag10_p").head(5).to_string(index=False))

    # ---- Step 14: structural change point (CUSUM on rolling sum-of-draw and on number-11 frequency, as example) ----
    # Use overall draw "sum" series' CUSUM as a general structural-change probe.
    sum_series = feat_df["sum"].values
    mean_sum = sum_series.mean()
    cusum = np.cumsum(sum_series - mean_sum)
    max_cusum_idx = int(np.argmax(np.abs(cusum)))
    change_point_draw = int(feat_df.iloc[max_cusum_idx]["draw_no"])

    # permutation test: is the max |CUSUM| bigger than under random shuffles of the same data?
    rng = np.random.default_rng(20260920)
    n_perm = 2000
    max_abs_cusum_obs = np.max(np.abs(cusum))
    perm_maxes = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(sum_series)
        c = np.cumsum(shuffled - mean_sum)
        perm_maxes[i] = np.max(np.abs(c))
    cp_p = (perm_maxes >= max_abs_cusum_obs).mean()

    cp_df = pd.DataFrame([{
        "series": "draw_sum", "candidate_change_point_draw_no": change_point_draw,
        "max_abs_cusum": max_abs_cusum_obs, "permutation_p_value": cp_p, "n_permutations": n_perm,
    }])
    cp_df.to_csv(RESULTS / "structural_statistics.csv", index=False)
    print(f"\n=== Step 14: CUSUM structural change probe on draw-sum series ===")
    print(f"Candidate change point near draw {change_point_draw}, permutation p={cp_p:.4f} (n_perm={n_perm})")

    # ---- Step 15: date features (weekday, month, season) ----
    feat_df["date_parsed"] = pd.to_datetime(feat_df["date"])
    feat_df["weekday"] = feat_df["date_parsed"].dt.day_name()
    feat_df["month"] = feat_df["date_parsed"].dt.month
    weekday_counts = feat_df["weekday"].value_counts()
    print(f"\n=== Step 15: weekday distribution of draws ===")
    print(weekday_counts.to_string())
    weekday_counts.to_csv(RESULTS / "date_weekday_distribution.csv")


if __name__ == "__main__":
    main()
