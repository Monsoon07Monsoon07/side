"""
Builds a strictly-causal, long-format (draw_no, number) feature table:
every feature for (t, num) uses ONLY information from draws 1..t-1.
Target y = whether `num` is one of the 5 main numbers in draw t.

This is the single shared feature-construction path used by every baseline
and ML model in walk_forward.py, so there is exactly one place leakage
could be introduced -- and it is guarded by construction (each feature at
row t is computed from occ[:t], never occ[t] or later).
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]
ROLL_WINDOWS = [20, 50, 100, 200]
EWMA_HALFLIVES = [20, 50, 100]
N_POOL = 31


def load_draws():
    df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)
    return df


def build_occupancy(df):
    n = len(df)
    occ = np.zeros((n, N_POOL + 1), dtype=np.int8)
    bonus_occ = np.zeros((n, N_POOL + 1), dtype=np.int8)
    for t, row in df.iterrows():
        for c in MAIN_COLS:
            occ[t, int(row[c])] = 1
        bonus_occ[t, int(row["bonus"])] = 1
    return occ, bonus_occ


def build_feature_table(df, min_train_draws=200):
    """Returns long-format DataFrame: one row per (t, number) for t >= min_train_draws
    (index t is 0-based position in df, i.e. predicting df.loc[t])."""
    n = len(df)
    occ, bonus_occ = build_occupancy(df)

    # cumulative pair co-occurrence matrix (causal: updated AFTER processing each draw)
    pair_cooc = np.zeros((N_POOL + 1, N_POOL + 1), dtype=np.int64)
    draws_processed = 0

    # per-number running stats (causal, updated after each draw)
    total_count = np.zeros(N_POOL + 1, dtype=np.int64)
    last_seen_idx = np.full(N_POOL + 1, -1, dtype=np.int64)

    rows = []
    prev_draw_numbers = None

    for t in range(n):
        if t >= min_train_draws:
            draw_no = int(df.loc[t, "draw_no"])
            for num in range(1, N_POOL + 1):
                full_freq = total_count[num] / draws_processed if draws_processed > 0 else np.nan
                gap = (t - 1 - last_seen_idx[num]) if last_seen_idx[num] >= 0 else t
                rec = {
                    "t_idx": t, "draw_no": draw_no, "number": num,
                    "full_freq": full_freq,
                    "gap_since_seen": gap,
                    "prev_appeared": int(occ[t - 1, num]) if t > 0 else 0,
                    "prev_was_bonus": int(bonus_occ[t - 1, num]) if t > 0 else 0,
                }
                for w in ROLL_WINDOWS:
                    lo = max(0, t - w)
                    hist = occ[lo:t, num]
                    rec[f"freq_last_{w}"] = hist.mean() if len(hist) > 0 else np.nan
                for hl in EWMA_HALFLIVES:
                    if t == 0:
                        rec[f"ewma_hl{hl}"] = np.nan
                    else:
                        alpha = 1 - 0.5 ** (1 / hl)
                        hist = occ[:t, num]
                        w_ = (1 - alpha) ** np.arange(len(hist) - 1, -1, -1)
                        rec[f"ewma_hl{hl}"] = float(np.sum(w_ * hist) / np.sum(w_))
                # pair-lift feature: average cumulative co-occurrence lift of `num` with
                # each of prev draw's 5 numbers, using ONLY pair counts through t-1
                if prev_draw_numbers is not None and draws_processed > 0:
                    p_pair_theo = 10 / (N_POOL * (N_POOL - 1) / 2) * (5 * 4 / 2) / 10  # placeholder overwritten below
                    lifts = []
                    for pn in prev_draw_numbers:
                        if pn == num:
                            continue
                        a, b = min(num, pn), max(num, pn)
                        obs = pair_cooc[a, b]
                        from math import comb
                        p_pair = comb(29, 3) / comb(31, 5)
                        exp = draws_processed * p_pair
                        lifts.append(obs / exp if exp > 0 else 1.0)
                    rec["pair_lift_vs_prev_draw"] = float(np.mean(lifts)) if lifts else 1.0
                else:
                    rec["pair_lift_vs_prev_draw"] = 1.0

                rec["y"] = int(occ[t, num])
                rows.append(rec)

        # ---- update causal running state AFTER emitting features for draw t ----
        cur_numbers = [num for num in range(1, N_POOL + 1) if occ[t, num] == 1]
        for num in cur_numbers:
            total_count[num] += 1
            last_seen_idx[num] = t
        for i in range(len(cur_numbers)):
            for j in range(i + 1, len(cur_numbers)):
                a, b = min(cur_numbers[i], cur_numbers[j]), max(cur_numbers[i], cur_numbers[j])
                pair_cooc[a, b] += 1
        draws_processed += 1
        prev_draw_numbers = cur_numbers

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_draws()
    feat = build_feature_table(df, min_train_draws=200)
    out_path = DATA.parent / "results" / "feature_table_cache.parquet"
    feat.to_parquet(out_path)
    print("feature table shape:", feat.shape)
    print(feat.head())
    print("y mean (should be close to 5/31=0.1613):", feat["y"].mean())
