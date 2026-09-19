"""Step 6: rolling-window and EWMA frequency analysis per number.
Reports descriptive rolling stats only (no window is chosen as 'best' by
peeking at full-period performance here; window selection for any model use
happens strictly inside walk_forward.py's training folds)."""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]
WINDOWS = [10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500]
HALF_LIVES = [10, 20, 30, 50, 75, 100, 150, 200]


def main():
    df = pd.read_csv(DATA / "miniloto_1_1404_clean.csv").sort_values("draw_no").reset_index(drop=True)
    n = len(df)
    occ = np.zeros((n, 32), dtype=np.int8)  # occ[t, num] = 1 if num was a main number at draw t
    for t, row in df.iterrows():
        for c in MAIN_COLS:
            occ[t, int(row[c])] = 1

    rows = []
    for t in range(n):
        draw_no = int(df.loc[t, "draw_no"])
        for num in range(1, 32):
            rec = {"draw_no": draw_no, "number": num}
            for w in WINDOWS:
                lo = max(0, t - w)
                hist = occ[lo:t, num]  # strictly prior draws only (no leakage)
                rec[f"freq_last_{w}"] = hist.mean() if len(hist) > 0 else np.nan
            for hl in HALF_LIVES:
                if t == 0:
                    rec[f"ewma_hl{hl}"] = np.nan
                    continue
                alpha = 1 - 0.5 ** (1 / hl)
                hist = occ[:t, num]
                weights = (1 - alpha) ** np.arange(len(hist) - 1, -1, -1)
                rec[f"ewma_hl{hl}"] = np.sum(weights * hist) / np.sum(weights)
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "rolling_statistics.csv", index=False)
    print("rolling_statistics rows:", len(out))
    print(out[out["draw_no"] == 1404].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
