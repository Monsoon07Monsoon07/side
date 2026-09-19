"""Step 4: enumerate all 169,911 possible mini loto combinations."""
import itertools
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    combos = list(itertools.combinations(range(1, 32), 5))
    assert len(combos) == 169911, len(combos)
    df = pd.DataFrame(combos, columns=["n1", "n2", "n3", "n4", "n5"])
    df.insert(0, "combo_id", np.arange(len(df)))

    df["sum"] = df[["n1", "n2", "n3", "n4", "n5"]].sum(axis=1)
    df["odd_count"] = df[["n1", "n2", "n3", "n4", "n5"]].apply(lambda r: sum(x % 2 == 1 for x in r), axis=1)
    df["low_count"] = df[["n1", "n2", "n3", "n4", "n5"]].apply(lambda r: sum(x <= 15 for x in r), axis=1)

    df.to_csv(DATA / "all_169911_combinations.csv", index=False)
    print("total combinations:", len(df))
    print(df.head())
    print("sum range:", df["sum"].min(), df["sum"].max())


if __name__ == "__main__":
    main()
