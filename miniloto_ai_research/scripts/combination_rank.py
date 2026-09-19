"""
Step 22 / Step 31: score and rank all 169,911 combinations for DRAW 1405.

Model-adoption rule (decided BEFORE looking at this output, in
walk_forward.py's fixed criteria): a model enters the final ensemble only if
model_performance.csv shows sig_better_than_random_fdr == True (i.e. it beat
the random/uniform baseline on out-of-sample Brier score, surviving
Benjamini-Hochberg FDR correction across all models tested).

walk_forward.py's result: ZERO of the 18 tested models (10 baselines + 8 ML
models) met this bar. So per the pre-registered rule, the validated ensemble
is EMPTY, and every one of the 169,911 combinations is scored identically
under the fair-draw model (each has probability 1/169911). This is not a
bug or a placeholder -- it is the correct, honest output of the adoption
rule given the walk-forward result, and it is the main empirical finding of
this project.

Because the task still requires a specific numerical Rank/score/10 tickets,
ties are broken by ONE fixed, pre-registered mechanism: a numpy Generator
seeded with SEED=20260920, applied AFTER scoring (so it cannot be tuned to
produce a favorable-looking result) and identical in spirit to literally
drawing 10 tickets uniformly at random -- which is the only defensible
action once no predictive edge exists. An exploratory (non-adopted,
research-only) composite score from the tested-but-unvalidated models is
also saved for transparency, clearly separated from the official ranking.
"""
import pathlib
import numpy as np
import pandas as pd

SEED = 20260920
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]


def main():
    combos = pd.read_csv(DATA / "all_169911_combinations.csv")
    perf = pd.read_csv(RESULTS / "model_performance.csv")

    adopted = perf[perf["sig_better_than_random_fdr"] == True]["model"].tolist()
    print(f"Models meeting adoption criteria (OOS beat random, FDR-significant): {adopted}")
    print(f"-> Validated ensemble size: {len(adopted)} (out of {len(perf)-1} non-random models tested)")

    combos["official_score"] = 1.0 / 169911.0  # uniform: no validated model to differentiate combos
    combos["model_components"] = "NONE_VALIDATED (uniform/random baseline per pre-registered adoption rule)"

    rng = np.random.default_rng(SEED)
    tie_break = rng.permutation(len(combos))
    combos["tie_break_key"] = tie_break
    combos = combos.sort_values(["official_score", "tie_break_key"], ascending=[False, True]).reset_index(drop=True)
    combos["rank"] = np.arange(1, len(combos) + 1)
    combos["relative_score"] = 1.0  # all combinations equally likely under the adopted (null) model

    # ---- exploratory-only composite (NOT used for ticket selection) ----
    # Build a same-training-cutoff (through draw 1404) feature snapshot and score
    # every number with the untuned raw average of tested (but unvalidated) ML
    # model outputs, purely for research transparency / Step 42 sensitivity checks.
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from feature_engineering import load_draws, build_feature_table
    from models import get_ml_models, FEATURE_COLS

    df = load_draws()
    feat = build_feature_table(df, min_train_draws=200)
    # "predict draw 1405": append one more synthetic row set using state as of after draw 1404
    # by re-using build_feature_table's causal logic up to t=len(df) (one-past-end).
    # Simplest safe approach: extend df conceptually is not needed -- reuse the last
    # available per-number feature snapshot (t_idx = len(df)-1 predicts draw 1404 using
    # data through 1403; we need one more step). We recompute with min_train_draws=len(df)-1
    # so the function naturally emits a row set for t=len(df)-1 AND we manually roll one
    # more step using the same causal update logic by calling with an appended dummy line
    # is unsafe (would need real draw 1405, which we must not use). Instead: build a
    # snapshot using ALL 1404 real draws as "training", producing next-draw features by
    # hand from the same running-state logic (duplicated minimal subset here).
    N_POOL = 31
    n = len(df)
    occ = np.zeros((n, N_POOL + 1), dtype=np.int8)
    for t, row in df.iterrows():
        for c in MAIN_COLS:
            occ[t, int(row[c])] = 1
    total_count = occ.sum(axis=0)
    last_seen_idx = np.full(N_POOL + 1, -1, dtype=np.int64)
    for num in range(1, N_POOL + 1):
        seen = np.where(occ[:, num] == 1)[0]
        if len(seen):
            last_seen_idx[num] = seen[-1]
    prev_draw_numbers = [num for num in range(1, N_POOL + 1) if occ[n - 1, num] == 1]

    from math import comb
    pair_cooc = np.zeros((N_POOL + 1, N_POOL + 1), dtype=np.int64)
    for t in range(n):
        cur = [num for num in range(1, N_POOL + 1) if occ[t, num] == 1]
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                a, b = min(cur[i], cur[j]), max(cur[i], cur[j])
                pair_cooc[a, b] += 1

    rows_1405 = []
    for num in range(1, N_POOL + 1):
        rec = {"number": num, "full_freq": total_count[num] / n,
               "gap_since_seen": (n - 1 - last_seen_idx[num]),
               "prev_appeared": int(occ[n - 1, num]), "prev_was_bonus": 0}
        for w in [20, 50, 100, 200]:
            hist = occ[max(0, n - w):n, num]
            rec[f"freq_last_{w}"] = hist.mean()
        for hl in [20, 50, 100]:
            alpha = 1 - 0.5 ** (1 / hl)
            hist = occ[:n, num]
            w_ = (1 - alpha) ** np.arange(len(hist) - 1, -1, -1)
            rec[f"ewma_hl{hl}"] = float(np.sum(w_ * hist) / np.sum(w_))
        p_pair = comb(29, 3) / comb(31, 5)
        lifts = []
        for pn in prev_draw_numbers:
            if pn == num:
                continue
            a, b = min(num, pn), max(num, pn)
            exp = n * p_pair
            lifts.append(pair_cooc[a, b] / exp if exp > 0 else 1.0)
        rec["pair_lift_vs_prev_draw"] = float(np.mean(lifts)) if lifts else 1.0
        rows_1405.append(rec)
    snap_1405 = pd.DataFrame(rows_1405)

    X_train = feat[FEATURE_COLS].fillna(feat[FEATURE_COLS].median())
    y_train = feat["y"].values
    X_pred = snap_1405[FEATURE_COLS].fillna(X_train.median())
    ml_models = get_ml_models(SEED)
    exploratory_number_scores = {"number": list(range(1, N_POOL + 1))}
    ensemble_probs = np.zeros(N_POOL)
    n_models = 0
    for name, model in ml_models.items():
        try:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_pred)[:, 1]
        except Exception:
            proba = np.full(N_POOL, 5 / 31)
        exploratory_number_scores[name] = proba
        ensemble_probs += proba
        n_models += 1
    ensemble_probs /= n_models
    exploratory_number_scores["exploratory_unvalidated_ensemble_mean"] = ensemble_probs
    exploratory_df = pd.DataFrame(exploratory_number_scores)
    exploratory_df.to_csv(RESULTS / "exploratory_unvalidated_number_scores_draw1405.csv", index=False)
    print("\n=== Exploratory (NOT validated, research-only) per-number scores for draw 1405 ===")
    print(exploratory_df[["number", "exploratory_unvalidated_ensemble_mean"]].sort_values("exploratory_unvalidated_ensemble_mean", ascending=False).to_string(index=False))

    combos.to_csv(RESULTS / "top169911.csv", index=False)
    combos.head(100).to_csv(RESULTS / "top100.csv", index=False)
    print(f"\nSaved official ranking (uniform score + pre-registered seeded tie-break) for all {len(combos)} combinations.")
    print(combos.head(10)[["rank", "n1", "n2", "n3", "n4", "n5", "official_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
