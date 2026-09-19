"""Baseline and ML model definitions for the mini loto walk-forward study.
Every model here consumes ONLY the causal feature table produced by
feature_engineering.py (rows for t already exclude any information from
draw t itself or later). Baselines are closed-form; ML models are thin
sklearn/xgboost/lightgbm wrappers with modest, fixed hyperparameters (no
tuning against test-fold outcomes anywhere in this file)."""
import numpy as np
import pandas as pd

BASELINE_SCORE_COLS = {
    "B_full_freq": "full_freq",
    "D_ewma50": "ewma_hl100",  # placeholder; overridden by selected halflife in walk_forward
    "F_hot": "freq_last_20",
    "H_pair": "pair_lift_vs_prev_draw",
}


def score_random(train_df, pred_df):
    return np.full(len(pred_df), 5 / 31)


def score_full_freq(train_df, pred_df):
    return pred_df["full_freq"].fillna(5 / 31).values


def score_rolling(train_df, pred_df, window_col):
    return pred_df[window_col].fillna(5 / 31).values


def score_ewma(train_df, pred_df, hl_col):
    return pred_df[hl_col].fillna(5 / 31).values


def score_gap(train_df, pred_df):
    # normalize gap into a probability-like score per draw: larger gap -> higher score
    g = pred_df.groupby("t_idx")["gap_since_seen"]
    ranked = pred_df["gap_since_seen"] / g.transform("sum").replace(0, np.nan)
    return (ranked * 5).fillna(5 / 31).values


def score_hot(train_df, pred_df, window_col="freq_last_20"):
    return pred_df[window_col].fillna(5 / 31).values


def score_cold(train_df, pred_df, window_col="freq_last_20"):
    inv = 1.0 / (pred_df[window_col].fillna(5 / 31) + 0.02)
    norm = inv.groupby(pred_df["t_idx"]).transform(lambda s: s / s.sum() * 5)
    return norm.values


def score_pair(train_df, pred_df):
    return pred_df["pair_lift_vs_prev_draw"].fillna(1.0).values * (5 / 31)


def score_bayesian(train_df, pred_df, prior_strength=100):
    prior_p = 5 / 31
    # use training-fold cumulative counts already encoded in full_freq * draws_processed;
    # approximate draws_processed via t_idx (rows are per-number, 31 per draw)
    n_train_draws = train_df["t_idx"].nunique()
    x = (pred_df["full_freq"].fillna(prior_p) * n_train_draws)
    shrunk = (x + prior_p * prior_strength) / (n_train_draws + prior_strength)
    return shrunk.values


def score_transition(train_df, pred_df):
    # causal conditional rate estimated from TRAIN data only, applied to pred rows'
    # prev_appeared flag
    cond = train_df.groupby("prev_appeared")["y"].mean()
    default = train_df["y"].mean()
    return pred_df["prev_appeared"].map(cond).fillna(default).values


FEATURE_COLS = [
    "full_freq", "gap_since_seen", "prev_appeared", "prev_was_bonus",
    "freq_last_20", "freq_last_50", "freq_last_100", "freq_last_200",
    "ewma_hl20", "ewma_hl50", "ewma_hl100", "pair_lift_vs_prev_draw",
]


def get_ml_models(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    import xgboost as xgb
    import lightgbm as lgb

    models = {
        "ML_LogReg_L2": LogisticRegression(penalty="l2", max_iter=2000, random_state=seed),
        "ML_LogReg_L1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=seed),
        "ML_LogReg_ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=2000, random_state=seed),
        "ML_RandomForest": RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=20, random_state=seed, n_jobs=-1),
        "ML_ExtraTrees": ExtraTreesClassifier(n_estimators=150, max_depth=4, min_samples_leaf=20, random_state=seed, n_jobs=-1),
        "ML_GradientBoosting": GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=seed),
        "ML_XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, random_state=seed, n_jobs=-1),
        "ML_LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed, verbosity=-1, n_jobs=-1),
    }
    return models
