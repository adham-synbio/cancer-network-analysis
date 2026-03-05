"""
Regression models to predict continuous DepMap essentiality scores from
network centrality features and Node2Vec embeddings.

Provides Spearman-rho cross-validation and final prediction tables.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    CENTRALITY_FEATURES,
    CV_N_SPLITS,
    RANDOM_STATE,
    XGB_PARAMS,
    PREDICTIONS_REGRESSION_CSV,
    NODE2VEC_EMBEDDINGS_CSV,
)

logger = logging.getLogger(__name__)


def _spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ok = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if ok.sum() < 10:
        return 0.0
    return float(spearmanr(y_true[ok], y_pred[ok]).correlation)


spearman_score = make_scorer(_spearman_scorer, greater_is_better=True)


def run_regression(
    base: pd.DataFrame,
    emb: pd.DataFrame,
    emb_cols: list = None,
) -> pd.DataFrame:
    """
    Fit ElasticNet and XGBoost regression models and evaluate via 5-fold CV.

    Parameters
    ----------
    base : pd.DataFrame
        Prioritized gene table with centrality features and ``depmap_median_all``.
    emb : pd.DataFrame
        Node2Vec embedding table (Gene, emb_0, emb_1, ...).
    emb_cols : list, optional
        Embedding column names; inferred from *emb* if not provided.

    Returns
    -------
    pd.DataFrame
        Gene table with predicted essentiality columns appended.
    """
    base = base[base["depmap_median_all"].notna()].copy()
    base["Gene"] = base["Gene"].astype(str)
    emb["Gene"] = emb["Gene"].astype(str)

    df = base.merge(emb, on="Gene", how="inner")
    logger.info("Merged feature table shape: %s", df.shape)

    if emb_cols is None:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

    feature_cols = CENTRALITY_FEATURES + emb_cols
    X = df[feature_cols].values
    y = df["depmap_median_all"].values

    cv = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ElasticNet
    enet = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9],
                    alphas=np.logspace(-4, 2, 30),
                    max_iter=8000,
                    cv=5,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    enet_spear = cross_val_score(enet, X, y, cv=cv, scoring=spearman_score)
    enet_r2 = cross_val_score(enet, X, y, cv=cv, scoring="r2")
    logger.info(
        "ElasticNet Spearman CV: %.3f +/- %.3f | R2: %.3f +/- %.3f",
        np.nanmean(enet_spear), np.nanstd(enet_spear),
        np.nanmean(enet_r2), np.nanstd(enet_r2),
    )
    enet.fit(X, y)
    df["pred_enet_emb"] = enet.predict(X)

    # XGBoost
    xgb_params = {**XGB_PARAMS}
    xgb_params.pop("eval_metric", None)
    xgb = XGBRegressor(**xgb_params)
    xgb_spear = cross_val_score(xgb, X, y, cv=cv, scoring=spearman_score)
    xgb_r2 = cross_val_score(xgb, X, y, cv=cv, scoring="r2")
    logger.info(
        "XGBoost Spearman CV: %.3f +/- %.3f | R2: %.3f +/- %.3f",
        np.nanmean(xgb_spear), np.nanstd(xgb_spear),
        np.nanmean(xgb_r2), np.nanstd(xgb_r2),
    )
    xgb.fit(X, y)
    df["pred_xgb_emb"] = xgb.predict(X)

    # Higher predicted essentiality score -> more essential (flip sign)
    df["pred_score_enet_emb"] = -df["pred_enet_emb"]
    df["pred_score_xgb_emb"] = -df["pred_xgb_emb"]

    out_cols = (
        ["Gene"]
        + CENTRALITY_FEATURES
        + ["depmap_median_all"]
        + emb_cols
        + ["pred_enet_emb", "pred_xgb_emb", "pred_score_enet_emb", "pred_score_xgb_emb"]
    )
    result = df[out_cols].sort_values("pred_score_xgb_emb", ascending=False)
    result.to_csv(PREDICTIONS_REGRESSION_CSV, index=False)
    logger.info("Saved regression predictions to %s", PREDICTIONS_REGRESSION_CSV)
    return result
