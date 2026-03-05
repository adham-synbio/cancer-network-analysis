"""
Binary classification of essential genes (top 10% DepMap) using network
centrality and Node2Vec embeddings.

Provides:
- Out-of-fold cross-validated AUROC / AUPRC
- Precision@K evaluation
- SHAP feature importance
- Comparison of feature sets: centrality-only vs centrality + node2vec
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    CENTRALITY_FEATURES,
    CV_N_SPLITS,
    ESSENTIALITY_PERCENTILE,
    RANDOM_STATE,
    XGB_PARAMS,
    MODEL_COMPARISON_CSV,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def build_labels(depmap_values: np.ndarray, percentile: int = ESSENTIALITY_PERCENTILE) -> np.ndarray:
    """
    Assign binary labels: 1 for genes in the most-essential percentile.

    Parameters
    ----------
    depmap_values : np.ndarray
        Continuous DepMap CRISPR effect scores.
    percentile : int
        Genes below this percentile (most negative) are labeled positive.

    Returns
    -------
    np.ndarray
        Binary label array (dtype int).
    """
    threshold = np.nanpercentile(depmap_values, percentile)
    return (depmap_values <= threshold).astype(int)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Compute Precision@K: fraction of true positives in the top-K predictions.

    Parameters
    ----------
    scores : np.ndarray
        Predicted probability scores.
    labels : np.ndarray
        Binary ground-truth labels.
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    float
        Precision at K.
    """
    mask = ~np.isnan(scores)
    s, l = scores[mask], labels[mask]
    idx = np.argsort(s)[::-1][: min(k, len(s))]
    return float(l[idx].mean()) if len(idx) > 0 else np.nan


def _make_xgb_classifier(**overrides) -> XGBClassifier:
    params = {**XGB_PARAMS, "eval_metric": "logloss", "tree_method": "hist"}
    params.update(overrides)
    return XGBClassifier(**params)


def _make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, n_jobs=-1
                ),
            ),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    feature_names: list,
    y: np.ndarray,
    suffix: str,
    k_values: list = None,
) -> dict:
    """
    Train Logistic Regression and XGBoost classifiers, evaluate with 5-fold
    out-of-fold CV, compute SHAP importances, and save result files.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame; must include a ``Gene`` column.
    feature_names : list
        Column names to use as features.
    y : np.ndarray
        Binary labels.
    suffix : str
        String appended to output filenames (e.g. ``"centrality_only"``).
    k_values : list, optional
        K values for Precision@K.  Defaults to [50, 100, 200, 500, 1000].

    Returns
    -------
    dict
        Performance metrics dictionary.
    """
    if k_values is None:
        k_values = [50, 100, 200, 500, 1000]

    X = df[feature_names].values
    cv = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Logistic Regression (cross-validated)
    logit = _make_logistic()
    oof_logit = cross_val_predict(logit, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc_logit = roc_auc_score(y, oof_logit)
    auprc_logit = average_precision_score(y, oof_logit)
    logger.info("[%s] Logistic AUROC: %.3f | AUPRC: %.3f", suffix, auroc_logit, auprc_logit)

    # XGBoost (cross-validated)
    xgb = _make_xgb_classifier()
    oof_xgb = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc_xgb = roc_auc_score(y, oof_xgb)
    auprc_xgb = average_precision_score(y, oof_xgb)
    logger.info("[%s] XGBoost AUROC: %.3f | AUPRC: %.3f", suffix, auroc_xgb, auprc_xgb)

    # Fit on full data for final probabilities and SHAP
    logit.fit(X, y)
    xgb.fit(X, y)
    prob_logit = logit.predict_proba(X)[:, 1]
    prob_xgb = xgb.predict_proba(X)[:, 1]

    # Precision@K
    prec_k = {k: precision_at_k(oof_xgb, y, k) for k in k_values}
    for k, p in prec_k.items():
        logger.info("[%s] XGBoost Precision@%d: %.3f", suffix, k, p)

    # SHAP feature importance
    explainer = shap.TreeExplainer(xgb)
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(df), size=min(3000, len(df)), replace=False)
    shap_values = explainer.shap_values(X[sample_idx])
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    shap_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
    )

    # Save outputs
    shap_path = RESULTS_DIR / f"shap_feature_importance_{suffix}.csv"
    shap_df.to_csv(shap_path, index=False)

    oof_path = RESULTS_DIR / f"classifier_oof_predictions_{suffix}.csv"
    pd.DataFrame(
        {
            "Gene": df["Gene"].values,
            "y_true": y,
            "oof_prob_logit": oof_logit,
            "oof_prob_xgb": oof_xgb,
        }
    ).to_csv(oof_path, index=False)

    rank = df[["Gene"] + [c for c in CENTRALITY_FEATURES if c in df.columns] + ["depmap_median_all"]].copy()
    rank["prob_logit"] = prob_logit
    rank["prob_xgb"] = prob_xgb
    scores_path = RESULTS_DIR / f"classifier_scores_{suffix}.csv"
    rank.sort_values("prob_xgb", ascending=False).to_csv(scores_path, index=False)

    logger.info("[%s] Results saved to %s, %s, %s", suffix, shap_path, oof_path, scores_path)

    return {
        "suffix": suffix,
        "n_features": len(feature_names),
        "logit_auroc": auroc_logit,
        "logit_auprc": auprc_logit,
        "xgb_auroc": auroc_xgb,
        "xgb_auprc": auprc_xgb,
        **{f"xgb_prec{k}": prec_k[k] for k in k_values},
    }


def run_feature_comparison(
    base: pd.DataFrame,
    emb: pd.DataFrame,
    emb_cols: list = None,
) -> pd.DataFrame:
    """
    Run the full classification pipeline comparing three feature sets:
    centrality-only, node2vec-only, and centrality + node2vec.

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
        Model comparison table sorted by XGBoost AUROC.
    """
    base = base[base["depmap_median_all"].notna()].copy()
    base["Gene"] = base["Gene"].astype(str)
    emb["Gene"] = emb["Gene"].astype(str)

    df = base.merge(emb, on="Gene", how="inner")

    if emb_cols is None:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

    y = build_labels(df["depmap_median_all"].values)

    results = []
    results.append(
        train_and_evaluate(df, CENTRALITY_FEATURES, y, "centrality_only")
    )
    results.append(
        train_and_evaluate(df, CENTRALITY_FEATURES + emb_cols, y, "centrality_plus_node2vec")
    )

    comparison = pd.DataFrame(results).sort_values("xgb_auroc", ascending=False)
    comparison.to_csv(MODEL_COMPARISON_CSV, index=False)
    logger.info("Saved model comparison to %s", MODEL_COMPARISON_CSV)
    return comparison
