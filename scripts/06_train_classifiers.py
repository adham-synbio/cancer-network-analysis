"""
Step 6 – Train and compare binary classifiers.

Compares three feature sets using 5-fold cross-validation:
    1. Centrality features only
    2. Node2Vec embeddings only
    3. Centrality + Node2Vec (combined)

For each feature set: Logistic Regression and XGBoost are evaluated on
AUROC, AUPRC, and Precision@K.  SHAP feature importances are saved for the
XGBoost models.

Outputs (per feature set):
    results/classifier_oof_predictions_{suffix}.csv
    results/classifier_scores_{suffix}.csv
    results/shap_feature_importance_{suffix}.csv

Summary:
    results/model_comparison_centrality_vs_combined.csv
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    CENTRALITY_FEATURES,
    ESSENTIALITY_PERCENTILE,
    RANDOM_STATE,
    CV_N_SPLITS,
    XGB_NODE2VEC_PARAMS,
    PRIORITIZED_GENES_CSV,
    NODE2VEC_EMBEDDINGS_CSV,
    OOF_NODE2VEC_CSV,
)
from gene_essentiality_analysis.models.classifiers import run_feature_comparison, build_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_node2vec_only(df: pd.DataFrame, emb_cols: list, y: np.ndarray) -> pd.DataFrame:
    """
    Train a node2vec-only XGBoost classifier with stratified 5-fold CV and
    save out-of-fold predictions.
    """
    X = df[emb_cols].values
    kf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        model = XGBClassifier(**XGB_NODE2VEC_PARAMS)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]

        fold_auc = roc_auc_score(y_val, oof[val_idx])
        logger.info("Node2vec-only fold %d/%d AUROC: %.4f", fold + 1, CV_N_SPLITS, fold_auc)

    overall_auc = roc_auc_score(y, oof)
    logger.info("Node2vec-only overall OOF AUROC: %.4f", overall_auc)

    oof_df = pd.DataFrame({"Gene": df["Gene"].values, "y_true": y, "oof_prob_xgb": oof})
    oof_df.to_csv(OOF_NODE2VEC_CSV, index=False)
    logger.info("Saved node2vec-only OOF predictions to %s", OOF_NODE2VEC_CSV)
    return oof_df


def main():
    logger.info("Loading prioritized gene table...")
    base = pd.read_csv(PRIORITIZED_GENES_CSV)

    logger.info("Loading Node2Vec embeddings...")
    emb = pd.read_csv(NODE2VEC_EMBEDDINGS_CSV)

    base = base[base["depmap_median_all"].notna()].copy()
    base["Gene"] = base["Gene"].astype(str)
    emb["Gene"] = emb["Gene"].astype(str)

    df = base.merge(emb, on="Gene", how="inner")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    y = build_labels(df["depmap_median_all"].values)

    logger.info("Training node2vec-only classifier...")
    train_node2vec_only(df, emb_cols, y)

    logger.info("Running centrality-only and combined model comparison...")
    comparison = run_feature_comparison(base, emb, emb_cols=emb_cols)

    logger.info("\nModel comparison summary:\n%s", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
