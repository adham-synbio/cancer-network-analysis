"""
Compute per-edge Spearman co-expression weights and combine them with the
STRING confidence score to produce a single composite edge weight.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from config import (
    COEXPR_WEIGHT_STRING,
    COEXPR_WEIGHT_SPEARMAN,
    WEIGHTED_EDGES_CSV,
)

logger = logging.getLogger(__name__)


def _edge_spearman(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Spearman rho between two expression vectors.

    Returns NaN if either vector is constant (undefined correlation).
    """
    if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
        return np.nan
    rho, _ = spearmanr(v1, v2)
    return float(rho)


def compute_coexpression_weights(
    ppi: pd.DataFrame,
    expr: pd.DataFrame,
    weight_string: float = COEXPR_WEIGHT_STRING,
    weight_spearman: float = COEXPR_WEIGHT_SPEARMAN,
) -> pd.DataFrame:
    """
    Annotate PPI edges with Spearman co-expression and a composite weight.

    The composite weight combines the normalized STRING confidence score and
    the positive Spearman rho::

        weight = w_string * s_scaled + w_spearman * rho.clip(lower=0)

    Parameters
    ----------
    ppi : pd.DataFrame
        PPI edge table with columns: gene1, gene2, combined_score.
        Only edges where both genes appear in *expr* are retained.
    expr : pd.DataFrame
        Expression matrix (genes x samples).
    weight_string : float
        Contribution of the normalized STRING score.
    weight_spearman : float
        Contribution of positive Spearman rho.

    Returns
    -------
    pd.DataFrame
        Columns: gene1, gene2, combined_score, spearman_tnbc, weight.
    """
    expressed_genes = set(expr.index)
    ppi_filt = ppi[
        ppi["gene1"].isin(expressed_genes) & ppi["gene2"].isin(expressed_genes)
    ].copy()
    logger.info(
        "PPI edges after expression filter: %d / %d", len(ppi_filt), len(ppi)
    )

    expr_map = {gene: expr.loc[gene].values for gene in expr.index}

    rho_values = []
    for row in tqdm(
        ppi_filt[["gene1", "gene2"]].itertuples(index=False),
        total=len(ppi_filt),
        desc="Computing Spearman weights",
    ):
        rho_values.append(_edge_spearman(expr_map[row[0]], expr_map[row[1]]))

    ppi_filt["spearman_tnbc"] = rho_values
    ppi_w = ppi_filt.dropna(subset=["spearman_tnbc"]).copy()

    s = ppi_w["combined_score"].astype(float)
    s_scaled = (s - s.min()) / (s.max() - s.min() + 1e-12)

    rho_pos = ppi_w["spearman_tnbc"].clip(lower=0)
    ppi_w["weight"] = weight_string * s_scaled + weight_spearman * rho_pos

    logger.info("Weighted edges (after NaN removal): %d", len(ppi_w))
    return ppi_w[["gene1", "gene2", "combined_score", "spearman_tnbc", "weight"]]


def load_or_compute_weights(
    ppi: pd.DataFrame,
    expr: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Return the weighted edge table, recomputing and caching if necessary.

    Parameters
    ----------
    ppi : pd.DataFrame
        PPI edge table (gene1, gene2, combined_score).
    expr : pd.DataFrame
        Expression matrix (genes x samples).
    **kwargs
        Forwarded to :func:`compute_coexpression_weights`.

    Returns
    -------
    pd.DataFrame
        Weighted edge table.
    """
    if WEIGHTED_EDGES_CSV.exists():
        logger.info("Loading cached weighted edges from %s", WEIGHTED_EDGES_CSV)
        return pd.read_csv(WEIGHTED_EDGES_CSV)

    ppi_w = compute_coexpression_weights(ppi, expr, **kwargs)
    ppi_w.to_csv(WEIGHTED_EDGES_CSV, index=False)
    logger.info("Saved weighted edges to %s", WEIGHTED_EDGES_CSV)
    return ppi_w
