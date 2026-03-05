"""
Integrate network centralities with DepMap essentiality to produce a
ranked list of candidate therapeutic targets.

Priority score logic
--------------------
Each centrality metric and the DepMap essentiality score are converted to
a [0, 1] rank, then averaged.  Higher centrality and stronger essentiality
both increase the priority score.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    CENTRALITY_FEATURES,
    DEPMAP_CORRELATIONS_CSV,
    PRIORITIZED_GENES_CSV,
)

logger = logging.getLogger(__name__)


def _rank01(series: pd.Series, ascending: bool = False) -> pd.Series:
    """
    Convert a numeric Series to [0, 1] ranks.

    Parameters
    ----------
    series : pd.Series
        Input values.
    ascending : bool
        If True, larger raw values map to lower ranks (useful for essentiality
        where more-negative = more essential).

    Returns
    -------
    pd.Series
        Rank-normalized values in [0, 1].
    """
    ranked = series.rank(method="average", ascending=ascending)
    span = ranked.max() - 1
    return 1.0 - (ranked - 1) / (span if span > 0 else 1)


def compute_depmap_correlations(
    centralities: pd.DataFrame,
    depmap_median: pd.DataFrame,
    centrality_cols: list = None,
) -> pd.DataFrame:
    """
    Compute Spearman correlations between each centrality metric and DepMap
    median essentiality.

    Parameters
    ----------
    centralities : pd.DataFrame
        Must contain a ``Gene`` column (or ``gene`` which is renamed) and
        the centrality metric columns.
    depmap_median : pd.DataFrame
        Must contain ``Gene`` and ``depmap_median_all`` columns.
    centrality_cols : list, optional
        Centrality columns to evaluate; defaults to ``CENTRALITY_FEATURES``.

    Returns
    -------
    pd.DataFrame
        Columns: metric, spearman_rho_vs_depmap_median, p_value, n.
    """
    if centrality_cols is None:
        centrality_cols = CENTRALITY_FEATURES

    cent = centralities.copy()
    if "gene" in cent.columns and "Gene" not in cent.columns:
        cent = cent.rename(columns={"gene": "Gene"})
    cent["Gene"] = cent["Gene"].astype(str)

    df = cent.merge(depmap_median, on="Gene", how="left")

    rows = []
    x = df["depmap_median_all"]
    for col in centrality_cols:
        y = df[col]
        valid = x.notna() & y.notna()
        n = int(valid.sum())
        if n >= 10:
            rho, p = spearmanr(x[valid], y[valid])
        else:
            rho, p = np.nan, np.nan
        rows.append(
            {"metric": col, "spearman_rho_vs_depmap_median": rho, "p_value": p, "n": n}
        )

    return (
        pd.DataFrame(rows)
        .sort_values("spearman_rho_vs_depmap_median")
        .reset_index(drop=True)
    )


def compute_priority_scores(
    centralities: pd.DataFrame,
    depmap_median: pd.DataFrame,
    centrality_cols: list = None,
) -> pd.DataFrame:
    """
    Merge centralities with DepMap essentiality and compute a composite
    priority score for each gene.

    Parameters
    ----------
    centralities : pd.DataFrame
        Centrality table (must have a Gene/gene column).
    depmap_median : pd.DataFrame
        Per-gene DepMap median essentiality (Gene, depmap_median_all).
    centrality_cols : list, optional
        Columns to include in the priority calculation.

    Returns
    -------
    pd.DataFrame
        Sorted by ``priority_score`` (descending).  Columns:
        Gene, <centrality_cols>, depmap_median_all, priority_score.
    """
    if centrality_cols is None:
        centrality_cols = CENTRALITY_FEATURES

    cent = centralities.copy()
    if "gene" in cent.columns and "Gene" not in cent.columns:
        cent = cent.rename(columns={"gene": "Gene"})
    cent["Gene"] = cent["Gene"].astype(str)

    df = cent.merge(depmap_median, on="Gene", how="left")

    for col in centrality_cols:
        df[f"rank_{col}"] = _rank01(df[col], ascending=False)

    # More-negative DepMap median -> more essential -> higher priority
    df["rank_depmap_essential"] = _rank01(df["depmap_median_all"], ascending=True)

    rank_cols = [f"rank_{col}" for col in centrality_cols] + ["rank_depmap_essential"]
    df["priority_score"] = df[rank_cols].mean(axis=1, skipna=True)

    output_cols = ["Gene"] + centrality_cols + ["depmap_median_all", "priority_score"]
    return (
        df[output_cols]
        .sort_values("priority_score", ascending=False)
        .reset_index(drop=True)
    )


def run_prioritization(
    centralities: pd.DataFrame,
    depmap_median: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute correlations and priority scores, save outputs, and return the
    priority table.

    Parameters
    ----------
    centralities : pd.DataFrame
        Centrality table.
    depmap_median : pd.DataFrame
        Per-gene DepMap median essentiality.

    Returns
    -------
    pd.DataFrame
        Prioritized gene table.
    """
    corr_df = compute_depmap_correlations(centralities, depmap_median)
    corr_df.to_csv(DEPMAP_CORRELATIONS_CSV, index=False)
    logger.info("Saved DepMap correlations to %s", DEPMAP_CORRELATIONS_CSV)

    priority = compute_priority_scores(centralities, depmap_median)
    priority.to_csv(PRIORITIZED_GENES_CSV, index=False)
    logger.info("Saved prioritized genes to %s", PRIORITIZED_GENES_CSV)

    return priority
