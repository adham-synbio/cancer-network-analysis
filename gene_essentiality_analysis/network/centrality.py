"""
Compute centrality metrics on a weighted NetworkX graph.

Metrics computed:
- Degree centrality (unweighted)
- Strength (weighted degree), normalized
- Betweenness centrality (approximate)
- Closeness centrality
- Eigenvector centrality (weighted)
- Weighted clustering coefficient
"""

import logging

import networkx as nx
import numpy as np
import pandas as pd

from config import BETWEENNESS_K, RANDOM_STATE, CENTRALITIES_CSV

logger = logging.getLogger(__name__)


def compute_centralities(G: nx.Graph, betweenness_k: int = BETWEENNESS_K) -> pd.DataFrame:
    """
    Compute a standard set of centrality metrics for every node in *G*.

    Parameters
    ----------
    G : nx.Graph
        Weighted undirected graph with a ``weight`` edge attribute.
    betweenness_k : int
        Number of pivot nodes for approximate betweenness centrality.
        Smaller values run faster; larger values are more accurate.

    Returns
    -------
    pd.DataFrame
        One row per node with columns:
        gene, degree_centrality, strength_norm, betweenness_centrality,
        closeness_centrality, eigenvector_centrality, clustering_coefficient.
    """
    logger.info("Computing degree centrality...")
    deg_c = nx.degree_centrality(G)

    logger.info("Computing weighted strength...")
    strength = {
        n: float(sum(d.get("weight", 1.0) for _, _, d in G.edges(n, data=True)))
        for n in G.nodes()
    }
    max_strength = max(strength.values()) if strength else 1.0
    strength_norm = {n: v / max_strength for n, v in strength.items()}

    k = min(betweenness_k, G.number_of_nodes())
    logger.info("Computing betweenness centrality (k=%d)...", k)
    bet_c = nx.betweenness_centrality(G, k=k, normalized=True, seed=RANDOM_STATE)

    logger.info("Computing closeness centrality...")
    close_c = nx.closeness_centrality(G)

    logger.info("Computing eigenvector centrality...")
    try:
        eig_c = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        logger.warning(
            "eigenvector_centrality_numpy failed; falling back to power iteration."
        )
        eig_c = nx.eigenvector_centrality(
            G, max_iter=500, tol=1e-6, weight="weight"
        )

    logger.info("Computing weighted clustering coefficient...")
    clust_c = nx.clustering(G, weight="weight")

    nodes = list(G.nodes())
    centralities = pd.DataFrame({"gene": nodes})
    centralities["degree_centrality"]     = centralities["gene"].map(deg_c)
    centralities["strength_norm"]         = centralities["gene"].map(strength_norm)
    centralities["betweenness_centrality"] = centralities["gene"].map(bet_c)
    centralities["closeness_centrality"]  = centralities["gene"].map(close_c)
    centralities["eigenvector_centrality"] = centralities["gene"].map(eig_c)
    centralities["clustering_coefficient"] = centralities["gene"].map(clust_c)

    logger.info("Centrality computation complete for %d nodes.", len(centralities))
    return centralities.sort_values("degree_centrality", ascending=False).reset_index(drop=True)


def load_or_compute_centralities(G: nx.Graph = None, **kwargs) -> pd.DataFrame:
    """
    Return the centrality table, recomputing and caching if necessary.

    Parameters
    ----------
    G : nx.Graph, optional
        Required if the cached CSV does not exist.
    **kwargs
        Forwarded to :func:`compute_centralities`.

    Returns
    -------
    pd.DataFrame
        Centrality table with a ``gene`` column.
    """
    if CENTRALITIES_CSV.exists():
        logger.info("Loading cached centralities from %s", CENTRALITIES_CSV)
        return pd.read_csv(CENTRALITIES_CSV)

    if G is None:
        raise ValueError(
            "Graph G must be supplied when the centrality cache does not exist."
        )

    cent = compute_centralities(G, **kwargs)
    cent.to_csv(CENTRALITIES_CSV, index=False)
    logger.info("Saved centralities to %s", CENTRALITIES_CSV)
    return cent
