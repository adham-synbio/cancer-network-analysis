"""
Build a weighted NetworkX graph from a PPI edge table and extract the
largest connected component (LCC).
"""

import logging

import networkx as nx
import pandas as pd

from config import (
    NETWORK_GRAPHML,
    NETWORK_EDGES_CSV,
    NETWORK_NODES_CSV,
    STRING_PPI_THRESHOLD_HC,
)

logger = logging.getLogger(__name__)


def build_weighted_graph(
    edges: pd.DataFrame,
    threshold: float = None,
) -> nx.Graph:
    """
    Construct an undirected weighted graph from an edge DataFrame.

    When multiple rows describe the same gene pair, the maximum weight is
    retained.

    Parameters
    ----------
    edges : pd.DataFrame
        Must contain columns: gene1, gene2, weight.
        Optionally contains combined_score for secondary filtering.
    threshold : float, optional
        If provided and a ``combined_score`` column exists, only edges with
        ``combined_score >= threshold`` are included.

    Returns
    -------
    nx.Graph
        Graph with ``weight`` edge attributes.
    """
    df = edges.copy()
    if threshold is not None and "combined_score" in df.columns:
        df = df[df["combined_score"] >= threshold]
        logger.info("Edges after threshold >= %s: %d", threshold, len(df))

    G = nx.Graph()
    for row in df[["gene1", "gene2", "weight"]].itertuples(index=False):
        g1, g2, w = row
        if g1 == g2:
            continue
        if G.has_edge(g1, g2):
            if w > G[g1][g2].get("weight", 0):
                G[g1][g2]["weight"] = float(w)
        else:
            G.add_edge(g1, g2, weight=float(w))

    logger.info(
        "Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


def largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Return the largest connected component of *G* as a standalone graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    nx.Graph
        Subgraph restricted to the LCC.
    """
    if G.number_of_edges() == 0:
        logger.warning("Graph has no edges; returning as-is.")
        return G
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()
    logger.info(
        "LCC: %d nodes, %d edges", G_lcc.number_of_nodes(), G_lcc.number_of_edges()
    )
    return G_lcc


def save_graph(G: nx.Graph) -> None:
    """
    Persist a graph as GraphML, an edge CSV, and a node CSV.

    Parameters
    ----------
    G : nx.Graph
        Graph to save.
    """
    nx.write_graphml(G, NETWORK_GRAPHML)
    nx.to_pandas_edgelist(G).to_csv(NETWORK_EDGES_CSV, index=False)
    pd.DataFrame({"gene": list(G.nodes())}).to_csv(NETWORK_NODES_CSV, index=False)
    logger.info(
        "Saved graph to %s, %s, %s",
        NETWORK_GRAPHML,
        NETWORK_EDGES_CSV,
        NETWORK_NODES_CSV,
    )


def load_graph() -> nx.Graph:
    """
    Load the saved weighted PPI graph from disk.

    Returns
    -------
    nx.Graph
        Graph with edge ``weight`` attributes as floats.

    Raises
    ------
    FileNotFoundError
        If the GraphML file does not exist.
    """
    if not NETWORK_GRAPHML.exists():
        raise FileNotFoundError(
            f"Network file not found: {NETWORK_GRAPHML}\n"
            "Run the build_network script first."
        )

    G = nx.read_graphml(NETWORK_GRAPHML)
    for u, v, data in G.edges(data=True):
        try:
            w = float(data.get("weight", 1.0))
            data["weight"] = w if (w > 0 and w == w) else 1.0  # reject 0, inf, nan
        except (TypeError, ValueError):
            data["weight"] = 1.0

    logger.info(
        "Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


def build_string_only_graph(ppi_csv_path, threshold: int = STRING_PPI_THRESHOLD_HC) -> nx.Graph:
    """
    Build and return a STRING-confidence-weighted LCC graph without co-expression.

    Parameters
    ----------
    ppi_csv_path : str or Path
        Path to the processed STRING PPI CSV (gene1, gene2, combined_score).
    threshold : int
        Minimum combined_score to retain an edge.

    Returns
    -------
    nx.Graph
        Largest connected component with normalized STRING weight attribute.
    """
    ppi = pd.read_csv(ppi_csv_path)
    ppi = ppi[ppi["combined_score"] >= threshold].copy()
    s = ppi["combined_score"].astype(float)
    ppi["weight"] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    G = build_weighted_graph(ppi)
    G_lcc = largest_connected_component(G)
    return G_lcc
