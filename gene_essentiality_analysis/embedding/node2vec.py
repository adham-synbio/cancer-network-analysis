"""
Generate Node2Vec embeddings for a NetworkX graph.

Falls back to a manual random-walk + Word2Vec implementation when the
``node2vec`` package is unavailable.
"""

import logging
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from config import (
    NODE2VEC_DIM,
    NODE2VEC_WALK_LENGTH,
    NODE2VEC_NUM_WALKS,
    NODE2VEC_P,
    NODE2VEC_Q,
    NODE2VEC_WORKERS,
    NODE2VEC_EMBEDDINGS_CSV,
)

logger = logging.getLogger(__name__)


def _weighted_random_walk(
    G: nx.Graph,
    start,
    length: int,
    precomputed_neighbors: dict,
) -> list:
    """Perform a single biased random walk starting from *start*."""
    walk = [start]
    current = start
    for _ in range(length - 1):
        nbrs, probs = precomputed_neighbors.get(current, ([], None))
        if not nbrs:
            break
        current = np.random.choice(nbrs, p=probs)
        walk.append(current)
    return walk


def _build_neighbor_table(G: nx.Graph) -> dict:
    """Pre-compute weighted neighbor lists for all nodes."""
    neighbor_table = {}
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if not nbrs:
            neighbor_table[node] = ([], None)
            continue
        weights = np.array(
            [G[node][nbr].get("weight", 1.0) for nbr in nbrs], dtype=float
        )
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        neighbor_table[node] = (nbrs, weights / weights.sum())
    return neighbor_table


def generate_embeddings(
    G: nx.Graph,
    dim: int = NODE2VEC_DIM,
    walk_length: int = NODE2VEC_WALK_LENGTH,
    num_walks: int = NODE2VEC_NUM_WALKS,
    p: float = NODE2VEC_P,
    q: float = NODE2VEC_Q,
    workers: int = NODE2VEC_WORKERS,
) -> pd.DataFrame:
    """
    Generate Node2Vec embeddings for every node in *G*.

    Uses the ``node2vec`` package when available, otherwise falls back to a
    simple weighted random-walk + gensim Word2Vec implementation.

    Parameters
    ----------
    G : nx.Graph
        Weighted undirected graph.
    dim : int
        Embedding dimensionality.
    walk_length : int
        Steps per random walk.
    num_walks : int
        Random walks per node.
    p : float
        Return parameter (BFS vs DFS balance).
    q : float
        In-out parameter (BFS vs DFS balance).
    workers : int
        Number of parallel workers for Word2Vec training.

    Returns
    -------
    pd.DataFrame
        Rows are nodes; columns are ``emb_0`` ... ``emb_{dim-1}``.
        An additional ``Gene`` column carries the node label.
    """
    nodes = list(G.nodes())
    emb_cols = [f"emb_{i}" for i in range(dim)]

    # Attempt to use the node2vec package
    try:
        from node2vec import Node2Vec  # type: ignore

        logger.info("Generating Node2Vec embeddings via node2vec package...")
        n2v = Node2Vec(
            G,
            dimensions=dim,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            weight_key="weight",
            quiet=True,
        )
        model = n2v.fit(window=10, min_count=1, batch_words=256, epochs=3)
        matrix = np.array([model.wv[str(n)] for n in nodes])

    except ImportError:
        logger.warning(
            "node2vec package not found; using manual random-walk fallback."
        )

        try:
            from gensim.models import Word2Vec  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Neither 'node2vec' nor 'gensim' is installed. "
                "Install at least one: pip install node2vec gensim"
            ) from exc

        logger.info("Building neighbor table...")
        neighbor_table = _build_neighbor_table(G)

        logger.info("Generating %d random walks of length %d...", num_walks * len(nodes), walk_length)
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = _weighted_random_walk(G, node, walk_length, neighbor_table)
                walks.append([str(n) for n in walk])

        logger.info("Training Word2Vec on %d walks...", len(walks))
        w2v = Word2Vec(
            sentences=walks,
            vector_size=dim,
            window=10,
            min_count=1,
            sg=1,
            workers=workers,
            epochs=3,
        )
        matrix = np.array([w2v.wv[str(n)] for n in nodes])

    df = pd.DataFrame(matrix, columns=emb_cols)
    df.insert(0, "Gene", nodes)
    logger.info("Embedding matrix shape: %s", df.shape)
    return df


def load_or_generate_embeddings(G: nx.Graph = None, **kwargs) -> pd.DataFrame:
    """
    Return embeddings from cache or generate them if the cache is absent.

    Parameters
    ----------
    G : nx.Graph, optional
        Required when the cache does not exist.
    **kwargs
        Forwarded to :func:`generate_embeddings`.

    Returns
    -------
    pd.DataFrame
        Embedding DataFrame with a ``Gene`` column.
    """
    if NODE2VEC_EMBEDDINGS_CSV.exists():
        logger.info("Loading cached embeddings from %s", NODE2VEC_EMBEDDINGS_CSV)
        return pd.read_csv(NODE2VEC_EMBEDDINGS_CSV)

    if G is None:
        raise ValueError(
            "Graph G must be supplied when the embedding cache does not exist."
        )

    emb = generate_embeddings(G, **kwargs)
    emb.to_csv(NODE2VEC_EMBEDDINGS_CSV, index=False)
    logger.info("Saved embeddings to %s", NODE2VEC_EMBEDDINGS_CSV)
    return emb
