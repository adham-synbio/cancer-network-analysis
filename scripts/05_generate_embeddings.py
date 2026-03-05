"""
Step 5 – Generate Node2Vec graph embeddings.

Loads the weighted PPI graph and trains a Node2Vec model to produce
128-dimensional embeddings for every gene node.

Output:
    results/node2vec_embeddings.csv
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gene_essentiality_analysis.network.build import load_graph
from gene_essentiality_analysis.embedding.node2vec import load_or_generate_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading PPI graph...")
    G = load_graph()

    logger.info("Generating Node2Vec embeddings...")
    emb = load_or_generate_embeddings(G)

    logger.info("Embeddings shape: %s", emb.shape)
    logger.info("Saved to results/node2vec_embeddings.csv")


if __name__ == "__main__":
    main()
