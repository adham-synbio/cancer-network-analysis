"""
Step 3 – Compute network centrality metrics.

Loads the saved PPI graph and computes:
    degree centrality, weighted strength, betweenness centrality (approximate),
    closeness centrality, eigenvector centrality, weighted clustering coefficient.

Results are saved to results/centralities_string_only.csv.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gene_essentiality_analysis.network.build import load_graph
from gene_essentiality_analysis.network.centrality import compute_centralities
from config import CENTRALITIES_CSV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading weighted PPI graph...")
    G = load_graph()

    logger.info("Computing centrality metrics...")
    cent = compute_centralities(G)

    cent.to_csv(CENTRALITIES_CSV, index=False)
    logger.info("Saved centralities to %s", CENTRALITIES_CSV)
    logger.info("Top 5 genes by degree centrality:\n%s", cent.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
