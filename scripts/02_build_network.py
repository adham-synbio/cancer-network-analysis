"""
Step 2 – Build the weighted TNBC PPI network.

1. Load (or build) the high-confidence STRING PPI table.
2. Load METABRIC TNBC expression data.
3. Filter PPI to expressed genes and compute Spearman co-expression weights.
4. Build the weighted graph and extract the largest connected component.
5. Save the graph as GraphML plus edge/node CSV files.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gene_essentiality_analysis.data.string_loader import load_or_build_ppi
from gene_essentiality_analysis.data.expression import load_metabric_tnbc
from gene_essentiality_analysis.network.coexpression import load_or_compute_weights
from gene_essentiality_analysis.network.build import (
    build_weighted_graph,
    largest_connected_component,
    save_graph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading STRING PPI table...")
    ppi = load_or_build_ppi()

    logger.info("Loading METABRIC TNBC expression data...")
    expr_tnbc, _ = load_metabric_tnbc()

    logger.info("Computing co-expression weights...")
    ppi_w = load_or_compute_weights(ppi, expr_tnbc)

    logger.info("Building weighted graph...")
    G = build_weighted_graph(ppi_w)
    G_lcc = largest_connected_component(G)

    logger.info("Saving graph...")
    save_graph(G_lcc)

    logger.info(
        "Network built: %d nodes, %d edges (LCC).",
        G_lcc.number_of_nodes(),
        G_lcc.number_of_edges(),
    )


if __name__ == "__main__":
    main()
