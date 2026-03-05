"""
Step 7 – Generate publication figures.

Figure 1:  Network analysis overview (distribution plots, contingency heatmap,
           priority score scatter).
Figure 2:  Classifier performance comparison (PR curves, ROC curves,
           score density, Precision@K).

Outputs:
    results/figures/network_analysis_visualization.png
    results/figures/Fig2_classifier_performance_comparison.png
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (
    CENTRALITIES_CSV,
    PRIORITIZED_GENES_CSV,
    OOF_CENTRALITY_CSV,
    OOF_NODE2VEC_CSV,
    OOF_COMBINED_CSV,
)
from gene_essentiality_analysis.visualization.plots import (
    plot_network_overview,
    plot_classifier_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading data for Figure 1...")
    centralities = pd.read_csv(CENTRALITIES_CSV)
    prioritized = pd.read_csv(PRIORITIZED_GENES_CSV)

    logger.info("Plotting Figure 1...")
    plot_network_overview(centralities, prioritized)

    logger.info("Loading OOF predictions for Figure 2...")
    oof_centrality = pd.read_csv(OOF_CENTRALITY_CSV)
    oof_node2vec = pd.read_csv(OOF_NODE2VEC_CSV)
    oof_combined = pd.read_csv(OOF_COMBINED_CSV)

    logger.info("Plotting Figure 2...")
    plot_classifier_comparison(oof_centrality, oof_node2vec, oof_combined)

    logger.info("All figures saved.")


if __name__ == "__main__":
    main()
