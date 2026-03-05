"""
Step 4 – Prioritize candidate therapeutic targets.

Merges network centrality metrics with DepMap CRISPR essentiality scores,
computes Spearman correlations between each centrality and essentiality,
and produces a ranked gene list.

Outputs:
    results/centrality_depmap_correlations.csv
    results/prioritized_genes_string_only_vs_depmap.csv
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import CENTRALITIES_CSV
from gene_essentiality_analysis.data.depmap import compute_median_essentiality
from gene_essentiality_analysis.prioritization.scoring import run_prioritization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading centralities...")
    centralities = pd.read_csv(CENTRALITIES_CSV)
    centralities = centralities.rename(columns={"gene": "Gene"}) if "gene" in centralities.columns else centralities

    logger.info("Computing DepMap median essentiality...")
    depmap_median = compute_median_essentiality()

    logger.info("Running gene prioritization...")
    priority = run_prioritization(centralities, depmap_median)

    logger.info("Top 10 prioritized genes:\n%s", priority.head(10)[["Gene", "priority_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
