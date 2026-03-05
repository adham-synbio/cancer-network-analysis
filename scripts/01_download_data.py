"""
Step 1 – Download all remotely accessible data.

Data that requires a manual portal download (TCGA Firehose, DepMap) is
described in README.md.  This script handles the automated downloads:
    - STRING v12 PPI and protein info files
    - METABRIC expression and clinical data
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gene_essentiality_analysis.data.string_loader import download_string_data
from gene_essentiality_analysis.data.expression import download_metabric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Downloading STRING v12 PPI data...")
    download_string_data()

    logger.info("Downloading METABRIC expression and clinical data...")
    download_metabric()

    logger.info("All automated downloads complete.")


if __name__ == "__main__":
    main()
