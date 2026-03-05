# Cancer Network Analysis

A modular Python pipeline for identifying candidate therapeutic targets in
Cancer gene essetiality analysis using protein-protein interaction
networks, graph embedding, gene expression, and machine learning.

## Overview

The pipeline integrates:

- **STRING v12** – high-confidence human PPI network
- **DepMap CRISPR** – genome-wide essentiality scores for target validation
- **Node2Vec** – graph embeddings capturing network topology
- **XGBoost + Logistic Regression** – binary classifiers predicting essential genes
- **SHAP** – feature importance and model interpretation

## Repository Structure

```
cancer-network-analysis/
├── config.py                         # All paths, thresholds, and hyperparameters
├── requirements.txt
├── scripts/                          # Numbered pipeline scripts (run in order)
│   ├── 01_download_data.py
│   ├── 02_build_network.py
│   ├── 03_compute_centralities.py
│   ├── 04_prioritize_genes.py
│   ├── 05_generate_embeddings.py
│   ├── 06_train_classifiers.py
│   └── 07_visualize.py
├── gene_essentiality_analysis/                    # Importable package
│   ├── data/
│   │   ├── download.py               # Generic file downloader with retry
│   │   ├── string_loader.py          # STRING PPI download and processing
│   │   ├── expression.py             # METABRIC and TCGA expression loading
│   │   └── depmap.py                 # DepMap CRISPR aggregation
│   ├── network/
│   │   ├── coexpression.py           # Spearman co-expression weights
│   │   ├── build.py                  # Graph construction and LCC extraction
│   │   └── centrality.py             # Centrality metric computation
│   ├── embedding/
│   │   └── node2vec.py               # Node2Vec embeddings (with fallback)
│   ├── models/
│   │   ├── classifiers.py            # Binary classification + SHAP
│   │   └── regression.py             # Regression (ElasticNet + XGBoost)
│   ├── prioritization/
│   │   └── scoring.py                # DepMap integration and priority scoring
│   └── visualization/
│       └── plots.py                  # Publication figures
├── data/                             # Not committed; created on first run
│   ├── raw/
│   ├── processed/
│   └── networks/
└── results/                          # Not committed; created on first run
    ├── figures/
    └── tables/
```

## Installation

```bash
git clone https://github.com/adham-synbio/tnbc-network-analysis.git
cd tnbc-network-analysis
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Data Requirements

### Automated downloads (handled by `scripts/01_download_data.py`)
| Dataset | Source |
|---------|--------|
| STRING v12 PPI (H. sapiens) | stringdb-downloads.org |
| METABRIC expression + clinical | cBioPortal |

### Manual downloads (place in `data/raw/`)
| File | Source | Notes |
|------|--------|-------|
| `CRISPRGeneEffect.csv` | [DepMap portal](https://depmap.org/portal/download/) | CRISPR gene effect |
| `tcga_brca_clinical.txt` | [GDC/cBioPortal](https://portal.gdc.cancer.gov/) | TCGA BRCA clinical |
| TCGA Firehose `.tar.gz` | [Broad GDAC Firehose](https://gdac.broadinstitute.org/) | RSEM normalized expression (optional; METABRIC used by default) |

## Usage

Run scripts in order from the repository root:

```bash
python scripts/01_download_data.py
python scripts/02_build_network.py
python scripts/03_compute_centralities.py
python scripts/04_prioritize_genes.py
python scripts/05_generate_embeddings.py
python scripts/06_train_classifiers.py
python scripts/07_visualize.py
```

Each script is idempotent: intermediate outputs are cached on disk and
reloaded on subsequent runs.

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STRING_PPI_THRESHOLD` | 700 | Minimum STRING combined_score |
| `STRING_PPI_THRESHOLD_HC` | 800 | Higher-confidence threshold |
| `ESSENTIALITY_PERCENTILE` | 10 | Top N% essential genes = positive class |
| `NODE2VEC_DIM` | 128 | Embedding dimensionality |
| `BETWEENNESS_K` | 500 | Pivots for approximate betweenness |
| `CV_N_SPLITS` | 5 | Cross-validation folds |

## Key Outputs

| File | Description |
|------|-------------|
| `data/networks/tnbc_ppi_weighted.graphml` | Weighted PPI graph |
| `results/centralities_string_only.csv` | Per-gene centrality metrics |
| `results/prioritized_genes_string_only_vs_depmap.csv` | Ranked target list |
| `results/node2vec_embeddings.csv` | 128-dim gene embeddings |
| `results/classifier_oof_predictions_centrality_plus_node2vec.csv` | OOF predictions |
| `results/model_comparison_centrality_vs_combined.csv` | AUROC / AUPRC summary |
| `results/shap_feature_importance_centrality_plus_node2vec.csv` | SHAP importances |
| `results/figures/Fig2_classifier_performance_comparison.png` | Figure 2 |

## Environment Variable

Set `CANCER_PROJECT_ROOT` to override the default project root (the repository
directory):

```bash
export CANCER_PROJECT_ROOT=/path/to/your/data
python scripts/02_build_network.py
```
