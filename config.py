"""
Central configuration for the TNBC network analysis pipeline.

All paths, thresholds, and hyperparameters are defined here so they can be
changed in one place without editing individual modules.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("TNBC_PROJECT_ROOT", Path(__file__).parent))

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_RAW_DIR        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
DATA_NETWORKS_DIR   = PROJECT_ROOT / "data" / "networks"
RESULTS_DIR         = PROJECT_ROOT / "results"
FIGURES_DIR         = RESULTS_DIR / "figures"
TABLES_DIR          = RESULTS_DIR / "tables"

# Ensure all directories exist at import time
for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_NETWORKS_DIR,
           RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# STRING database
# ---------------------------------------------------------------------------
STRING_SPECIES          = "9606"  # Homo sapiens
STRING_VERSION          = "v12.0"
STRING_BASE_URL         = "https://stringdb-downloads.org/download"
STRING_PPI_THRESHOLD    = 700     # combined_score cutoff for high-confidence edges
STRING_PPI_THRESHOLD_HC = 800     # higher-confidence threshold for network building

# Raw download paths
STRING_PPI_GZ   = DATA_RAW_DIR / "string12_physical_links_9606.txt.gz"
STRING_INFO_GZ  = DATA_RAW_DIR / "string12_protein_info_9606.txt.gz"

# Processed output
STRING_PPI_CSV  = DATA_PROCESSED_DIR / "string_ppi_hc.csv"

# ---------------------------------------------------------------------------
# METABRIC
# ---------------------------------------------------------------------------
METABRIC_EXPR_URL = (
    "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric_expression.csv.gz"
)
METABRIC_CLIN_URL = (
    "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric_clinical_data.tsv"
)
METABRIC_EXPR_GZ  = DATA_RAW_DIR / "metabric_expr.csv.gz"
METABRIC_CLIN_TSV = DATA_RAW_DIR / "metabric_clinical.tsv"
METABRIC_TNBC_CSV = DATA_PROCESSED_DIR / "tnbc_log_expr_metabric.csv"

# ---------------------------------------------------------------------------
# TCGA BRCA (Firehose)
# ---------------------------------------------------------------------------
TCGA_EXPR_TAR = DATA_RAW_DIR / (
    "gdac.broadinstitute.org_BRCA.Merge_rnaseqv2__illuminahiseq_rnaseqv2"
    "__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz"
)
TCGA_CLIN_TXT       = DATA_RAW_DIR / "tcga_brca_clinical.txt"
TCGA_EXPR_FULL_CSV  = DATA_RAW_DIR / "tcga_brca_firehose_expr_full.csv"
TCGA_TNBC_EXPR_CSV  = DATA_PROCESSED_DIR / "tnbc_expr_matrix.csv"
TCGA_TNBC_SAMPLES   = DATA_PROCESSED_DIR / "tnbc_samples_used.csv"

# ---------------------------------------------------------------------------
# DepMap
# ---------------------------------------------------------------------------
DEPMAP_CRISPR_CSV = DATA_RAW_DIR / "CRISPRGeneEffect.csv"

# ---------------------------------------------------------------------------
# Network outputs
# ---------------------------------------------------------------------------
WEIGHTED_EDGES_CSV    = DATA_PROCESSED_DIR / "tnbc_weighted_ppi_edges.csv"
NETWORK_GRAPHML       = DATA_NETWORKS_DIR / "tnbc_ppi_weighted.graphml"
NETWORK_EDGES_CSV     = DATA_NETWORKS_DIR / "tnbc_ppi_weighted_edges.csv"
NETWORK_NODES_CSV     = DATA_NETWORKS_DIR / "tnbc_ppi_weighted_nodes.csv"

# ---------------------------------------------------------------------------
# Results outputs
# ---------------------------------------------------------------------------
CENTRALITIES_CSV            = RESULTS_DIR / "centralities_string_only.csv"
DEPMAP_CORRELATIONS_CSV     = RESULTS_DIR / "centrality_depmap_correlations.csv"
PRIORITIZED_GENES_CSV       = RESULTS_DIR / "prioritized_genes_string_only_vs_depmap.csv"
NODE2VEC_EMBEDDINGS_CSV     = RESULTS_DIR / "node2vec_embeddings.csv"

# Model prediction files
PREDICTIONS_REGRESSION_CSV  = RESULTS_DIR / "network_only_predictions_with_node2vec.csv"
CLASSIFIER_SCORES_CSV       = RESULTS_DIR / "classifier_scores_top10pct.csv"
CLASSIFIER_SCORES_SHAP_CSV  = RESULTS_DIR / "classifier_scores_top10pct_with_shap.csv"

# Out-of-fold predictions
OOF_CENTRALITY_CSV   = RESULTS_DIR / "classifier_oof_predictions_centrality_only.csv"
OOF_COMBINED_CSV     = RESULTS_DIR / "classifier_oof_predictions_centrality_plus_node2vec.csv"
OOF_NODE2VEC_CSV     = RESULTS_DIR / "classifier_oof_predictions_node2vec_only.csv"

# SHAP outputs
SHAP_IMPORTANCE_CSV  = RESULTS_DIR / "shap_feature_importance_classifier.csv"
SHAP_GROUPED_CSV     = RESULTS_DIR / "shap_grouped_importance.csv"
SHAP_EXPECTED_NPY    = RESULTS_DIR / "shap_expected_value_xgb.npy"

# Comparison outputs
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison_centrality_vs_combined.csv"
NODE2VEC_IMPORTANCE  = RESULTS_DIR / "node2vec_feature_importance.csv"

# Figure outputs
FIGURE_NETWORK_VIZ   = FIGURES_DIR / "network_analysis_visualization.png"
FIGURE_CLASSIFIER    = FIGURES_DIR / "Fig2_classifier_performance_comparison.png"

# ---------------------------------------------------------------------------
# Algorithm hyperparameters
# ---------------------------------------------------------------------------
# Co-expression
COEXPR_WEIGHT_STRING  = 0.5   # weight given to normalized STRING score
COEXPR_WEIGHT_SPEARMAN = 0.5  # weight given to positive Spearman rho

# Network
BETWEENNESS_K = 500  # number of pivots for approximate betweenness centrality

# Node2Vec
NODE2VEC_DIM         = 128
NODE2VEC_WALK_LENGTH = 80
NODE2VEC_NUM_WALKS   = 10
NODE2VEC_P           = 1.0
NODE2VEC_Q           = 1.0
NODE2VEC_WORKERS     = 4

# Classification
ESSENTIALITY_PERCENTILE = 10   # top N% most essential genes = positive class
ESSENTIALITY_THRESHOLD  = -0.5 # DepMap score cutoff for visualization
CENTRALITY_TOP_QUANTILE = 0.75 # degree centrality threshold for "central" label
CV_N_SPLITS = 5
RANDOM_STATE = 42

# XGBoost hyperparameters (shared across regressor and classifier)
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Network XGBoost (node2vec-only, lighter)
XGB_NODE2VEC_PARAMS = dict(
    objective="binary:logistic",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=9,
    random_state=RANDOM_STATE,
)

# Centrality feature names (fixed ordering used across modules)
CENTRALITY_FEATURES = [
    "degree_centrality",
    "strength_norm",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "clustering_coefficient",
]
