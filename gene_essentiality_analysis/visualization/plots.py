"""
Publication-quality figures for the TNBC network analysis pipeline.

All functions accept DataFrames as inputs rather than reading from disk
so they can be used independently of the pipeline.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import (
    CENTRALITY_FEATURES,
    ESSENTIALITY_THRESHOLD,
    CENTRALITY_TOP_QUANTILE,
    FIGURE_NETWORK_VIZ,
    FIGURE_CLASSIFIER,
)

logger = logging.getLogger(__name__)

sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Figure 1 – Network analysis overview
# ---------------------------------------------------------------------------

def plot_network_overview(
    centralities: pd.DataFrame,
    prioritized: pd.DataFrame,
    output_path=None,
) -> plt.Figure:
    """
    Four-panel figure summarizing the network analysis.

    Panels:
    (a) Distribution of essential vs non-essential genes (DepMap histogram)
    (b) Distribution of central vs non-central genes (degree centrality)
    (c) 2x2 contingency heatmap (central x essential)
    (d) Priority score vs DepMap essentiality scatter

    Parameters
    ----------
    centralities : pd.DataFrame
        Centrality table (must contain ``degree_centrality``).
    prioritized : pd.DataFrame
        Prioritized gene table (must contain ``depmap_median_all``,
        ``degree_centrality``, ``priority_score``).
    output_path : str or Path, optional
        File path to save the figure.  Defaults to ``FIGURE_NETWORK_VIZ``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None:
        output_path = FIGURE_NETWORK_VIZ

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # (a) Essential vs non-essential
    ax = axes[0, 0]
    ax.set_title("(a) Distribution: Essential vs Non-Essential Genes", fontweight="bold")
    data = prioritized.dropna(subset=["depmap_median_all"])
    essential = data[data["depmap_median_all"] < ESSENTIALITY_THRESHOLD]
    non_essential = data[data["depmap_median_all"] >= ESSENTIALITY_THRESHOLD]
    bins = np.linspace(data["depmap_median_all"].min(), data["depmap_median_all"].max(), 50)
    ax.hist(essential["depmap_median_all"], bins=bins, alpha=0.7, color="red", density=True,
            label=f"Essential (n={len(essential)})")
    ax.hist(non_essential["depmap_median_all"], bins=bins, alpha=0.7, color="blue", density=True,
            label=f"Non-essential (n={len(non_essential)})")
    ax.axvline(ESSENTIALITY_THRESHOLD, color="black", linestyle="--",
               label=f"Threshold ({ESSENTIALITY_THRESHOLD})")
    ax.set_xlabel("DepMap Median Essentiality Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Central vs non-central
    ax = axes[0, 1]
    ax.set_title("(b) Distribution: Central vs Non-Central Genes", fontweight="bold")
    cdata = centralities.dropna(subset=["degree_centrality"])
    threshold_c = cdata["degree_centrality"].quantile(CENTRALITY_TOP_QUANTILE)
    central = cdata[cdata["degree_centrality"] >= threshold_c]
    non_central = cdata[cdata["degree_centrality"] < threshold_c]
    bins_c = np.linspace(cdata["degree_centrality"].min(), cdata["degree_centrality"].max(), 50)
    ax.hist(central["degree_centrality"], bins=bins_c, alpha=0.7, color="green", density=True,
            label=f"Central (n={len(central)})")
    ax.hist(non_central["degree_centrality"], bins=bins_c, alpha=0.7, color="orange", density=True,
            label=f"Non-central (n={len(non_central)})")
    ax.axvline(threshold_c, color="black", linestyle="--", label="75th percentile")
    ax.set_xlabel("Degree Centrality")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) 2x2 contingency heatmap
    ax = axes[1, 0]
    ax.set_title("(c) Central vs Essential Gene Categories", fontweight="bold")
    merged = prioritized.dropna(subset=["depmap_median_all", "degree_centrality"]).copy()
    if len(merged) > 0:
        merged["is_essential"] = merged["depmap_median_all"] < ESSENTIALITY_THRESHOLD
        merged["is_central"] = merged["degree_centrality"] >= threshold_c
        matrix = np.array([
            [len(merged[merged["is_central"] & merged["is_essential"]]),
             len(merged[merged["is_central"] & ~merged["is_essential"]])],
            [len(merged[~merged["is_central"] & merged["is_essential"]]),
             len(merged[~merged["is_central"] & ~merged["is_essential"]])],
        ])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Essential", "Non-essential"],
                    yticklabels=["Central", "Non-central"],
                    ax=ax, cbar_kws={"shrink": 0.8})
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                pct = (matrix[i, j] / total) * 100
                ax.text(j + 0.5, i + 0.7, f"({pct:.1f}%)",
                        ha="center", va="center", fontsize=10, color="red", fontweight="bold")
        ax.set_ylabel("Centrality Category")
        ax.set_xlabel("Essentiality Category")

    # (d) Priority score vs DepMap scatter
    ax = axes[1, 1]
    ax.set_title("(d) Priority Score vs DepMap Essentiality", fontweight="bold")
    valid = prioritized.dropna(subset=["priority_score", "depmap_median_all"])
    if len(valid) > 0:
        ax.scatter(valid["priority_score"], valid["depmap_median_all"],
                   alpha=0.6, s=20, c="purple", edgecolors="black", linewidth=0.5)
        if len(valid) > 1:
            z = np.polyfit(valid["priority_score"], valid["depmap_median_all"], 1)
            p_fit = np.poly1d(z)
            ax.plot(valid["priority_score"], p_fit(valid["priority_score"]),
                    "r--", alpha=0.8, linewidth=2, label="Trend line")
            rho, p_val = spearmanr(valid["priority_score"], valid["depmap_median_all"])
            ax.text(0.05, 0.95,
                    f"rho = {rho:.3f}\np = {p_val:.3e}\nn = {len(valid)}",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    verticalalignment="top", fontsize=10)
        ax.axhline(ESSENTIALITY_THRESHOLD, color="red", linestyle="--", alpha=0.7,
                   label=f"Essentiality threshold ({ESSENTIALITY_THRESHOLD})")
        ax.set_xlabel("Priority Score")
        ax.set_ylabel("DepMap Median Essentiality Score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved Figure 1 to %s", output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – Classifier performance comparison
# ---------------------------------------------------------------------------

def _pr_roc(labels: np.ndarray, scores: np.ndarray):
    """Compute PR and ROC curve data plus AUPRC and AUROC."""
    mask = ~np.isnan(scores)
    l, s = labels[mask], scores[mask]
    prec, rec, _ = precision_recall_curve(l, s)
    auprc = average_precision_score(l, s)
    fpr, tpr, _ = roc_curve(l, s)
    auroc = auc(fpr, tpr)
    return prec, rec, fpr, tpr, auprc, auroc


def plot_classifier_comparison(
    oof_centrality: pd.DataFrame,
    oof_node2vec: pd.DataFrame,
    oof_combined: pd.DataFrame,
    output_path=None,
) -> plt.Figure:
    """
    Four-panel classifier comparison figure.

    Panels:
    (a) Precision-Recall curves for all three models
    (b) ROC curves for all three models
    (c) Score density by label (combined model)
    (d) Precision@K curve (combined model)

    Parameters
    ----------
    oof_centrality : pd.DataFrame
        OOF predictions for the centrality-only model.
        Must contain: y_true, oof_prob_xgb.
    oof_node2vec : pd.DataFrame
        OOF predictions for the node2vec-only model.
    oof_combined : pd.DataFrame
        OOF predictions for the centrality + node2vec model.
    output_path : str or Path, optional
        Defaults to ``FIGURE_CLASSIFIER``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if output_path is None:
        output_path = FIGURE_CLASSIFIER

    prec_c, rec_c, fpr_c, tpr_c, auprc_c, roc_c = _pr_roc(
        oof_centrality["y_true"].values, oof_centrality["oof_prob_xgb"].values
    )
    prec_e, rec_e, fpr_e, tpr_e, auprc_e, roc_e = _pr_roc(
        oof_node2vec["y_true"].values, oof_node2vec["oof_prob_xgb"].values
    )
    prec_m, rec_m, fpr_m, tpr_m, auprc_m, roc_m = _pr_roc(
        oof_combined["y_true"].values, oof_combined["oof_prob_xgb"].values
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    colors = {"centrality": "#C44E52", "node2vec": "#55A868", "combined": "#4C72B0"}

    # (a) PR curves
    ax = axs[0, 0]
    ax.plot(rec_c, prec_c, color=colors["centrality"], lw=2,
            label=f"Centrality only (AUPRC = {auprc_c:.3f})")
    ax.plot(rec_e, prec_e, color=colors["node2vec"], lw=2,
            label=f"Node2vec only (AUPRC = {auprc_e:.3f})")
    ax.plot(rec_m, prec_m, color=colors["combined"], lw=2,
            label=f"Centrality + Node2vec (AUPRC = {auprc_m:.3f})")
    ax.set_title("(a) Precision-Recall Comparison (5-Fold CV)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # (b) ROC curves
    ax = axs[0, 1]
    ax.plot(fpr_c, tpr_c, color=colors["centrality"], lw=2,
            label=f"Centrality only (AUROC = {roc_c:.3f})")
    ax.plot(fpr_e, tpr_e, color=colors["node2vec"], lw=2,
            label=f"Node2vec only (AUROC = {roc_e:.3f})")
    ax.plot(fpr_m, tpr_m, color=colors["combined"], lw=2,
            label=f"Centrality + Node2vec (AUROC = {roc_m:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("(b) ROC Comparison (5-Fold CV)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # (c) Score density by label (combined model)
    ax = axs[1, 0]
    tmp = pd.DataFrame({
        "score": oof_combined["oof_prob_xgb"].values,
        "label": np.where(
            oof_combined["y_true"] == 1, "Essential (top 10%)", "Non-essential"
        ),
    })
    sns.kdeplot(data=tmp, x="score", hue="label", ax=ax,
                fill=True, common_norm=False,
                palette={"Essential (top 10%)": colors["centrality"],
                         "Non-essential": colors["node2vec"]})
    ax.set_title("(c) Score Distribution (Centrality + Node2Vec)")
    ax.set_xlabel("Predicted P(essential)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

    # (d) Precision@K
    ax = axs[1, 1]
    k_values = [50, 100, 200, 500, 1000, 1500, 2000]
    prec_k = [
        _precision_at_k(oof_combined["oof_prob_xgb"].values,
                        oof_combined["y_true"].values, k)
        for k in k_values
    ]
    ax.plot(k_values, prec_k, "o-", linewidth=2, markersize=6, color=colors["node2vec"])
    for i, (k, p) in enumerate(zip(k_values, prec_k)):
        if i % 2 == 0:
            ax.annotate(f"{p:.3f}", (k, p), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("K (Top K Predictions)")
    ax.set_ylabel("Precision@K")
    ax.set_title("(d) Precision@K (Centrality + Node2Vec)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved Figure 2 to %s", output_path)
    return fig


def _precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    mask = ~np.isnan(scores)
    s, l = scores[mask], labels[mask]
    idx = np.argsort(s)[::-1][: min(k, len(s))]
    return float(l[idx].mean()) if len(idx) > 0 else np.nan
