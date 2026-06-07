"""
Evaluation utilities: metrics, ROC curves, confusion matrices, comparison report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve, classification_report,
)
import mlflow
import logging

logger = logging.getLogger(__name__)

METRICS_ORDER = ["roc_auc", "f1", "precision", "recall", "accuracy"]


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    }


def plot_roc_curve(y_true, y_prob, model_name: str, save_path: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}", color="#2563EB")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_path, f"roc_{model_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(save_path, f"cm_{model_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_tournament_comparison(results: dict, save_path: str) -> str:
    """Bar chart comparing all models across all metrics."""
    df = pd.DataFrame(results).T[METRICS_ORDER]

    fig, axes = plt.subplots(1, len(METRICS_ORDER), figsize=(18, 5))
    colors = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]

    for i, metric in enumerate(METRICS_ORDER):
        vals = df[metric].sort_values(ascending=False)
        bars = axes[i].bar(range(len(vals)), vals.values,
                           color=colors[:len(vals)], edgecolor="white", linewidth=0.5)
        axes[i].set_title(metric.upper().replace("_", " "), fontweight="bold")
        axes[i].set_ylim(0, 1.05)
        axes[i].set_xticks(range(len(vals)))
        axes[i].set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
        axes[i].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Model Tournament — All Contestants Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_path, "tournament_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def generate_comparison_report(results: dict, champion: str, save_path: str) -> str:
    """Generate an HTML comparison report artifact."""
    df = pd.DataFrame(results).T[METRICS_ORDER].sort_values("roc_auc", ascending=False)

    rows = ""
    for model, row in df.iterrows():
        highlight = ' style="background:#dbeafe;font-weight:bold;"' if model == champion else ""
        badge = ' 🏆' if model == champion else ""
        rows += f"""
        <tr{highlight}>
            <td>{model}{badge}</td>
            {''.join(f'<td>{v:.4f}</td>' for v in row)}
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>MLOps Model Tournament — Results</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #1e293b; }}
  h1 {{ color: #1e40af; }} h2 {{ color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom:6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top:16px; }}
  th {{ background: #1e40af; color: white; padding: 10px; text-align:center; }}
  td {{ padding: 9px 12px; text-align:center; border-bottom: 1px solid #e5e7eb; }}
  tr:hover {{ background: #f8fafc; }}
  .champion {{ background: #2563eb; color:white; padding:6px 16px; border-radius:20px; font-weight:bold; }}
  .metric {{ display:inline-block; margin:6px; background:#f1f5f9; padding:6px 14px; border-radius:8px; }}
  img {{ max-width:100%; margin-top:20px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>🏆 MLOps Model Tournament — Results</h1>
<p>Dataset: UCI Bank Marketing (synthetic mirror) | Champion selected by: <strong>ROC-AUC</strong></p>

<h2>Champion</h2>
<p><span class="champion">{champion}</span></p>
{''.join(f'<span class="metric">{m.upper()}: {results[champion][m]:.4f}</span>' for m in METRICS_ORDER)}

<h2>All Contestants</h2>
<table>
  <tr><th>Model</th>{''.join(f'<th>{m.upper()}</th>' for m in METRICS_ORDER)}</tr>
  {rows}
</table>

<h2>Visual Comparison</h2>
<img src="tournament_comparison.png" alt="Tournament Comparison Chart">

<h2>Individual ROC Curves</h2>
{''.join(f'<img src="roc_{m}.png" alt="ROC {m}">' for m in results)}

<footer style="margin-top:40px;color:#94a3b8;font-size:12px;">
  Generated by mlops-model-tournament | github.com/jumma786
</footer>
</body>
</html>"""

    path = os.path.join(save_path, "tournament_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report saved: {path}")
    return path
