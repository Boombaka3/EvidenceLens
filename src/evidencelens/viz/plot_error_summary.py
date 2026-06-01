import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Style utilities
# ---------------------------------------------------------------------------

def setup_plot_style():
    sns.set_style("whitegrid", {
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
    })
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16


def get_colors(n=6):
    return sns.color_palette("deep", n)


def ensure_output_dir(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


# ---------------------------------------------------------------------------
# Plot 1 — Error type counts (horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_error_type_counts(df, out_path):
    all_errors = []
    for val in df["error_types"].dropna():
        all_errors.extend([e.strip() for e in str(val).split(";") if e.strip()])

    if not all_errors:
        print("WARNING: no error types found in table — plot 1 skipped")
        return

    error_counts = Counter(all_errors)
    labels  = [k for k, _ in sorted(error_counts.items(), key=lambda x: x[1])]
    values  = [error_counts[k] for k in labels]
    total   = sum(values)
    colors  = get_colors(max(6, len(labels)))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, values, color=colors[:len(labels)])

    for bar, v in zip(bars, values):
        pct = 100 * v / total if total else 0
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{v} ({pct:.1f}%)",
            va="center", ha="left", fontsize=12,
        )

    ax.set_title(f"LLM Reasoning Error Types (n={len(df)})", pad=12)
    ax.set_xlabel("Count")
    ax.set_ylabel("Error Type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.3)

    plt.tight_layout()
    out = ensure_output_dir(out_path)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Hallucination risk by dataset (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_hallucination_risk_by_dataset(df, out_path):
    datasets    = ["SciFact", "PubMedQA", "QASPER"]
    risk_levels = ["low", "medium", "high"]
    colors      = get_colors(6)
    risk_colors = {
        "low":    colors[2],
        "medium": colors[0],
        "high":   colors[3],
    }

    x     = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for j, risk in enumerate(risk_levels):
        counts = [
            len(df[(df["source_dataset"] == ds) & (df["hallucination_risk"] == risk)])
            for ds in datasets
        ]
        offset = (j - 1) * width
        bars = ax.bar(x + offset, counts, width, label=risk, color=risk_colors[risk])

        for bar, v in zip(bars, counts):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(v),
                    ha="center", va="bottom", fontsize=12,
                )

    ax.set_title(f"Hallucination Risk by Dataset (n={len(df)})", pad=12)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(title="Risk Level")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = ensure_output_dir(out_path)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Support label distribution (horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_support_label_distribution(df, out_path):
    label_counts = Counter(df["support_label"].dropna().astype(str))
    if not label_counts:
        print("WARNING: no support_label values found — plot 3 skipped")
        return

    labels = [k for k, _ in sorted(label_counts.items(), key=lambda x: x[1])]
    values = [label_counts[k] for k in labels]
    total  = sum(values)
    colors = get_colors(max(6, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors[:len(labels)])

    for bar, v in zip(bars, values):
        pct = 100 * v / total if total else 0
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{v} ({pct:.1f}%)",
            va="center", ha="left", fontsize=12,
        )

    ax.set_title(f"LLM Answer Support Labels (n={len(df)})", pad=12)
    ax.set_xlabel("Count")
    ax.set_ylabel("Support Label")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.3)

    plt.tight_layout()
    out = ensure_output_dir(out_path)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Confidence vs hallucination risk heatmap
# ---------------------------------------------------------------------------

def plot_confidence_vs_risk_heatmap(df, out_path):
    sub = df[["llm_confidence", "hallucination_risk"]].dropna()
    if sub.empty:
        print("WARNING: no confidence/risk data — plot 4 skipped")
        return

    crosstab = pd.crosstab(sub["llm_confidence"], sub["hallucination_risk"])

    # Ensure consistent column order
    for col in ["low", "medium", "high"]:
        if col not in crosstab.columns:
            crosstab[col] = 0
    crosstab = crosstab[["low", "medium", "high"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Confidence vs Hallucination Risk (n={len(df)})", pad=12)
    ax.set_xlabel("Hallucination Risk")
    ax.set_ylabel("LLM Confidence")

    plt.tight_layout()
    out = ensure_output_dir(out_path)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

setup_plot_style()

ROOT  = Path(__file__).resolve().parents[3]
TABLE = ROOT / "outputs/tables/error_analysis_table.csv"
FIGS  = ROOT / "outputs/figures"

df = pd.read_csv(TABLE)

plot_error_type_counts(
    df,
    FIGS / "error_type_counts.png",
)
plot_hallucination_risk_by_dataset(
    df,
    FIGS / "hallucination_risk_by_dataset.png",
)
plot_support_label_distribution(
    df,
    FIGS / "support_label_distribution.png",
)
plot_confidence_vs_risk_heatmap(
    df,
    FIGS / "confidence_vs_risk_heatmap.png",
)
