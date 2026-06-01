import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "outputs/tables/error_analysis_table.csv"
FIG1  = ROOT / "outputs/figures/error_type_counts.png"
FIG2  = ROOT / "outputs/figures/hallucination_risk_by_dataset.png"
FIG1.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TABLE)

# Plot 1 — Error type counts
all_errors = []
for val in df["error_types"].dropna():
    all_errors.extend([e.strip() for e in val.split(";") if e.strip()])
error_counts = Counter(all_errors)

if error_counts:
    labels, values = zip(*sorted(error_counts.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title(f"LLM Reasoning Error Types (n={len(df)})", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_xlabel("Error Type")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(FIG1, dpi=150)
    plt.close()
    print(f"Saved: {FIG1}")
else:
    print("WARNING: no error types found in table — plot 1 skipped")

# Plot 2 — Hallucination risk by dataset
RISK_COLORS = {"low": "#55A868", "medium": "#4C72B0", "high": "#C44E52"}
datasets = sorted(df["source_dataset"].unique())
risk_levels = ["low", "medium", "high"]
x = range(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for j, risk in enumerate(risk_levels):
    counts = [len(df[(df["source_dataset"] == ds) & (df["hallucination_risk"] == risk)]) for ds in datasets]
    ax.bar([i + j * width for i in x], counts, width, label=risk, color=RISK_COLORS[risk])

ax.set_xticks([i + width for i in x])
ax.set_xticklabels(datasets)
ax.set_title(f"Hallucination Risk by Dataset (n={len(df)})", fontsize=13)
ax.set_ylabel("Count")
ax.legend(title="Risk Level")
plt.tight_layout()
fig.savefig(FIG2, dpi=150)
plt.close()
print(f"Saved: {FIG2}")
