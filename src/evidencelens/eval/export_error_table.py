from pathlib import Path
import jsonlines, csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
AUDIT_FILE   = ROOT / "outputs/predictions/evidence_audit_outputs.jsonl"
OUT_DETAIL   = ROOT / "outputs/tables/error_analysis_table.csv"
OUT_SUMMARY  = ROOT / "outputs/tables/error_summary_by_dataset.csv"
OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)

records = list(jsonlines.open(AUDIT_FILE))

DETAIL_COLS = [
    "id", "source_dataset", "task_type", "input_claim_or_question",
    "gold_label", "llm_answer", "llm_confidence", "support_label",
    "hallucination_risk", "limitation_preserved", "overgeneralization",
    "false_certainty", "error_types", "explanation",
]

with open(OUT_DETAIL, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=DETAIL_COLS)
    writer.writeheader()
    for r in records:
        row = {k: r.get(k, "") for k in DETAIL_COLS}
        row["error_types"] = "; ".join(r.get("error_types", []))
        writer.writerow(row)
print(f"Written: {OUT_DETAIL} ({len(records)} rows)")

datasets = sorted(set(r.get("source_dataset", "") for r in records))
summary_rows = []
for ds in datasets:
    subset = [r for r in records if r.get("source_dataset") == ds]
    all_errors = []
    for r in subset:
        all_errors.extend(r.get("error_types", []))
    top_errors = "; ".join(f"{k}:{v}" for k, v in Counter(all_errors).most_common(3))
    summary_rows.append({
        "source_dataset":    ds,
        "total":             len(subset),
        "high_risk_count":   sum(1 for r in subset if r.get("hallucination_risk") == "high"),
        "error_type_counts": top_errors,
    })

with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["source_dataset", "total", "high_risk_count", "error_type_counts"])
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"Written: {OUT_SUMMARY} ({len(summary_rows)} rows)")
