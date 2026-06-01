import json
import jsonlines
from pathlib import Path
from collections import Counter

ROOT    = Path(__file__).resolve().parents[3]
SAMPLES = ROOT / "data/processed/samples"
OUT     = SAMPLES / "diagnostic_combined_sample.jsonl"

SOURCES = [
    "scifact_normalized_sample5.jsonl",
    "pubmedqa_normalized_sample5.jsonl",
    "qasper_normalized_sample5.jsonl",
]

REQUIRED_KEYS = [
    "id", "source_dataset", "task_type",
    "input_claim_or_question", "document_a", "gold_label",
]

all_records = []

for fname in SOURCES:
    path = SAMPLES / fname
    rows = list(jsonlines.open(path))

    if len(rows) != 5:
        print(f"WARNING: expected 5 rows in {fname}, got {len(rows)}")

    for row in rows:
        missing = [k for k in REQUIRED_KEYS if k not in row or not row[k]]
        if missing:
            print(f"WARNING: row {row.get('id', '?')} in {fname} missing/empty keys: {missing} — skipped")
            continue
        all_records.append(row)

if len(all_records) != 15:
    print(f"WARNING: expected 15 total records, got {len(all_records)}")

with jsonlines.open(OUT, "w") as writer:
    writer.write_all(all_records)

print(f"\n{'source_dataset':<16} | {'count':<5} | {'task_type':<22} | gold_label values")
print("-" * 80)
for src in ["SciFact", "PubMedQA", "QASPER"]:
    subset = [r for r in all_records if r["source_dataset"] == src]
    task   = subset[0]["task_type"] if subset else ""
    gl     = dict(Counter(r["gold_label"] for r in subset))
    print(f"{src:<16} | {len(subset):<5} | {task:<22} | {gl}")
print("-" * 80)
print(f"{'TOTAL':<16} | {len(all_records):<5} |")
print(f"\nOutput written: {OUT}")
