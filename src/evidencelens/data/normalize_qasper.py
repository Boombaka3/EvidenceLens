import json
import jsonlines
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/external/repos/qasper/extracted/qasper-train-v0.3.json"
OUT    = ROOT / "data/processed/samples/qasper_normalized_sample5.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

with SOURCE.open("r", encoding="utf-8") as f:
    data = json.load(f)

records = []

for paper_id, paper in data.items():
    if len(records) >= 5:
        break

    title    = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    qas      = paper.get("qas") or []
    sentences = [s.strip() for s in abstract.split(".") if s.strip()]

    for i, qa in enumerate(qas):
        if len(records) >= 5:
            break

        question = qa.get("question", "")
        if not question or not question.strip():
            continue

        answers = qa.get("answers", [])

        if all(a.get("answer", {}).get("unanswerable", True) for a in answers):
            continue

        first_valid = None
        for a in answers:
            ans = a.get("answer", {})
            if not ans.get("unanswerable", True):
                first_valid = ans
                break

        if first_valid is None:
            continue

        records.append({
            "id": f"qasper_{paper_id}_q{i}",
            "source_dataset": "QASPER",
            "task_type": "paper_qa",
            "input_claim_or_question": question.strip(),
            "document_a": {
                "doc_id": paper_id,
                "title": title,
                "abstract": abstract,
                "sentences": sentences,
                "metadata": {},
            },
            "document_b": None,
            "gold_label": "see_answer",
            "gold_evidence": json.dumps({
                "answer":   first_valid.get("free_form_answer", ""),
                "evidence": first_valid.get("extractive_spans", []),
            }),
            "target_error_types": [
                "missing_evidence",
                "unsupported_claim",
                "paper_section_misread",
            ],
        })

if len(records) < 5:
    print(f"WARNING: only {len(records)} valid QA pairs found (expected 5)")

with jsonlines.open(OUT, "w") as writer:
    writer.write_all(records)

print(f"Records written: {len(records)}")
print(f"Output: {OUT}")
print(f"Sample ids: {[r['id'] for r in records]}")


if __name__ == "__main__":
    pass
