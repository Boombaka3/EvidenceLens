import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "pubmedqa" / "ori_pqal.json"
OUT = ROOT / "data" / "processed" / "samples" / "pubmedqa_normalized_sample5.jsonl"

_ERROR_MAP = {
    "maybe": ["false_certainty", "missing_limitation"],
    "yes":   ["overgeneralization", "condition_dropping"],
    "no":    ["overgeneralization", "condition_dropping"],
}


def main():
    with RAW.open("r", encoding="utf-8") as f:
        data = json.load(f)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for pubmed_id, entry in list(data.items())[:5]:
        question = entry.get("QUESTION", "")
        contexts = entry.get("CONTEXTS", [])
        long_answer = entry.get("LONG_ANSWER", "")
        final_decision = entry.get("final_decision", "")
        year = entry.get("YEAR", "")
        meshes = entry.get("MESHES", [])

        records.append({
            "id": f"pubmedqa_{pubmed_id}",
            "source_dataset": "PubMedQA",
            "task_type": "biomedical_qa",
            "input_claim_or_question": question,
            "document_a": {
                "doc_id": pubmed_id,
                "title": "",
                "abstract": long_answer,
                "sentences": contexts,
                "metadata": {"year": year, "meshes": meshes},
            },
            "document_b": None,
            "gold_label": final_decision,
            "gold_evidence": long_answer,
            "target_error_types": _ERROR_MAP.get(final_decision, ["false_certainty", "missing_limitation"]),
        })

    with OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
