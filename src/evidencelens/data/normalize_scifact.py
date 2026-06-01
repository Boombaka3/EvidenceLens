import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "scifact"
OUT = ROOT / "data" / "processed" / "samples" / "scifact_normalized_sample5.jsonl"

_ERROR_MAP = {
    "CONTRADICT": ["unsupported_claim", "contradiction_with_source"],
    "SUPPORT":    ["wrong_evidence", "missing_evidence"],
    None:         ["missing_evidence", "false_certainty"],
}


def _label_from_evidence(evidence: dict) -> str:
    for doc_evidence in evidence.values():
        for entry in doc_evidence:
            label = entry.get("label", "").upper()
            if label == "CONTRADICT":
                return "contradict"
            if label == "SUPPORT":
                return "support"
    return "unknown"


def _error_types(label: str) -> list:
    key = label.upper() if label in ("contradict", "support") else None
    if label == "contradict":
        key = "CONTRADICT"
    elif label == "support":
        key = "SUPPORT"
    return _ERROR_MAP.get(key, _ERROR_MAP[None])


def _load_corpus() -> dict:
    corpus = {}
    with (RAW / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus[str(doc["doc_id"])] = doc
    return corpus


def _load_claims():
    claims = []
    with (RAW / "claims_train.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))
    return claims


def main():
    corpus = _load_corpus()
    claims = _load_claims()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for claim in claims:
        if len(records) >= 5:
            break

        evidence = claim.get("evidence", {})
        cited_ids = [str(x) for x in claim.get("cited_doc_ids", [])]

        first_doc_id = None
        for cid in cited_ids:
            if cid in corpus:
                first_doc_id = cid
                break

        if first_doc_id is None:
            continue

        doc = corpus[first_doc_id]
        label = _label_from_evidence(evidence)

        evidence_sentences = []
        for doc_evidence in evidence.values():
            for entry in doc_evidence:
                for sent in entry.get("sentences", []):
                    abstract = doc.get("abstract", [])
                    if isinstance(abstract, list) and sent < len(abstract):
                        evidence_sentences.append(abstract[sent])

        records.append({
            "id": f"scifact_{claim.get('id')}",
            "source_dataset": "SciFact",
            "task_type": "claim_verification",
            "input_claim_or_question": claim.get("claim", ""),
            "document_a": {
                "doc_id": first_doc_id,
                "title": doc.get("title", ""),
                "abstract": " ".join(doc.get("abstract", [])),
                "sentences": doc.get("abstract", []),
                "metadata": {},
            },
            "document_b": None,
            "gold_label": label,
            "gold_evidence": " ".join(evidence_sentences),
            "target_error_types": _error_types(label),
        })

    with OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
