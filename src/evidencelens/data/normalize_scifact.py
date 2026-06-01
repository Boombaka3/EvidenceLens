import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW  = ROOT / "data" / "raw" / "scifact"
OUT  = ROOT / "data" / "processed" / "samples" / "scifact_normalized_sample5.jsonl"

_ERROR_MAP = {
    "CONTRADICT": ["unsupported_claim", "contradiction_with_source"],
    "SUPPORT":    ["wrong_evidence", "missing_evidence"],
    "default":    ["missing_evidence", "false_certainty"],
}


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

        # Skip claims with no evidence annotation
        if not evidence:
            continue

        doc_id     = list(evidence.keys())[0]
        ev_entries = evidence[doc_id]
        label_raw  = ev_entries[0]["label"]          # "SUPPORT" or "CONTRADICT"
        sent_idxs  = ev_entries[0]["sentences"]       # list of int indices

        gold_label = "support" if label_raw == "SUPPORT" else "contradict"

        doc      = corpus.get(doc_id, {})
        abstract = doc.get("abstract", [])            # list of strings

        ev_sents      = [abstract[i] for i in sent_idxs if i < len(abstract)]
        gold_evidence = " ".join(ev_sents)

        error_types = _ERROR_MAP.get(label_raw, _ERROR_MAP["default"])

        records.append({
            "id": f"scifact_{claim.get('id')}",
            "source_dataset": "SciFact",
            "task_type": "claim_verification",
            "input_claim_or_question": claim.get("claim", ""),
            "document_a": {
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "abstract": " ".join(abstract),
                "sentences": abstract,
                "metadata": {},
            },
            "document_b": None,
            "gold_label": gold_label,
            "gold_evidence": gold_evidence,
            "target_error_types": error_types,
        })

    if len(records) < 5:
        print(f"WARNING: only {len(records)} valid records found (expected 5)")

    with OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Records written: {len(records)}")
    print(f"{'id':<20} {'gold_label':<12} claim (first 60 chars)")
    print("-" * 70)
    for r in records:
        print(f"{r['id']:<20} {r['gold_label']:<12} {r['input_claim_or_question'][:60]}")


if __name__ == "__main__":
    main()
