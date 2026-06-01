#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/orange/ruogu.fang/qixuan/EvidenceLens"
CONDA_SH="/blue/ruogu.fang/qixuan.wu/conda/etc/profile.d/conda.sh"
ENV_NAME="evidencelens_l4_b200"

cd "$PROJECT_ROOT"

source "$CONDA_SH"
conda activate "$ENV_NAME"

python -m pip install -q --upgrade datasets huggingface_hub pandas jsonlines tqdm

mkdir -p \
  data/external/repos \
  data/raw/scifact \
  data/raw/pubmedqa \
  data/raw_sample/scifact \
  data/raw_sample/pubmedqa \
  data/raw_sample/qasper \
  data/processed/samples \
  outputs/debug \
  scripts

echo "============================================================"
echo "[1] Download SciFact source files"
echo "============================================================"

cd "$PROJECT_ROOT/data/external/repos"

if [ ! -d scifact ]; then
  git clone --depth 1 https://github.com/allenai/scifact.git
else
  cd scifact
  git pull
  cd ..
fi

cd "$PROJECT_ROOT/data/external/repos/scifact"
bash ./script/download-data.sh

cd "$PROJECT_ROOT"
rsync -av data/external/repos/scifact/data/ data/raw/scifact/

echo "============================================================"
echo "[2] Download PubMedQA repo"
echo "============================================================"

cd "$PROJECT_ROOT/data/external/repos"

if [ ! -d pubmedqa ]; then
  git clone --depth 1 https://github.com/pubmedqa/pubmedqa.git
else
  cd pubmedqa
  git pull
  cd ..
fi

cd "$PROJECT_ROOT"
rsync -av data/external/repos/pubmedqa/data/ data/raw/pubmedqa/

echo "============================================================"
echo "[3] Create sample-only verification script"
echo "============================================================"

cat > scripts/sample_verify_datasets.py <<'PY'
from pathlib import Path
import json
from itertools import islice

ROOT = Path("/orange/ruogu.fang/qixuan/EvidenceLens")
RAW = ROOT / "data" / "raw"
RAW_SAMPLE = ROOT / "data" / "raw_sample"
PROCESSED = ROOT / "data" / "processed" / "samples"
DEBUG = ROOT / "outputs" / "debug"

RAW_SAMPLE.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
DEBUG.mkdir(parents=True, exist_ok=True)

N_SCIFACT = 5
N_PUBMEDQA = 5
N_QASPER = 3

report = []
previews = {}

def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows)} rows -> {path}")

def read_jsonl(path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows

def count_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def schema(obj, depth=0, max_depth=3):
    if depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: schema(v, depth + 1, max_depth) for k, v in list(obj.items())[:15]}
    if isinstance(obj, list):
        if not obj:
            return []
        return [schema(obj[0], depth + 1, max_depth)]
    return type(obj).__name__

def add_report(text=""):
    report.append(text)
    print(text)

# ------------------------------------------------------------
# SciFact
# ------------------------------------------------------------
def sample_scifact():
    add_report("\n# SciFact\n")

    base = RAW / "scifact"
    claim_path = base / "claims_train.jsonl"
    corpus_path = base / "corpus.jsonl"

    required = [claim_path, corpus_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing SciFact file: {p}")

    add_report(f"- claims_train.jsonl rows: {count_jsonl(claim_path)}")
    add_report(f"- corpus.jsonl rows: {count_jsonl(corpus_path)}")

    claims = read_jsonl(claim_path, N_SCIFACT)

    corpus = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["doc_id"])] = row

    sample_raw_claims = RAW_SAMPLE / "scifact" / "claims_train_sample5.jsonl"
    sample_raw_corpus = RAW_SAMPLE / "scifact" / "corpus_for_sample5.jsonl"
    sample_norm = PROCESSED / "scifact_normalized_sample5.jsonl"

    used_docs = {}
    normalized = []

    for c in claims:
        cited_ids = [str(x) for x in c.get("cited_doc_ids", [])]
        first_doc_id = cited_ids[0] if cited_ids else None
        doc = corpus.get(first_doc_id) if first_doc_id else None

        if doc:
            used_docs[first_doc_id] = doc

        normalized.append({
            "id": f"scifact_{c.get('id')}",
            "source_dataset": "SciFact",
            "task_type": "claim_verification",
            "input_claim_or_question": c.get("claim"),
            "document_a": {
                "doc_id": first_doc_id,
                "title": doc.get("title") if doc else None,
                "sentences": doc.get("abstract") if doc else None,
            },
            "document_b": None,
            "gold_label": "see_evidence_field",
            "gold_evidence": c.get("evidence"),
            "target_error_types": [
                "unsupported_claim",
                "wrong_evidence",
                "overgeneralization"
            ]
        })

    write_jsonl(claims, sample_raw_claims)
    write_jsonl(list(used_docs.values()), sample_raw_corpus)
    write_jsonl(normalized, sample_norm)

    first_claim = claims[0]
    first_doc = next(iter(used_docs.values())) if used_docs else None

    previews["scifact_first_claim"] = first_claim
    previews["scifact_first_corpus_doc"] = first_doc
    previews["scifact_claim_schema"] = schema(first_claim)
    previews["scifact_corpus_schema"] = schema(first_doc)

    add_report("\nSciFact claim schema:")
    add_report(json.dumps(schema(first_claim), indent=2, ensure_ascii=False))

    add_report("\nSciFact corpus schema:")
    add_report(json.dumps(schema(first_doc), indent=2, ensure_ascii=False))

# ------------------------------------------------------------
# PubMedQA
# ------------------------------------------------------------
def sample_pubmedqa():
    add_report("\n# PubMedQA\n")

    base = RAW / "pubmedqa"
    json_files = sorted(base.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No PubMedQA JSON files found under {base}")

    add_report(f"- found JSON files: {len(json_files)}")

    selected_file = None
    records = []

    for p in json_files:
        try:
            obj = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue

        if isinstance(obj, dict) and len(obj) > 0:
            selected_file = p
            for k, v in list(obj.items())[:N_PUBMEDQA]:
                if isinstance(v, dict):
                    rec = dict(v)
                    rec["_pubid"] = k
                    records.append(rec)
            if records:
                break

    if not records:
        raise RuntimeError("Could not find dict-style PubMedQA records.")

    add_report(f"- selected file: {selected_file}")
    add_report(f"- sampled records: {len(records)}")

    raw_out = RAW_SAMPLE / "pubmedqa" / "pubmedqa_sample5.jsonl"
    norm_out = PROCESSED / "pubmedqa_normalized_sample5.jsonl"

    normalized = []

    for r in records:
        question = r.get("QUESTION") or r.get("question")
        contexts = r.get("CONTEXTS") or r.get("contexts") or r.get("context")
        long_answer = r.get("LONG_ANSWER") or r.get("long_answer")
        final_decision = r.get("final_decision") or r.get("FINAL_DECISION")

        normalized.append({
            "id": f"pubmedqa_{r.get('_pubid')}",
            "source_dataset": "PubMedQA",
            "task_type": "biomedical_qa",
            "input_claim_or_question": question,
            "document_a": {
                "doc_id": r.get("_pubid"),
                "title": None,
                "sentences": contexts,
            },
            "document_b": None,
            "gold_label": final_decision,
            "gold_evidence": long_answer,
            "target_error_types": [
                "false_certainty",
                "missing_limitation",
                "unsupported_clinical_claim"
            ]
        })

    write_jsonl(records, raw_out)
    write_jsonl(normalized, norm_out)

    first = records[0]
    previews["pubmedqa_selected_file"] = str(selected_file)
    previews["pubmedqa_first_record"] = first
    previews["pubmedqa_schema"] = schema(first)

    add_report("\nPubMedQA schema:")
    add_report(json.dumps(schema(first), indent=2, ensure_ascii=False))

# ------------------------------------------------------------
# QASPER sample only through Hugging Face streaming
# ------------------------------------------------------------
def sample_qasper():
    add_report("\n# QASPER\n")

    from datasets import load_dataset

    try:
        ds = load_dataset("allenai/qasper", split="train", streaming=True)
    except Exception as e:
        add_report(f"- streaming load failed: {repr(e)}")
        add_report("- trying non-streaming load for train split")
        ds = load_dataset("allenai/qasper", split="train")

    rows = list(islice(iter(ds), N_QASPER))

    if not rows:
        raise RuntimeError("QASPER returned zero rows.")

    raw_out = RAW_SAMPLE / "qasper" / "qasper_train_sample3.jsonl"
    norm_out = PROCESSED / "qasper_normalized_sample3.jsonl"

    normalized = []

    for r in rows:
        normalized.append({
            "id": f"qasper_{r.get('id')}",
            "source_dataset": "QASPER",
            "task_type": "paper_qa",
            "input_claim_or_question": None,
            "document_a": {
                "doc_id": r.get("id"),
                "title": r.get("title"),
                "abstract": r.get("abstract"),
                "full_text_available": "full_text" in r,
            },
            "document_b": None,
            "gold_label": "see_original_qas",
            "gold_evidence": r.get("qas"),
            "target_error_types": [
                "missing_evidence",
                "unsupported_answer",
                "paper_section_misread"
            ]
        })

    write_jsonl(rows, raw_out)
    write_jsonl(normalized, norm_out)

    first = rows[0]
    previews["qasper_first_record"] = first
    previews["qasper_schema"] = schema(first)

    add_report(f"- sampled records: {len(rows)}")

    add_report("\nQASPER schema:")
    add_report(json.dumps(schema(first), indent=2, ensure_ascii=False))

# ------------------------------------------------------------
# Unified diagnostic sample
# ------------------------------------------------------------
def make_combined_sample():
    add_report("\n# Combined Sample\n")

    files = [
        PROCESSED / "scifact_normalized_sample5.jsonl",
        PROCESSED / "pubmedqa_normalized_sample5.jsonl",
        PROCESSED / "qasper_normalized_sample3.jsonl",
    ]

    combined = []
    for p in files:
        if p.exists():
            combined.extend(read_jsonl(p))

    out = PROCESSED / "diagnostic_combined_sample.jsonl"
    write_jsonl(combined, out)

    add_report(f"- combined normalized rows: {len(combined)}")
    add_report(f"- combined sample path: {out}")

def main():
    sample_scifact()
    sample_pubmedqa()
    sample_qasper()
    make_combined_sample()

    report_path = DEBUG / "sample_dataset_schema_report.md"
    preview_path = DEBUG / "sample_dataset_record_previews.json"

    report_path.write_text("\n".join(report), encoding="utf-8")
    preview_path.write_text(json.dumps(previews, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n============================================================")
    print("[DONE]")
    print(f"Report: {report_path}")
    print(f"Preview JSON: {preview_path}")
    print(f"Combined normalized sample: {PROCESSED / 'diagnostic_combined_sample.jsonl'}")
    print("============================================================")

if __name__ == "__main__":
    main()
PY

echo "============================================================"
echo "[4] Run sample verification"
echo "============================================================"

python scripts/sample_verify_datasets.py | tee outputs/debug/sample_dataset_schema_stdout.txt

echo ""
echo "DONE."
