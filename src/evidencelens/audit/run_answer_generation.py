from pathlib import Path
import json, jsonlines, re
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

import anthropic

PROMPT_FILE = ROOT / "src/evidencelens/prompts/answer_generation_prompt.txt"
INPUT_FILE  = ROOT / "data/processed/samples/diagnostic_combined_sample.jsonl"
OUTPUT_FILE = ROOT / "outputs/predictions/answer_generation_outputs.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

prompt_template = PROMPT_FILE.read_text()
records = list(jsonlines.open(INPUT_FILE))
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model  = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

def build_message(record):
    doc = record["document_a"]
    title    = doc.get("title") or ""
    abstract = doc.get("abstract") or ""
    sentences = doc.get("sentences") or []
    sentences_text = " ".join(str(s) for s in sentences if s)
    # Use abstract if sentences is empty, sentences if abstract is empty
    text = sentences_text if sentences_text else abstract

    return (prompt_template
        .replace("{title}", title)
        .replace("{abstract}", abstract)
        .replace("{sentences}", text)
        .replace("{input_claim_or_question}",
                 str(record.get("input_claim_or_question") or "")))

def strip_fences(text):
    return re.sub(r"```(?:json)?|```", "", text).strip()

results = []
for i, record in enumerate(records):
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": build_message(record)}]
        )
        raw = msg.content[0].text
        try:
            parsed = json.loads(strip_fences(raw))
            row = {
                "id": record["id"],
                "source_dataset": record["source_dataset"],
                "input_claim_or_question": record["input_claim_or_question"],
                "gold_label": record["gold_label"],
                "llm_answer": parsed.get("answer", ""),
                "llm_main_claim": parsed.get("main_claim", ""),
                "llm_evidence_sentences": parsed.get("evidence_sentences", []),
                "llm_limitations": parsed.get("limitations", []),
                "llm_confidence": parsed.get("confidence", ""),
                "llm_raw_response": raw,
            }
        except json.JSONDecodeError:
            row = {
                "id": record["id"],
                "source_dataset": record["source_dataset"],
                "input_claim_or_question": record["input_claim_or_question"],
                "gold_label": record["gold_label"],
                "llm_answer": "PARSE_ERROR",
                "llm_main_claim": "", "llm_evidence_sentences": [],
                "llm_limitations": [], "llm_confidence": "",
                "llm_raw_response": raw,
            }
    except Exception as e:
        print(f"  API_ERROR: {e}")
        row = {
            "id": record["id"],
            "source_dataset": record["source_dataset"],
            "input_claim_or_question": record["input_claim_or_question"],
            "gold_label": record["gold_label"],
            "llm_answer": "API_ERROR",
            "llm_main_claim": "", "llm_evidence_sentences": [],
            "llm_limitations": [], "llm_confidence": "",
            "llm_raw_response": str(e),
        }
    results.append(row)
    print(f"[{i+1}/{len(records)}] {record['id']} — confidence: {row.get('llm_confidence','')}")

with jsonlines.open(OUTPUT_FILE, "w") as writer:
    writer.write_all(results)

from collections import Counter
conf = Counter(r["llm_confidence"] for r in results)
print(f"\nTotal: {len(results)}")
print(f"PARSE_ERROR: {sum(1 for r in results if r['llm_answer']=='PARSE_ERROR')}")
print(f"API_ERROR:   {sum(1 for r in results if r['llm_answer']=='API_ERROR')}")
print(f"Confidence:  {dict(conf)}")
