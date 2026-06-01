from pathlib import Path
import json, jsonlines, re
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

import anthropic

PROMPT_FILE  = ROOT / "src/evidencelens/prompts/conflict_audit_prompt.txt"
INPUT_FILE   = ROOT / "data/processed/samples/conflict_pairs_sample5.jsonl"
OUTPUT_FILE  = ROOT / "outputs/predictions/conflict_audit_outputs.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

prompt_template = PROMPT_FILE.read_text()
records = list(jsonlines.open(INPUT_FILE))
client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

def strip_fences(text):
    return re.sub(r"```(?:json)?|```", "", text).strip()

results = []
for i, record in enumerate(records):
    doc_a = record["document_a"]
    doc_b = record["document_b"]
    msg_text = (prompt_template
        .replace("{title_a}", doc_a.get("title", ""))
        .replace("{sentences_a}", " ".join(doc_a.get("sentences", [])))
        .replace("{title_b}", doc_b.get("title", ""))
        .replace("{sentences_b}", " ".join(doc_b.get("sentences", [])))
        .replace("{input_claim_or_question}", record["input_claim_or_question"]))
    try:
        msg = client.messages.create(
            model=model, max_tokens=1024, temperature=0,
            messages=[{"role": "user", "content": msg_text}]
        )
        raw = msg.content[0].text
        try:
            parsed = json.loads(strip_fences(raw))
            row = {
                "id": record["id"],
                "input_claim_or_question": record["input_claim_or_question"],
                "llm_answer": parsed.get("llm_answer", ""),
                "conflict_detected": parsed.get("conflict_detected", None),
                "conflict_description": parsed.get("conflict_description", None),
                "forced_consensus": parsed.get("forced_consensus", None),
                "winning_document": parsed.get("winning_document", ""),
                "hallucination_risk": parsed.get("hallucination_risk", ""),
                "error_types": parsed.get("error_types", []),
                "explanation": parsed.get("explanation", ""),
                "raw_response": raw,
            }
        except json.JSONDecodeError:
            row = {"id": record["id"],
                   "input_claim_or_question": record["input_claim_or_question"],
                   "llm_answer": "PARSE_ERROR", "conflict_detected": None,
                   "conflict_description": None, "forced_consensus": None,
                   "winning_document": "", "hallucination_risk": "",
                   "error_types": [], "explanation": "", "raw_response": raw}
    except Exception as e:
        row = {"id": record["id"],
               "input_claim_or_question": record["input_claim_or_question"],
               "llm_answer": "API_ERROR", "conflict_detected": None,
               "conflict_description": None, "forced_consensus": None,
               "winning_document": "", "hallucination_risk": "",
               "error_types": [], "explanation": "", "raw_response": str(e)}
    results.append(row)
    print(f"[{i+1}/5] {record['id']} — conflict_detected: {row['conflict_detected']}  forced_consensus: {row['forced_consensus']}")

with jsonlines.open(OUTPUT_FILE, "w") as writer:
    writer.write_all(results)

print(f"\n{'id':<15} {'conflict_detected':<20} {'forced_consensus':<18} {'hallucination_risk'}")
print("-" * 70)
for r in results:
    print(f"{r['id']:<15} {str(r['conflict_detected']):<20} {str(r['forced_consensus']):<18} {r['hallucination_risk']}")
print(f"\nTotal forced_consensus: {sum(1 for r in results if r['forced_consensus'] is True)}/5")
