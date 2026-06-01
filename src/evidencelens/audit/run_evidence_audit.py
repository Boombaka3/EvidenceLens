from pathlib import Path
import json, jsonlines, re
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

import anthropic

PROMPT_FILE  = ROOT / "src/evidencelens/prompts/evidence_audit_prompt.txt"
ANSWERS_FILE = ROOT / "outputs/predictions/answer_generation_outputs.jsonl"
SAMPLES_FILE = ROOT / "data/processed/samples/diagnostic_combined_sample.jsonl"
OUTPUT_FILE  = ROOT / "outputs/predictions/evidence_audit_outputs.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

prompt_template = PROMPT_FILE.read_text()
answers  = {r["id"]: r for r in jsonlines.open(ANSWERS_FILE)}
samples  = {r["id"]: r for r in jsonlines.open(SAMPLES_FILE)}
client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model    = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

def strip_fences(text):
    return re.sub(r"```(?:json)?|```", "", text).strip()

results = []
for i, (rid, ans) in enumerate(answers.items()):
    rec = samples.get(rid, {})
    doc = rec.get("document_a", {})
    sentences = " ".join(doc.get("sentences", []))
    evidence  = " | ".join(ans.get("llm_evidence_sentences", []))

    msg_text = (prompt_template
        .replace("{title}", doc.get("title", ""))
        .replace("{abstract}", doc.get("abstract", ""))
        .replace("{sentences}", sentences)
        .replace("{input_claim_or_question}", rec.get("input_claim_or_question", ""))
        .replace("{gold_label}", rec.get("gold_label", ""))
        .replace("{llm_answer}", ans.get("llm_answer", ""))
        .replace("{llm_evidence_sentences}", evidence))

    try:
        msg = client.messages.create(
            model=model, max_tokens=1024, temperature=0,
            messages=[{"role": "user", "content": msg_text}]
        )
        raw = msg.content[0].text
        try:
            parsed = json.loads(strip_fences(raw))
            row = {
                "id": rid,
                "source_dataset": rec.get("source_dataset", ""),
                "task_type": rec.get("task_type", ""),
                "input_claim_or_question": rec.get("input_claim_or_question", ""),
                "gold_label": rec.get("gold_label", ""),
                "llm_answer": ans.get("llm_answer", ""),
                "llm_confidence": ans.get("llm_confidence", ""),
                "extracted_claim": parsed.get("extracted_claim", ""),
                "support_label": parsed.get("support_label", ""),
                "limitation_preserved": parsed.get("limitation_preserved", None),
                "overgeneralization": parsed.get("overgeneralization", None),
                "false_certainty": parsed.get("false_certainty", None),
                "hallucination_risk": parsed.get("hallucination_risk", ""),
                "error_types": parsed.get("error_types", []),
                "explanation": parsed.get("explanation", ""),
                "audit_raw_response": raw,
            }
        except json.JSONDecodeError:
            row = {"id": rid, "support_label": "PARSE_ERROR",
                   "audit_raw_response": raw,
                   "source_dataset": rec.get("source_dataset",""),
                   "task_type": rec.get("task_type",""),
                   "input_claim_or_question": rec.get("input_claim_or_question",""),
                   "gold_label": rec.get("gold_label",""),
                   "llm_answer": ans.get("llm_answer",""),
                   "llm_confidence": ans.get("llm_confidence",""),
                   "extracted_claim":"","limitation_preserved":None,
                   "overgeneralization":None,"false_certainty":None,
                   "hallucination_risk":"","error_types":[],"explanation":""}
    except Exception as e:
        print(f"  API_ERROR: {e}")
        row = {"id": rid, "support_label": "API_ERROR",
               "audit_raw_response": str(e),
               "source_dataset": rec.get("source_dataset",""),
               "task_type": rec.get("task_type",""),
               "input_claim_or_question": rec.get("input_claim_or_question",""),
               "gold_label": rec.get("gold_label",""),
               "llm_answer": ans.get("llm_answer",""),
               "llm_confidence": ans.get("llm_confidence",""),
               "extracted_claim":"","limitation_preserved":None,
               "overgeneralization":None,"false_certainty":None,
               "hallucination_risk":"","error_types":[],"explanation":""}
    results.append(row)
    print(f"[{i+1}/{len(answers)}] {rid} — risk: {row.get('hallucination_risk','')}")

with jsonlines.open(OUTPUT_FILE, "w") as writer:
    writer.write_all(results)

from collections import Counter
print(f"\nTotal audited: {len(results)}")
print(f"support_label:      {dict(Counter(r['support_label'] for r in results))}")
print(f"hallucination_risk: {dict(Counter(r['hallucination_risk'] for r in results))}")
print(f"Non-empty error_types: {sum(1 for r in results if r.get('error_types'))}")
print(f"PARSE_ERROR: {sum(1 for r in results if r['support_label']=='PARSE_ERROR')}")
