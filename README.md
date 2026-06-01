# EvidenceLens

> Prototype evaluation framework for LLM scientific reasoning and evidence conflict detection.

## Goal

LLMs can produce fluent but unsupported scientific explanations. EvidenceLens tests whether
LLM-generated answers are grounded in source evidence, preserve uncertainty and limitations,
and detect conflicts across documents — rather than producing plausible but unsupported reasoning.

## Datasets

| Dataset              | Role                              | Records |
|----------------------|-----------------------------------|---------|
| SciFact              | Scientific claim verification     | 5       |
| PubMedQA             | Biomedical yes/no/maybe QA        | 5       |
| QASPER               | Research paper QA                 | 5       |
| Manual conflict pairs| Multi-document conflict detection | 5       |

## Pipeline

```
raw data (SciFact / PubMedQA / QASPER)
        |
        v
normalize_{dataset}.py
        |
        v
diagnostic_combined_sample.jsonl  (15 records)
        |
        v
run_answer_generation.py  -->  answer_generation_outputs.jsonl
        |
        v
run_evidence_audit.py     -->  evidence_audit_outputs.jsonl
        |
        v
run_conflict_audit.py     -->  conflict_audit_outputs.jsonl  (5 conflict pairs)
        |
        v
export_error_table.py     -->  error_analysis_table.csv
plot_error_summary.py     -->  error_type_counts.png
                               hallucination_risk_by_dataset.png
```

## Error Taxonomy

| Error Type               | Definition                                                        |
|--------------------------|-------------------------------------------------------------------|
| unsupported_claim        | LLM makes a claim not supported by the source text                |
| wrong_evidence           | LLM cites irrelevant or incorrect evidence                        |
| missing_evidence         | LLM gives answer but no source-grounded support                   |
| overgeneralization       | LLM expands a narrow result into a broad claim                    |
| condition_dropping       | LLM removes study conditions, dataset limits, or population limits|
| false_certainty          | LLM says yes/no when evidence is uncertain or maybe               |
| missing_limitation       | LLM ignores limitations stated or implied in source               |
| contradiction_with_source| LLM answer conflicts with source text                             |
| conflict_ignored         | In multi-document setting, LLM hides disagreement                 |
| paper_section_misread    | LLM pulls evidence from wrong section or wrong context            |

## How to Run

```bash
# 1. Set up environment
bash setup_env.sh         # or setup_env.bat on Windows

# 2. Add API key
# Edit .env — set ANTHROPIC_API_KEY

# 3. Download QASPER
python scripts/download_qasper.py

# 4. Normalize datasets
python src/evidencelens/data/normalize_scifact.py
python src/evidencelens/data/normalize_pubmedqa.py
python src/evidencelens/data/normalize_qasper.py

# 5. Build combined sample
python src/evidencelens/data/build_diagnostic_dataset.py

# 6. Run pipeline
python src/evidencelens/audit/run_answer_generation.py
python src/evidencelens/audit/run_evidence_audit.py
python src/evidencelens/audit/run_conflict_audit.py

# 7. Export results
python src/evidencelens/eval/export_error_table.py
python src/evidencelens/viz/plot_error_summary.py
```

## Outputs

| File | Description |
|------|-------------|
| data/processed/samples/diagnostic_combined_sample.jsonl | 15 normalized records |
| outputs/predictions/answer_generation_outputs.jsonl | LLM answers per record |
| outputs/predictions/evidence_audit_outputs.jsonl | Audit results per record |
| outputs/predictions/conflict_audit_outputs.jsonl | Conflict handling results |
| outputs/tables/error_analysis_table.csv | Per-record error analysis |
| outputs/figures/error_type_counts.png | Error type distribution |
| outputs/figures/hallucination_risk_by_dataset.png | Risk by dataset |

## Status

Prototype — 15 diagnostic records across SciFact, PubMedQA, and QASPER.
5 manually constructed conflict pairs for multi-document conflict detection.
