# EvidenceLens

> Prototype evaluation framework for LLM scientific reasoning and evidence conflict detection.

## Goal

LLMs can produce fluent but unsupported scientific explanations. EvidenceLens tests whether
LLM-generated answers are grounded in source evidence, preserve uncertainty and limitations,
and detect conflicts across documents — rather than producing plausible but unsupported reasoning.

## Datasets

| Dataset               | Role                               | Records | Gold Labels                    |
|-----------------------|------------------------------------|---------|--------------------------------|
| SciFact               | Scientific claim verification      | 5       | support / contradict           |
| PubMedQA              | Biomedical yes/no/maybe QA         | 5       | yes / no / maybe               |
| QASPER                | Research paper QA                  | 5       | see_answer (extractive/free)   |
| Manual conflict pairs | Multi-document conflict detection  | 5       | conflict_or_conditionally_supported |

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
                               support_label_distribution.png
                               confidence_vs_risk_heatmap.png
```

## Error Taxonomy

| Error Type                | Definition                                                         |
|---------------------------|--------------------------------------------------------------------|
| unsupported_claim         | LLM makes a claim not supported by the source text                 |
| wrong_evidence            | LLM cites irrelevant or incorrect evidence                         |
| missing_evidence          | LLM gives answer but no source-grounded support                    |
| overgeneralization        | LLM expands a narrow result into a broad claim                     |
| condition_dropping        | LLM removes study conditions, dataset limits, or population limits |
| false_certainty           | LLM says yes/no when evidence is uncertain or maybe                |
| missing_limitation        | LLM ignores limitations stated or implied in source                |
| contradiction_with_source | LLM answer conflicts with source text                              |
| conflict_ignored          | In multi-document setting, LLM hides disagreement                  |
| paper_section_misread     | LLM pulls evidence from wrong section or wrong context             |

## How to Run

### Setup

```bash
bash setup_env.sh      # or setup_env.bat on Windows
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run in order

```bash
python scripts/download_qasper.py

python src/evidencelens/data/normalize_scifact.py
python src/evidencelens/data/normalize_pubmedqa.py
python src/evidencelens/data/normalize_qasper.py
python src/evidencelens/data/build_diagnostic_dataset.py

python src/evidencelens/audit/run_answer_generation.py
python src/evidencelens/audit/run_evidence_audit.py
python src/evidencelens/audit/run_conflict_audit.py

python src/evidencelens/eval/export_error_table.py
python src/evidencelens/viz/plot_error_summary.py
```

### Verify combined sample

```python
python -c "
import jsonlines
from collections import Counter
rows = list(jsonlines.open('data/processed/samples/diagnostic_combined_sample.jsonl'))
print(Counter(r['source_dataset'] for r in rows))
assert len(rows) == 15
print('PASS')
"
```

---

## Results (Prototype Run — n=15 + 5 conflict pairs)

### Key metrics

| Metric                              | Value              |
|-------------------------------------|--------------------|
| Records audited                     | 15                 |
| Records with detected errors        | 6 / 15 (40%)       |
| Support label: supported            | 12 / 15 (80%)      |
| Support label: partially_supported  | 3 / 15 (20%)       |
| Support label: contradicted         | 0 / 15 (0%)        |
| Hallucination risk: low             | 13 / 15 (87%)      |
| Hallucination risk: medium          | 2 / 15 (13%)       |
| Hallucination risk: high            | 0 / 15 (0%)        |
| Conflicts correctly detected        | 5 / 5 (100%)       |
| Forced consensus (conflict hidden)  | 0 / 5 (0%)         |

### Error type distribution

![Error Types](outputs/figures/error_type_counts.png)

Top error types: `condition_dropping` = 2 (33%), `overgeneralization` = 2 (33%),
`missing_limitation` = 1 (17%), `unsupported_claim` = 1 (17%).

### Hallucination risk by dataset

![Hallucination Risk](outputs/figures/hallucination_risk_by_dataset.png)

### Support label distribution

![Support Labels](outputs/figures/support_label_distribution.png)

### Confidence vs hallucination risk

![Confidence vs Risk](outputs/figures/confidence_vs_risk_heatmap.png)

### Key finding

The heatmap reveals a miscalibration case: 1 record where the LLM reported high confidence
but the audit assigned medium hallucination risk. This is the core finding EvidenceLens is
designed to surface — LLM confidence does not equal reasoning faithfulness. Errors concentrated
in scope failures: `overgeneralization` and `condition_dropping` together account for 67% of
detected errors.

### Interpretation

QASPER records produced zero detected errors, likely because the `see_answer` gold label does
not define a strict pass/fail boundary — the audit compares the LLM answer to paper content
rather than a binary correct/incorrect label. The conflict detection result (5/5 detected,
0 forced consensus) indicates the model acknowledged disagreement in every manually constructed
conflict pair, though this is measured by the model auditing its own answer in the same generation
pass and cannot be taken as fully independent verification.

### Limitation

All audit metrics — `support_label`, `hallucination_risk`, `overgeneralization`,
`false_certainty`, `limitation_preserved`, `error_types`, `conflict_detected`, and
`forced_consensus` — are produced by an LLM judging either its own output or another LLM's
output. No rule-based or deterministic verification layer exists. A model biased toward
self-approval will systematically underreport errors. Additionally, the evidence audit prompt
exposes the `gold_label` to the auditing model before it renders judgment, which may inflate
`support_label` accuracy. These limitations are inherent to the LLM-as-judge evaluation
paradigm and must be accounted for when interpreting any result from this pipeline.

---

## Outputs

| File                                                       | Description                      |
|------------------------------------------------------------|----------------------------------|
| data/processed/samples/diagnostic_combined_sample.jsonl    | 15 normalized records            |
| outputs/predictions/answer_generation_outputs.jsonl        | LLM answers per record           |
| outputs/predictions/evidence_audit_outputs.jsonl           | Audit results per record         |
| outputs/predictions/conflict_audit_outputs.jsonl           | Conflict handling results        |
| outputs/tables/error_analysis_table.csv                    | Per-record error analysis        |
| outputs/tables/error_summary_by_dataset.csv                | Dataset-level summary            |
| outputs/figures/error_type_counts.png                      | Error type distribution          |
| outputs/figures/hallucination_risk_by_dataset.png          | Risk by dataset                  |
| outputs/figures/support_label_distribution.png             | Support label distribution       |
| outputs/figures/confidence_vs_risk_heatmap.png             | Confidence vs risk heatmap       |

## Status

Prototype — 15 diagnostic records across SciFact, PubMedQA, and QASPER.
5 manually constructed conflict pairs for multi-document conflict detection.
Full pipeline run complete.
