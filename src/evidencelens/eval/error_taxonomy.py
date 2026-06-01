ERROR_TYPES = {
    "unsupported_claim":         "LLM makes a claim not supported by the source text",
    "wrong_evidence":            "LLM cites irrelevant or incorrect evidence",
    "missing_evidence":          "LLM gives answer but no source-grounded support",
    "overgeneralization":        "LLM expands a narrow result into a broad claim",
    "condition_dropping":        "LLM removes study conditions, dataset limits, or population limits",
    "false_certainty":           "LLM says yes/no when evidence is uncertain or maybe",
    "missing_limitation":        "LLM ignores limitations stated or implied in source",
    "contradiction_with_source": "LLM answer conflicts with source text",
    "conflict_ignored":          "In multi-document setting, LLM hides disagreement",
    "paper_section_misread":     "LLM pulls evidence from wrong section or wrong context",
}

TASK_TYPE_DEFAULT_ERRORS = {
    "claim_verification":      ["unsupported_claim", "wrong_evidence", "contradiction_with_source"],
    "biomedical_qa":           ["false_certainty", "missing_limitation", "overgeneralization"],
    "paper_qa":                ["missing_evidence", "unsupported_claim", "paper_section_misread"],
    "multi_document_conflict": ["conflict_ignored", "condition_dropping", "overgeneralization"],
}
