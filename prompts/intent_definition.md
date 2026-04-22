Intent: definition_or_explanation
Mode: direct_answer

Output shape:
- direct_answer: 2–4 sentence factual answer using [C-N] chunk citations (preferred) or [P-N]. For purely conceptual terms with no chunk support, follow S1 fallback and add the required uncertainty_note.
- recommended_datasets: []
- recommended_papers: []
- methodology_hints: [] (unless the definition itself IS a method and is supported by a chunk).
