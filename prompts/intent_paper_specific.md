Intent: paper_specific_question
Mode: direct_answer + single paper card

Output shape:
- direct_answer: 2–4 sentence answer about the specific paper, citing [P-N] (the paper) and [C-N] (relevant chunks).
- recommended_papers: exactly 1 entry — the paper the user asked about — with a one-line reason explaining why it answers the question.
- recommended_datasets: [] unless the question specifically asks about datasets used in that paper and a matching [DS-N] exists in the evidence block.
- methodology_hints: [] unless the question is about methodology.
