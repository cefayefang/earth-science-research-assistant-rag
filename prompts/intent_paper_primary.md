Intent: paper_recommendation
Mode: recommendation, paper-primary

Output shape:
- direct_answer: 3–5 sentence framing of the research landscape covered by these papers — what subtopics they cover, how they differ.
- recommended_papers: 3–5 entries (primary output). Each reason should summarize the paper's focus and relevance to the query.
- recommended_datasets: 0–2 entries, ONLY if notable datasets used by the recommended papers appear in the DATASETS section.
- methodology_hints: 2–3 entries describing common methods used in the recommended papers, each citing a [C-N] chunk.
