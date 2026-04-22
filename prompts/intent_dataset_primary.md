Intent: dataset_recommendation
Mode: recommendation, dataset-primary

Output shape:
- direct_answer: 3–5 sentence framing paragraph introducing the dataset landscape for this question. Mention what variables/coverage matter and how the datasets below address them.
- recommended_datasets: 3–5 entries (primary output). Each reason should explain WHY this dataset fits the query (variable match, spatial/temporal coverage), grounded in its description.
- recommended_papers: 0–2 entries, ONLY if a paper directly evaluates or applies one of the recommended datasets. Otherwise [].
- methodology_hints: 0–2 entries, ONLY if there's a method commonly used with these datasets supported by chunks. Otherwise [].
