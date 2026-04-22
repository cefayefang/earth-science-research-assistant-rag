# Evaluation Results — V0 (no-RAG) vs V1 (full system)

Generated: 2026-04-21 23:40:48

## Aggregate metrics by category

| Variant | Category | n | Paper Recall@5 | Dataset Recall@5 | Paper MRR | Dataset MRR | Grounding Rate | Methodology Cites Chunks | Abstention Rate | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | recommendation_easy | 8 | — | — | — | — | 0.000 | — | — | 4.658 |
| V0 | _all | 20 | — | — | — | — | 0.000 | — | 0.000 | 4.308 |
| V0 | recommendation_hard | 2 | — | — | — | — | 0.000 | — | — | 3.521 |
| V0 | direct_answer | 5 | — | — | — | — | 0.000 | — | — | 3.949 |
| V0 | hybrid | 3 | — | — | — | — | 0.000 | — | — | 4.376 |
| V0 | oos | 2 | — | — | — | — | 0.000 | — | 0.000 | 4.485 |
| V1 | recommendation_easy | 8 | 0.680 | 0.125 | 1.000 | 0.160 | 1.000 | 1.000 | — | 14.983 |
| V1 | _all | 20 | 0.641 | 0.179 | 0.914 | 0.216 | 1.000 | 1.000 | 0.000 | 14.700 |
| V1 | recommendation_hard | 2 | 0.667 | 0.167 | 1.000 | 0.167 | 1.000 | 1.000 | — | 17.628 |
| V1 | direct_answer | 5 | 0.509 | — | 0.840 | — | 1.000 | 1.000 | — | 13.230 |
| V1 | hybrid | 3 | 0.750 | 0.333 | 0.778 | 0.400 | 1.000 | 1.000 | — | 12.934 |
| V1 | oos | 2 | — | — | — | — | 1.000 | 1.000 | 0.000 | 16.965 |