# Evaluation Results — V0 (no-RAG) vs V1 (full system)

Generated: 2026-04-26 20:12:53

## Retrieval metrics

| Variant | Category | n | Paper R@5 | Paper P@5 | Paper F1@5 | Paper MRR | DS R@5 | DS P@5 | DS F1@5 | DS MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | recommendation_easy | 8 | 0.708 | 0.507 | 0.547 | 0.905 | 0.210 | 0.175 | 0.190 | 0.317 |
| V1 | _all | 20 | 0.688 | 0.456 | 0.506 | 0.784 | 0.213 | 0.169 | 0.187 | 0.326 |
| V1 | recommendation_hard | 2 | 0.667 | 0.500 | 0.571 | 1.000 | 0.167 | 0.100 | 0.125 | 0.100 |
| V1 | direct_answer | 5 | 0.577 | 0.370 | 0.406 | 0.692 | — | — | — | — |
| V1 | hybrid | 3 | 0.833 | 0.467 | 0.556 | 0.583 | 0.250 | 0.200 | 0.222 | 0.500 |
| V1 | oos | 2 | — | — | — | — | — | — | — | — |

## Grounding & citation metrics

| Variant | Category | n | Grounding Rate | Cited Anything | Unique Sources | Meth Chunks | ROUGE-L |
|---|---|---|---|---|---|---|---|
| V1 | recommendation_easy | 8 | 1.000 | 0.750 | 2.500 | 1.000 | — |
| V1 | _all | 20 | 1.000 | 0.850 | 2.450 | 1.000 | 0.211 |
| V1 | recommendation_hard | 2 | 1.000 | 1.000 | 3.000 | 1.000 | — |
| V1 | direct_answer | 5 | 1.000 | 1.000 | 2.200 | — | 0.211 |
| V1 | hybrid | 3 | 1.000 | 1.000 | 3.333 | 1.000 | — |
| V1 | oos | 2 | 1.000 | 0.500 | 1.000 | — | — |

## Abstention & latency

| Variant | Category | n | Abstention Rate | Latency mean (s) | Latency P95 (s) |
|---|---|---|---|---|---|
| V1 | recommendation_easy | 8 | — | 16.545 | 20.030 |
| V1 | _all | 20 | 0.000 | 13.687 | 20.030 |
| V1 | recommendation_hard | 2 | — | 14.473 | 13.966 |
| V1 | direct_answer | 5 | — | 10.506 | 11.460 |
| V1 | hybrid | 3 | — | 12.844 | 14.005 |
| V1 | oos | 2 | 0.000 | 10.681 | 9.889 |