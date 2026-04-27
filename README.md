# Earth Science Research Assistant

A hybrid RAG system that helps researchers find datasets and papers for Earth science questions, with strict LLM grounding and multi-turn conversation support. Built for Harvard EPS 210.

---

## Capabilities

- **Hybrid retrieval** — local-first (82 full-text PDFs + 2,500+ dataset records from NASA CMR / STAC / Copernicus CDS / CDSE) supplemented by runtime OpenAlex (papers) and Zenodo (datasets) APIs.
- **Structured spatial / temporal matching** — real bbox IoU + date-range overlap, with hard filtering when the user marks region or timescale as must-have.
- **Provenance-aware reranking** — local papers and external OpenAlex results scored with separate feature sets (local ceiling 0.85 vs external 0.60), so local full-text evidence naturally dominates. Datasets scored on 5 weighted features.
- **Grounded answer generation** — every `[DS-N] / [P-N] / [C-N]` citation is validated by `_verify_grounding()` against the evidence block before the answer is returned; violations are surfaced in a `GroundingReport`.
- **Per-query evidence cache** — each query snapshots `parsed_query`, `openalex`, `zenodo`, `local_datasets`, `chunks`, `evidence_block.txt`, `final_answer`, and `grounding_report` to disk for full reproducibility.
- **Multi-turn conversation** — pre-RAG intent classifier routes every turn into 5 branches (chitchat / new_question / re_recommend / detail_followup / out_of_scope). Session state round-trips between client and backend so follow-ups ("tell me about paper 1", "more datasets", "第二篇") resolve against prior results.
- **Gradio UI + FastAPI** — Gradio chat for end users at port 7860, JSON/pretty endpoints at port 8000.

---

## Architecture

```
           User Query + SessionState (from client)
                         │
                         ▼
             ┌────────────────────────┐
             │   Intent Classifier     │  → chitchat / new_question /
             └───────────┬─────────────┘    re_recommend / detail_followup /
                         │                  out_of_scope
                         ▼
             ┌────────────────────────┐
             │   Query Parser          │  history-aware, inherits topic
             │   (+ region_bbox,       │  from prior turns for short
             │    parsed_timescale)    │  follow-ups
             └───────────┬─────────────┘
                         │
     ┌───────────────────┼────────────────────────────┐
     │                   │                            │
     ▼                   ▼                            ▼
 PAPERS             DATASETS                       CHUNKS
 local chunks       local 2,500+ + Zenodo          82 PDFs → 3,700+
 + OpenAlex         + structured spatial/          heading-aware
 quota 7 + 3        temporal overlap + hard        semantic search
                    must-have filter
     │                   │                            │
     └──────────┬────────┴───────┬────────────────────┘
                │                │
                ▼                ▼
        Paper–Dataset Linker  (chunk mention > abstract mention > has_doi > semantic)
                │
                ▼
        Evidence Block  (DATASETS / PAPERS / CHUNKS with stable [DS-N] / [P-N] / [C-N] IDs)
                │
                ▼
        Answer Generator  →  JSON FinalAnswer  →  _verify_grounding()
                │
                ▼
        Evidence Cache Writer  →  generated/evidence_cache/<timestamp>_<qhash>/
```

---

## Project Structure

```
earth-science-research-assistant-rag/
├── app/
│   ├── main.py                 FastAPI app + /query, /query/pretty, /health endpoints (thin)
│   ├── router.py               Intent routing, session state updates, handler dispatch
│   ├── core/
│   │   ├── config.py           Settings loader + OpenAI client singleton
│   │   ├── schemas.py          All Pydantic models (SessionState, ParsedQuery, FinalAnswer,
│   │   │                       GroundingReport, DatasetCandidate, PaperCandidate, …)
│   │   └── spatial_temporal_match.py   Bbox parsing, IoU+containment overlap, date-range overlap
│   ├── clients/
│   │   ├── openalex_client.py  OpenAlex search (recent + impactful buckets) + single-work lookup
│   │   └── zenodo_client.py    Zenodo dataset API client
│   ├── ingestion/              One-time preprocessing — run once before serving
│   │   ├── preprocess.py       Orchestrates steps 1–4 below
│   │   ├── paper_registry.py   id_track.xlsx → paper_registry.jsonl + OpenAlex enrichment
│   │   ├── dataset_normalizer.py   Unifies 4 metadata source schemas → normalized_datasets.jsonl
│   │   ├── pdf_extractor.py    PyMuPDF primary, pdfplumber fallback
│   │   ├── chunker.py          Heading-aware + fixed-size fallback chunking
│   │   └── embedder.py         sentence-transformer (BAAI/bge-small-en-v1.5) + ChromaDB
│   └── pipeline/               Per-query RAG stages
│       ├── intent_classifier.py    Pre-RAG routing (5 intents) + Chinese/English ordinal parsing
│       ├── query_parser.py         OpenAI → ParsedQuery (history-aware)
│       ├── paper_matcher.py        OpenAlex ↔ local PDF fuzzy title match
│       ├── chunk_retriever.py      Chunk vector search — global and per-paper (for detail follow-ups)
│       ├── dataset_retriever.py    Local + Zenodo retrieval + must-have hard filter
│       ├── linker.py               Paper-dataset linker (alias-aware, fuzzy)
│       ├── reranker.py             Provenance-aware paper reranker + dataset reranker
│       ├── answer_generator.py     Grounded prompt + JSON output + _verify_grounding
│       └── evidence_cache_writer.py    Per-query snapshot to disk
├── config/
│   └── settings.yaml           All weights, model names, paths, thresholds
├── prompts/                    Modular prompt fragments, composed by answer_generator
│   ├── base_rules.md               HARD / SOFT / abstention grounding rules
│   ├── intent_definition.md        Per-intent instruction fragments:
│   ├── intent_paper_specific.md      definition_or_explanation
│   ├── intent_paper_primary.md       paper_specific_question
│   ├── intent_dataset_primary.md     paper / dataset recommendation
│   ├── intent_methodology.md         methodology_support
│   ├── intent_research_starter.md    research_starter
│   └── intent_fallback.md            other
├── local_database/
│   ├── dataset_metadata/       4 raw metadata JSONs + dataset_aliases.json (123 aliases)
│   ├── fulltext_paper/         82 PDFs + id_track.xlsx (local_id ↔ title ↔ filename)
│   ├── fetch_metadata.py       Re-fetch dataset metadata
│   ├── fetch_papers.py         Re-fetch / download papers
│   └── paper_manifest.json     Paper index (status, DOI, filename)
├── evaluation/
│   ├── eval_set_v2.json        20-query eval set with gold IDs
│   ├── metrics.py              Recall@k, MRR, grounding rate, abstention rate, latency
│   ├── run_eval.py             V0 (no-RAG baseline) vs V1 (full system) runner
│   └── results/                per_sample_v0/v1.jsonl, summary.csv, comparison_table.md
├── generated/                  All derived artifacts (git-ignored)
│   ├── chroma/                     ChromaDB — datasets + chunks collections
│   ├── evidence_cache/             Per-query snapshots
│   ├── parsed_papers/              <local_id>.json per paper (cleaned text)
│   ├── debug/                      Last-parsed-query, last-chunk-candidates, last-reranked-* dumps
│   ├── chunks.jsonl
│   ├── paper_registry.jsonl
│   └── normalized_datasets.jsonl
├── ui.py                       Gradio UI — holds session state, renders datasets/papers side panel
├── start.sh                    Launches FastAPI (8000) + Gradio UI (7860)
├── requirements.txt
└── .env                        OPENAI_API_KEY
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) — `$0.10` is enough for one full eval run with `gpt-4o-mini`

### 2. Install

```bash
cd earth-science-research-assistant-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment

Create `.env`:

```
OPENAI_API_KEY=sk-proj-...
```

### 4. Preprocessing (one-time)

Builds all indices: paper registry, dataset normalization, PDF extraction, chunk embedding.

```bash
python -m app.ingestion.preprocess
```

Produces:
- `generated/paper_registry.jsonl` — local_id ↔ openalex_id ↔ title
- `generated/normalized_datasets.jsonl` — unified 2,500+ dataset records
- `generated/chunks.jsonl` — 3,700+ section-aware chunks
- `generated/parsed_papers/<local_id>.json` — cleaned PDF text
- `generated/chroma/` — two ChromaDB collections (datasets, chunks)

### 5. Start the server

```bash
./start.sh
```

- API: `http://localhost:8000` (`/query`, `/query/pretty`, `/health`)
- UI: `http://localhost:7860`

---

## Usage

**Gradio UI** (recommended for humans):

```bash
python ui.py   # or ./start.sh, which launches both
```

**JSON API**:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What datasets can I use to study drought in Central Asia?"}' \
  | python -m json.tool
```

**Pretty text** (for debugging):

```bash
curl -s -X POST http://localhost:8000/query/pretty \
  -H "Content-Type: application/json" \
  -d '{"query":"..."}'
```

**Multi-turn**: include `history` (last few `{role, content}` messages) and `session_state` (the object returned from the previous `/query` response) in the request body.

---

## Evaluation

```bash
  python -m evaluation.run_eval --skip-v0   # run without v0 comparison              
  python -m evaluation.run_eval             # run with v0 comparison
```

Runs V0 (no-RAG baseline) and V1 (full system) on 20 queries across 5 categories (recommendation_easy / recommendation_hard / direct_answer / hybrid / oos). Writes per-sample JSONL, summary CSV, and a markdown comparison table to `evaluation/results/`.

Metrics: Paper Recall@5/@10, Dataset Recall@5/@10, MRR, grounding rate, methodology-cites-chunks rate, abstention rate (OOS), latency.
