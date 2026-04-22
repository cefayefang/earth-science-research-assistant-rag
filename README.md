# Earth Science Research Assistant

A hybrid RAG (Retrieval-Augmented Generation) system that helps researchers find relevant datasets and papers for Earth science questions. Built for Harvard EPS 210.

The system implements a **local-first retrieval architecture** with strict LLM grounding: every recommendation, citation, and methodology claim is traceable to a record in the evidence cache, and every emitted citation tag is validated at code level before the answer is returned.

---

## What It Does

Given a natural-language research question, the system:

1. **Parses the query** into structured intent, variables, region (with bounding box), and timescale
2. **Local-first paper retrieval** via chunk-level vector search over 82 full-text PDFs
3. **Runtime API retrieval** from OpenAlex (papers) and Zenodo (datasets) as supplementary coverage
4. **Local dataset metadata** search over 2,500+ records (NASA CMR, STAC, Copernicus CDS, CDSE)
5. **Structured spatial / temporal matching** — real bbox overlap + date-range overlap, plus hard filtering when the query's region/timescale is marked as must-have
6. **Provenance-aware paper reranking** — local papers and external OpenAlex results scored with separate feature sets so local full-text evidence naturally dominates (no hand-tuned tier bonus)
7. **Grounded answer generation** — LLM constrained to the merged evidence cache; every citation is validated by `_verify_grounding()` before return
8. **Per-query evidence cache** snapshot to disk for full reproducibility

---

## Architecture

```
                       User Query
                           │
                           ▼
            ┌──────────────────────────┐
            │     Query Parser          │   OpenAI (gpt-4o-mini)
            │     (query_parser.py)     │ → ParsedQuery + region_bbox
            └────────────┬──────────────┘
                         │
       ┌─────────────────┼─────────────────────────────────────────┐
       │                 │                                          │
       │  ┌──────────────┴────────────────┐                         │
       │  │   PAPER retrieval (2-source)  │                         │
       │  │                               │                         │
       │  │  [local]    local chunks  →  source papers              │
       │  │             (surfaces a paper only if its chunks hit)   │
       │  │  [external] OpenAlex API  →  metadata-only supplement   │
       │  │                                                         │
       │  │   Slot quota: local 7, external 3  (of 10 shown to LLM) │
       │  └────────┬────────────────────────────────────────────────┘
       │           │
       │  ┌────────┴──────────┐            ┌──────────────────────┐
       │  │  DATASET retrieval │            │  CHUNK retrieval     │
       │  │                    │            │                      │
       │  │  Local 2,500+      │            │  82 PDFs → 3,700+    │
       │  │  (NASA CMR, STAC,  │            │  chunks              │
       │  │   CDS, CDSE)       │            │  heading-aware       │
       │  │  + Zenodo runtime  │            │                      │
       │  │                    │            │  semantic similarity │
       │  │  + structured      │            │                      │
       │  │    spatial/temporal│            └──────────┬───────────┘
       │  │    overlap         │                       │
       │  └──────────┬─────────┘                       │
       │             │                                  │
       │             └──────────┬───────────────────────┘
       │                        │
       │                        ▼
       │             ┌─────────────────────────┐
       │             │  Paper–Dataset Linker    │
       │             │  chunk mention > abstract│
       │             │  mention > has_doi       │
       │             │  > semantic              │
       │             └──────────────┬───────────┘
       │                            │
       ▼                            ▼
  ┌─────────────────────────────────────────────────────┐
  │              Evidence Block Assembly                 │
  │                                                      │
  │   --- DATASETS ---                                   │
  │   [DS-1 | id=nasa_cmr_mod13a1_061 | source=nasa_cmr]│
  │       ...                                            │
  │   --- PAPERS ---                                     │
  │   [P-1 | local_id=12 | openalex=w...]                │
  │       Evaluating cumulative... (2023)                │
  │       Evidence level: fulltext_supported             │
  │   --- CHUNKS ---                                     │
  │   [C-1 | chunk_id=drought_012_..._chunk_008]         │
  │       We used MODIS NDVI (MOD13A1)...                │
  └──────────────────────────┬───────────────────────────┘
                             │
                             ▼
             ┌──────────────────────────────┐
             │   Answer Generator            │   OpenAI (gpt-4o-mini)
             │  + grounded prompt (JSON out) │
             │  + grounding validator        │ → FinalAnswer
             │  + methodology_hints (H4)     │   + GroundingReport
             └──────────────────────────────┘
                             │
                             ▼
             ┌──────────────────────────────┐
             │  Evidence Cache Writer        │   snapshots everything
             │  generated/evidence_cache/    │   retrieved this query
             │    <timestamp>_<qhash>/       │   to disk
             └──────────────────────────────┘
```

---

## Core Design: Evidence Cache + Grounding Validator

This is the central differentiator of the system. The project proposal required that the system "only generate results based on retrieved records to avoid fabricated references." We implement this at **three enforcement layers**, not just one:

### 1. What counts as "our database" — the evidence cache

"The corpus" is not just the local PDF library. For any given query, the system's evidence is the **union** of:

| Source | When embedded | Role |
|---|---|---|
| Local PDF chunks (82 papers → 3,700+ chunks) | Offline, once, into ChromaDB | Primary grounding for methodology and paper claims |
| Local dataset catalog (NASA CMR / STAC / Copernicus CDS / CDSE, 2,500+ records) | Offline, once, into ChromaDB | Authoritative dataset descriptions |
| OpenAlex API results for this query | Runtime, embedded on-the-fly | External paper coverage beyond local corpus |
| Zenodo API results for this query | Runtime, embedded on-the-fly | Long-tail dataset discovery |

After retrieval, every surviving candidate is snapshotted to `generated/evidence_cache/<timestamp>_<query_hash>/`:

```
parsed_query.json          openalex.jsonl            zenodo.jsonl
local_datasets.jsonl       chunks.jsonl              evidence_block.txt
final_answer.json          grounding_report.json     manifest.json
```

The `evidence_block.txt` is **verbatim** what was sent to the LLM. Combined with the input JSON files, any query's behavior is fully reproducible offline, and any grounding claim can be audited after the fact.

### 2. The LLM prompt enforces hard / soft / abstention rules

**HARD (absolute)** — the "only use the evidence cache" contract:
- `H1` Every `[DS-N]` / `[P-N]` / `[C-N]` tag must resolve to an entry in the evidence block
- `H2` `recommended_datasets`: only from the DATASETS section above
- `H3` `recommended_papers`: only from the PAPERS section above
- `H4` `methodology_hints`: each hint must cite ≥1 `[C-N]` chunk; if no chunk supports a claim → return `[]`
- `H5` Dataset capabilities / variables / coverage: only what the description states
- `H6` Paper findings: only what abstract or cited chunk states

**SOFT (definitional fallback only)** — narrowly scoped:
- For basic concept definitions in `direct_answer` / `hybrid` mode (e.g. "What is NDVI?"), if no chunk defines the term, the LLM may use standard textbook knowledge but MUST add an explicit flag to `uncertainty_notes`. This does NOT extend to methodology, dataset, or paper claims. Rule `S2` explicitly scopes this fallback so that recommendations and dataset/paper factual claims remain evidence-only.

**Abstention** — behavior when evidence is genuinely absent:
- In recommendation mode, if the evidence block is empty for a given section, the LLM must return `recommended_* = []` and `uncertainty_notes` must state "No corpus evidence matched this query."
- Hard-enforced in code: even if the LLM ignores the rule, `answer_generator.py` forces `recommended_* = []` when all three evidence sections fall below the relevance threshold.

### 3. Code-level verification — `_verify_grounding()`

Prompt instructions are not enough — an LLM can still emit a phantom tag. After the LLM returns its JSON, `_verify_grounding()` in `answer_generator.py`:

1. Parses every `[DS-N]` / `[P-N]` / `[C-N]` tag in `direct_answer` text, `recommended_datasets.ref`, `recommended_papers.ref`, and all `citations` arrays
2. Checks each tag against the actual ID map built from the evidence block
3. Produces a `GroundingReport` with `grounded_ok`, `grounding_rate = tags_found / tags_total`, and a list of specific `violations`
4. Drops any `methodology_hints` that lack a valid `[C-N]` citation (rule H4 enforcement, not just instruction)

The `GroundingReport` is returned to the caller and persisted to the evidence cache, so any hallucinated citation is caught and logged before reaching the user.

**Summary for the "no fabrication" requirement**: recommendations and factual claims go through (1) evidence-only retrieval → (2) prompt-level hard rules → (3) code-level tag verification. Only pure concept definitions can fall back to general knowledge, and they must declare so in `uncertainty_notes`.

---

## Paper Retrieval: Local vs External

Paper candidates come from two provenance classes, each scored with its own formula. "Local dominance" is not a hand-tuned bonus — it falls out of the fact that the two formulas use **mutually exclusive feature sets**, and the local feature set has two features the external one doesn't.

| Provenance | Surfaces when… | Scoring formula | Max score | Slots |
|---|---|---|---|---|
| **local** | At least one of the paper's chunks hit the top chunk retrieval | `0.40·chunk_relevance + 0.25·fulltext_bonus + 0.10·recency + 0.10·impact` | **0.85** | 7 |
| **external** | OpenAlex returned it (either not in local corpus, or in local but no chunk hit) | `0.40·semantic_similarity + 0.10·recency + 0.10·impact` | **0.60** | 3 |

Because the local ceiling (0.85) exceeds the external ceiling (0.60), local papers naturally sort higher under the same weighted-sum framework. No extra tier_bonus or hardcoded ordering is needed.

Quota: 7 local + 3 external = top-10 shown to the LLM. Quota overflow: if one side runs short, leftover slots are filled from the other side by score. Both configurable under `reranking.paper_tier_quota` in `settings.yaml`.

---

## Structured Spatial / Temporal Matching

Mitigates Yu et al. (2025) "text-only retrieval" critique. Instead of embedding `"Region: -90 -180 90 180"` as a string and defaulting `spatial_match=0.5`, the system:

1. Parses dataset bbox strings (three source conventions: NASA CMR space-separated, STAC/CDSE JSON list, Copernicus CDS None)
2. LLM returns `region_bbox` for the query (e.g. `[40, 30, 90, 55]` for Central Asia)
3. Computes IoU + containment overlap as real `spatial_match` score
4. Same for `temporal_match` with open-ended "to present" ranges resolved to `date.today()`

All logic in `spatial_temporal_match.py` — ~200 lines, 4 functions, unit-tested.

---

## Local Data Sources

| Source | Type | Records | Pipeline role |
|--------|------|---------|---------------|
| NASA CMR (expanded) | Dataset metadata | ~2,000 | Persistent local cache |
| STAC collections | Dataset metadata | ~135 | Persistent local cache |
| Copernicus CDS | Dataset metadata | ~133 | Persistent local cache |
| CDSE collections | Dataset metadata | ~270 | Persistent local cache |
| Local PDF library | Full-text papers | 82 papers / 3,700+ chunks | Dual-indexed (paper-level + chunk-level) |
| OpenAlex | Paper metadata | runtime API | Per-query cache |
| Zenodo | Dataset metadata | runtime API | Per-query cache |

---

## Reranking Weights

**Papers** — provenance-specific feature sets (`chunk_relevance` and `semantic_similarity` are mutually exclusive):

| Feature | Weight | Applies to |
|---|---|---|
| `chunk_relevance` | 0.40 | local only |
| `semantic_similarity` | 0.40 | external only |
| `fulltext_bonus` (1.0 / 0.0 binary) | 0.25 | local only |
| `recency_score` | 0.10 | both |
| `impact_score` | 0.10 | both |

Since the two relevance features never apply to the same candidate, there is no double-counting. Max local score = 0.85; max external score = 0.60.

**Datasets** — single weighted sum over 5 features:

| Feature | Weight |
|---|---|
| `semantic_similarity` (metadata embedding cosine) | 0.35 |
| `variable_match` (query variables ∩ dataset keywords/variables) | 0.20 |
| `literature_support` (see tiers below) | 0.25 |
| `spatial_match` (structured bbox IoU + containment) | 0.10 |
| `temporal_match` (structured date-range overlap) | 0.10 |

Datasets also pass through a **hard filter** before scoring: when the query marks `must_have_constraints.region=true` (or `.timescale=true`), candidates whose corresponding match score falls below `must_have_gating.region_threshold` (0.3) / `must_have_gating.temporal_threshold` (0.3) are dropped entirely — not just softly penalized.

### Literature Support Tiers (per dataset)

`literature_support` starts from a **baseline** determined by the candidate's provenance, and is **upgraded** if the dataset is explicitly mentioned in retrieved evidence. Take max:

| Score | Trigger |
|-------|---------|
| **1.0** | A chunk in the evidence block explicitly mentions the dataset (by title or alias) — strongest evidence |
| **0.85** | A retrieved OpenAlex paper's abstract mentions the dataset |
| **0.7** | Baseline for any DOI-bearing candidate — Zenodo records always, local authoritative archives (NASA CMR / STAC / CDS / CDSE) when their DOI field is populated |
| **0.5** | Baseline for candidates that survived purely by vector similarity and have no DOI |

Dataset mention matching uses `local_database/dataset_metadata/dataset_aliases.json` (123 entries mapping common Earth-science product codes — MOD13A1, SMAP, IMERG, ERA5, etc. — to canonical title fragments). Fuzzy title matching is allowed for titles ≤10 words at `partial_ratio > 90`.

The compact [0.5, 1.0] range is intentional: the score distinguishes evidence strength without letting this single feature dominate the weighted sum when other features (semantic match, variable match, spatial/temporal overlap) are decisive.

### Evidence Strength (UI label) — Multi-factor

Final `evidence_strength` shown to users is NOT just `literature_support` — it combines paper evidence with source provenance (Phase 8). A dataset is labeled:

- **high** when `literature_support ≥ 0.8`
- **medium** when:
  - `literature_support ≥ 0.5`, OR
  - the dataset comes from a curated source (NASA CMR / STAC / Copernicus CDS / CDSE / Zenodo) AND has a DOI, OR
  - the dataset comes from a curated source alone, OR
  - it has any DOI
- **low** only when all three signals are absent (no paper mention, no curated source, no DOI)

Rationale: a MODIS product from NASA CMR is a well-known, DOI-backed, institutionally-published dataset; labeling it "low evidence" simply because no retrieved paper happened to cite it by name would be misleading. Phase 8 separates citation backing (`literature_support`) from source provenance.

---

## Setup

### 1. Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) — `$0.10` is enough for one full eval run with `gpt-4o-mini`

### 2. Install dependencies

```bash
cd earth-science-research-assistant-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set environment variables

Create `.env`:

```
OPENAI_API_KEY=sk-proj-...
```

### 4. Preprocessing (one-time)

Builds all indices: paper registry, dataset normalization, PDF extraction, chunk embedding.

```bash
python -m app.preprocess
```

Produces:
- `generated/paper_registry.jsonl` — local_id ↔ openalex_id ↔ title
- `generated/normalized_datasets.jsonl` — unified 2,500+ dataset records
- `generated/chunks.jsonl` — 3,700+ section-aware chunks
- `generated/parsed_papers/<local_id>.json` — cleaned PDF text
- `generated/chroma/` — two ChromaDB collections:
  - `datasets` (2,500+ records)
  - `chunks` (3,700+ records)

### 5. Start the server

```bash
./start.sh
```

API at `http://localhost:8000`, UI at `http://localhost:7860`.

---

## Usage

### Pretty (for humans / debugging)

```bash
curl -s -X POST http://localhost:8000/query/pretty \
  -H "Content-Type: application/json" \
  -d '{"query":"What datasets can I use to study drought in Central Asia?"}'
```

Output includes: recommendations with citation tags, grounding report, evidence cache path.

### JSON (for programmatic use / UI)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What datasets can I use to study drought in Central Asia?"}' | python -m json.tool
```

### Gradio UI (for end users)

`python ui.py` → `http://localhost:7860`. Renders dataset names + DOIs cleanly; hides internal `[DS-N]` tags.

---

## Evaluation

20-query evaluation set in `evaluation/eval_set_v2.json`, 5 categories:

| Category | n | Purpose |
|---|---|---|
| `recommendation_easy` | 8 | Single-topic retrieval (drought, wildfire, precipitation, sea ice, ocean chlorophyll, PM2.5, land cover, soil moisture) |
| `recommendation_hard` | 2 | Cross-topic synthesis (Tibetan Plateau + elevation; groundwater with weak paper support) |
| `direct_answer` | 5 | Definition / explanation (NDVI, SPEI vs SPI, CHIRPS, paper summary, limitations) |
| `hybrid` | 3 | Explanation + recommendation (SPEI, LST + UHI, burned area tropical) |
| `oos` | 2 | Out-of-scope (hydrothermal microbes, earthquakes) — tests abstention |

All gold IDs are verified to exist in the corpus.

### Running the eval

```bash
python -m evaluation.run_eval
```

Runs **V0** (no-RAG baseline) and **V1** (full tiered system) on all 20 queries. Outputs:

- `evaluation/results/per_sample_v0.jsonl`
- `evaluation/results/per_sample_v1.jsonl`
- `evaluation/results/summary.csv`
- `evaluation/results/comparison_table.md`

Metrics:
- **Paper Recall@5 / @10** (gold from local corpus)
- **Dataset Recall@5 / @10**
- **MRR** (papers + datasets)
- **Grounding rate** (fraction of emitted tags that resolve)
- **Methodology cites chunks rate** (fraction of hints with `[C-N]` citation)
- **Abstention rate** (for OOS samples)
- **Latency**

---

## API Reference

### `POST /query`

**Response fields**:
| Field | Description |
|-------|-------------|
| `answer` | LLM-generated plain-text response |
| `answer_mode` | `direct_answer` / `recommendation` / `hybrid` |
| `recommended_datasets` | Dataset objects with `citations` (list of `[C-N]` tags) |
| `recommended_papers` | Paper objects with `evidence_level`, `year`, `citations` |
| `methodology_hints` | Each hint has `hint` text + required `[C-N]` citations |
| `uncertainty_notes` | Caveats about evidence quality or abstention |
| `grounding_report` | `{grounded_ok, grounding_rate, tags_found, tags_total, violations}` |

### `POST /query/pretty`

Same info as plain text.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Project Structure

```
earth-science-research-assistant-rag/
├── app/
│   ├── main.py                   # FastAPI pipeline
│   ├── schemas.py                # Pydantic models (incl. GroundingReport, MethodHint)
│   ├── config.py                 # Settings loader
│   ├── query_parser.py           # OpenAI → ParsedQuery (+ region_bbox)
│   ├── openalex_client.py        # OpenAlex search + single-work lookup
│   ├── zenodo_client.py          # Zenodo dataset API
│   ├── dataset_normalizer.py     # Unifies 4 metadata source schemas
│   ├── embedder.py               # sentence-transformer + ChromaDB (datasets/chunks)
│   ├── dataset_retriever.py      # Local + Zenodo dataset retrieval + must-have hard filter
│   ├── chunk_retriever.py        # Chunk vector search
│   ├── paper_matcher.py          # OpenAlex ↔ local PDF fuzzy match
│   ├── linker.py                 # Paper-dataset linker (chunk/abstract mention → literature_support)
│   ├── reranker.py               # Provenance-aware paper reranker (local/external) + dataset reranker
│   ├── spatial_temporal_match.py # Structured bbox overlap + date-range overlap
│   ├── answer_generator.py       # [Phase 2] Grounded prompt + JSON output + _verify_grounding
│   ├── evidence_cache_writer.py  # [Phase 3] Per-query evidence snapshot
│   ├── pdf_extractor.py          # PyMuPDF + pdfplumber
│   ├── chunker.py                # Heading-aware + fixed-size fallback chunking
│   ├── paper_registry.py         # id_track.xlsx → paper_registry.jsonl
│   └── preprocess.py             # One-shot indexing pipeline
├── config/
│   └── settings.yaml             # All weights, model names, paths
├── local_database/
│   ├── dataset_metadata/
│   │   ├── nasa_cmr_expanded_metadata.json
│   │   ├── stac_metadata.json
│   │   ├── copernicus_cds_metadata.json
│   │   ├── CDSE_collections.json
│   │   └── dataset_aliases.json  # [Phase 8] 123 alias → canonical fragment mappings
│   ├── fulltext_paper/           # 82 PDFs + id_track.xlsx
│   ├── fetch_metadata.py         # Re-fetch dataset metadata
│   ├── fetch_papers.py           # Re-fetch and download papers
│   └── paper_manifest.json       # Paper index (status, DOI, filename)
├── evaluation/
│   ├── eval_set_v2.json          # 20-query eval set with gold IDs
│   ├── metrics.py                # Recall@k, MRR, grounding rate, abstention
│   ├── run_eval.py               # V0 + V1 runner
│   └── results/                  # per_sample_v0/v1.jsonl, summary.csv, comparison_table.md
├── generated/                    # ChromaDB + all derived artifacts (git-ignored)
│   ├── chroma/                   # Three collections
│   ├── evidence_cache/           # [Phase 3] Per-query snapshots
│   ├── parsed_papers/            # <local_id>.json per paper
│   ├── chunks.jsonl
│   ├── paper_registry.jsonl
│   └── normalized_datasets.jsonl
├── requirements.txt
├── start.sh                      # Launches FastAPI + Gradio UI
├── ui.py                         # Gradio user-facing UI
└── .env                          # OPENAI_API_KEY
```

---

## Configuration

All in `config/settings.yaml`. Key options:

```yaml
llm:
  provider: openai
  default_model: gpt-4o-mini
  temperature: 0.1
  max_output_tokens: 1024

retrieval:
  dataset_top_k: 15
  chunk_top_k: 20

reranking:
  paper_weights:
    chunk_relevance:      0.40   # local only
    semantic_similarity:  0.40   # external only (mutually exclusive)
    fulltext_bonus:       0.25   # 1.0/0.0 binary, local only
    recency_score:        0.10
    impact_score:         0.10
  paper_tier_quota:
    local:    7   # chunk-backed local papers
    external: 3   # OpenAlex metadata-only supplement
  dataset_weights:
    semantic_similarity: 0.35
    variable_match:      0.20
    literature_support:  0.25
    spatial_match:       0.10
    temporal_match:      0.10
  must_have_gating:
    region_threshold:    0.3   # hard-drop if must_have.region=true and spatial_match < this
    temporal_threshold:  0.3   # hard-drop if must_have.timescale=true and temporal_match < this
  literature_support_scores:
    chunk_explicit_mention: 1.0
    abstract_mention:       0.85
    has_doi:                0.7
    semantic_only:          0.5
```

---

## Changelog

- **Phase 1** — Migrated LLM provider from Gemini to OpenAI (`gpt-4o-mini`)
- **Phase 2** — Rewrote answer_generator with strict grounding prompt, stable ID citation, JSON output, `_verify_grounding()` validator; fixed hardcoded empty `methodology_hints`
- **Phase 3** — Evidence cache writer (per-query snapshot to `generated/evidence_cache/<query_id>/`)
- **Phase 4** — Structured spatial / temporal matching (parse bbox strings, LLM-emitted `region_bbox`, IoU + date-range overlap)
- **Phase 5** — `run_eval.py` with V0 (no-RAG) vs V1 comparison, programmatic metrics
- **Phase 6** — Local-first tiered paper retrieval (A: chunk-backed, B: semantic local, C: OpenAlex supplement), 4+3+3 slot quota
- **Phase 7** — Paper metadata enrichment via OpenAlex (real year + abstract), filename fallback only for year
- **Phase 8** — Multi-factor evidence_strength (literature_support + source provenance + DOI); `dataset_aliases.json` (123 entries, MODIS/VIIRS/Sentinel/GPM/SMAP/ERA5/etc.) enables proper alias-based chunk mention matching; fuzzy title match relaxed to 10 words
- **Phase 9 (refactor)** — Unified the reranker around a single consistent mental model:
  - **Paper** scoring split by provenance into two mutually-exclusive feature sets (local uses `chunk_relevance + fulltext_bonus`, external uses `semantic_similarity`); removed the old `tier_bonus` since local dominance now falls out of the formula ceilings (0.85 vs 0.60). Tier A/B/C collapsed into **local / external** with quota 7 + 3.
  - **Dataset** `literature_support` switched to a **baseline-plus-upgrade** scheme: `has_doi` (0.7) is the new baseline for any DOI-bearing candidate, upgraded to `abstract_mention` (0.85) or `chunk_explicit_mention` (1.0) if the dataset is named in retrieved evidence. The never-firing `zenodo_doi_matches_openalex` branch was removed as dead code.
  - **Must-have hard filter** — when the query marks `region` or `timescale` as must-have, candidates below the threshold are dropped before scoring instead of soft-penalized.
  - **Dataset dedup** (e.g. ERA5 canonical vs regional subset) moved from a planned code-level entity-resolution step into a prompt rule inside `answer_generator` — the LLM decides which version to recommend given titles, DOIs, and sources.
  - Deleted paper-level semantic retrieval (`local_paper_retriever.py`, `local_paper_index.py`) — a paper now surfaces only via chunk hits, which is strictly stronger evidence.

---

## Notes

- **Grounding guarantee**: every `[DS-N]/[P-N]/[C-N]` tag the LLM emits must resolve to the evidence block or it's flagged in `grounding_report.violations`. Typical queries achieve 100 % grounding rate.
- **Evidence cache**: every query dumps its full retrieval state to `generated/evidence_cache/<timestamp>_<qhash>/` — `parsed_query.json`, `openalex.jsonl`, `zenodo.jsonl`, `local_datasets.jsonl`, `chunks.jsonl`, `evidence_block.txt` (exactly what the LLM saw), `final_answer.json`, `grounding_report.json`, `manifest.json`. Enables reproducible evaluation and forensic debugging.
- **Local-first**: local PDFs are guaranteed 7 / 10 evidence-block paper slots; OpenAlex fills the remaining 3. The split is enforced by quota, and local papers' score ceiling (0.85) already exceeds the external ceiling (0.60) from the formula design, so even without the quota local would usually come first.
- **Evidence strength decoupling** (Phase 8): the UI label "high / medium / low" is multi-factor and considers source provenance, not just paper citation signal. Datasets from curated authorities (NASA CMR, STAC, Copernicus CDS, CDSE, Zenodo) with a DOI always reach at least "medium" even when no retrieved paper mentions them by name.
- **Zenodo embeddings are on-the-fly**: unlike the 2,500+ local dataset records (embedded once into ChromaDB during preprocessing), Zenodo's ~10 runtime records are embedded with the same model (`BAAI/bge-small-en-v1.5`) on each query so their cosine similarities are directly comparable to the ChromaDB distances.
- **Dataset deduplication is LLM-side**: the system does not do cross-source entity resolution between (e.g.) a canonical ERA5 and a Zenodo-hosted subset. Instead, the `answer_generator` prompt instructs the LLM to recommend only one version of each underlying resource, preferring the authoritative/canonical source unless the subset specifically matches the query scope.
- **Filename mislabeling in corpus**: some files have topic-incorrect filenames (e.g. `soil_moisture_guenther_2006.pdf` is actually the MEGAN isoprene paper). The system uses `id_track.xlsx.original_title` and actual PDF content — filename is only a last-resort year fallback, never a title source.
