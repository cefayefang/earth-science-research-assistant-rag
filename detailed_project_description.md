# Earth Science Research Assistant

A hybrid RAG (Retrieval-Augmented Generation) system that helps researchers find relevant datasets and papers for Earth science questions. Built for Harvard EPS 210.

The system implements a **local-first retrieval architecture** with strict LLM grounding: every recommendation, citation, and methodology claim is traceable to a record in the evidence cache, and every emitted citation tag is validated at code level before the answer is returned.

---

## What It Does

Given a natural-language research question, the system:

1. **Classifies intent** pre-RAG into one of 5 routes (chitchat / new_question / re_recommend / detail_followup / out_of_scope); only real research questions run the full pipeline
2. **Parses the query** (history-aware) into structured intent, variables, region (with bounding box), and timescale — short follow-ups inherit topic context from prior turns
3. **Local-first paper retrieval** via chunk-level vector search over 82 full-text PDFs
4. **Runtime API retrieval** from OpenAlex (papers) and Zenodo (datasets) as supplementary coverage
5. **Local dataset metadata** search over 2,500+ records (NASA CMR, STAC, Copernicus CDS, CDSE)
6. **Structured spatial / temporal matching** — real bbox overlap + date-range overlap, plus hard filtering when the query's region/timescale is marked as must-have
7. **Provenance-aware paper reranking** — local papers and external OpenAlex results scored with separate feature sets so local full-text evidence naturally dominates (no hand-tuned tier bonus)
8. **Grounded answer generation** — LLM constrained to the merged evidence cache; every citation is validated by `_verify_grounding()` before return
9. **Per-query evidence cache** snapshot to disk for full reproducibility
10. **Session state** round-trips to the client so follow-ups ("tell me about paper 1", "more datasets", "第二篇") resolve against what the user actually saw in the previous turn

---

## Architecture

```
                       User Query  +  SessionState (from client)
                           │
                           ▼
            ┌──────────────────────────┐
            │   Intent Classifier       │   OpenAI (gpt-4o-mini)
            │   (intent_classifier.py)  │ → chitchat / new_question /
            └────────────┬──────────────┘   re_recommend /
                         │                  detail_followup / out_of_scope
         ┌───────────────┼────────────────┐
         │               │                │
    chitchat /     detail_followup    new_question /
    out_of_scope   (per-paper chunk    re_recommend
    (short reply,   retrieval OR       (full RAG below,
     no RAG)        cached metadata)   re_recommend adds
                                        exclusion list)
                         │
                         ▼
            ┌──────────────────────────┐
            │     Query Parser          │   OpenAI (gpt-4o-mini)
            │     (query_parser.py)     │ → ParsedQuery + region_bbox
            │  (inherits phenomenon /   │   (history-aware for short
            │   variables / region /    │    follow-ups)
            │   timescale from history) │
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

## Multi-Turn Conversation Layer

The system is not single-shot. Every turn goes through a pre-RAG **intent classifier** and a **session state** that round-trips between the UI / API client and the backend. This lets the system handle "more papers", "tell me about the second dataset", and even short one-word follow-ups — without running the full RAG pipeline on every message, and without losing track of what was already recommended.

The backend is stateless: `SessionState` is sent by the client with each `/query` request and returned updated in the response. The UI holds it in a `gr.State`; programmatic clients echo it back verbatim.

### Intent Classifier (Pre-RAG Router)

`pipeline/intent_classifier.py` runs one fast LLM call on `(recent_history, latest_message)` before any retrieval, classifying the turn into one of 5 routes. Only `new_question` and `re_recommend` trigger the full RAG pipeline; the other three have their own handlers:

| Intent | Trigger examples | Handler behavior |
|---|---|---|
| `chitchat` | "hi", "thanks!" | `_handle_chitchat` — short LLM reply using last 6 history messages, no retrieval, no evidence cache |
| `out_of_scope` | "cook pasta", "weather today" | `_handle_out_of_scope` — canned polite refusal |
| `new_question` | "What datasets can I use for drought in Central Asia?" | Full RAG |
| `re_recommend` | "more papers", "any other datasets?", "different ones" | Full RAG + `wants_fresh=True` + `exclude_*_ids` from session state; intent classifier also supplies a `rewritten_query` that restores the topic from prior turns |
| `detail_followup` | "tell me about paper 1", "what variables does that dataset cover?", "第二篇" | `_handle_detail_followup` — three-path resolver described below |

The classifier also extracts:
- `target_ref` + `target_kind` (`paper` / `dataset` / null) — which item the user is asking about
- `requested_count` + `requested_count_target` — "give me 3 papers", "two datasets", "four". Recognized forms: digits, English number words (`two`–`five`), and this is threaded into `answer_generator` as a `COUNT CONSTRAINT` section that overrides the intent template's default range.

### Session State (round-tripped between client and backend)

Defined in `core/schemas.py::SessionState`:

| Field | Purpose |
|---|---|
| `recommended_paper_ids` / `recommended_dataset_ids` | Accumulated across turns. Used as `exclude_*` on `re_recommend` so the user doesn't see the same recommendations twice. |
| `last_recommended_papers` / `last_recommended_datasets` | Positional snapshots (position 1, 2, 3…) of the most recent recommendation list. Let "paper 1" / "the second dataset" resolve to a concrete item. |
| `last_turn_chunks` | Up to 10 chunks retrieved last turn, cached as `CachedChunk` objects for keyword-fallback follow-ups. |
| `last_turn_ephemeral_dataset_metadata` | Zenodo records' metadata for any dataset that ended up in the last turn's recommendation list. Zenodo is a runtime source (not in the normalized local catalog), so caching it per-turn avoids re-hitting the Zenodo API when the user asks a follow-up about a Zenodo dataset. Authoritative sources (NASA CMR / STAC / CDS / CDSE) are not cached here — they're re-loaded from `load_normalized_datasets()`. |
| `turn_count` | Incremented each turn, for logging / UI. |

On `new_question` and `re_recommend`, `_update_session` rebuilds the state from the new answer. On `detail_followup`, the prior positional list is preserved (so a second follow-up like "what about the third paper?" still works) but `last_turn_chunks` may be updated if the per-paper retrieval surfaced new chunks.

### Detail Follow-up: three resolution paths

`router.py::_handle_detail_followup` — triggered when `intent_type == "detail_followup"`:

1. **`parse_target_position`** extracts a 1-indexed position from `target_ref`. Supported forms: `paper 1` / `dataset 2` (explicit kind-N wins), English ordinals (`first` / `1st` / `second` / `2nd` / …), and **Chinese ordinals** (`第一` / `第二` / `第三` / `第四` / `第五`).
2. **If `target_kind == "dataset"`** (or the matched position is a dataset): `_answer_from_dataset` looks up the dataset in `load_normalized_datasets()` or the ephemeral Zenodo cache, builds a metadata-only evidence block (provider, DOI, variables, keywords, spatial/temporal info, description), optionally attaches 1–3 chunks from `last_turn_chunks` that mention the dataset's name or top keywords as "usage context", and answers with a strict "only use this metadata" prompt. No full RAG. No Zenodo re-fetch.
3. **If the matched position is a local paper** (has `local_id`): `chunk_retriever.retrieve_chunks_for_paper` runs a semantic query **scoped to just that paper's chunks** (Chroma `where={"local_id": ...}`). This is strictly stronger than reusing `last_turn_chunks`, because the earlier turn's top-10 may not include the specific chunks the user's detail question needs. `_answer_from_chunks` then composes an answer from up to 5 of those chunks.
4. **If the matched position is an external paper** (OpenAlex, no `local_id`): the query is rewritten as `"{user_query} — regarding the paper titled \"{title}\""` and sent through the full RAG pipeline, so chunk retrieval has a chance to surface related local passages.
5. **Fallback chain**: if no position parses, or the position didn't match anything, fall back to keyword matching on `last_turn_chunks` via `find_chunks_for_target`; if that also returns nothing, fall back to full RAG with an enriched query.

### History-aware query parsing

When the intent is `new_question` or `re_recommend`, `pipeline/query_parser.py` receives the history alongside the current message. Folded into user/assistant pairs (last 3 verbatim, assistant responses truncated to 400 chars), the history is injected into the parsing prompt with explicit guidance:

> Because prior turns are shown, first think (silently) about how the CURRENT query relates to them — continuation, focus shift, or new topic. Then let that understanding flow into the fields below … `local_query` / `openalex_query` / `zenodo_query`: write the COMPLETE retrieval string even for short follow-ups. Inherit the established phenomenon, variables, region, and timescale from prior turns; the retrieval layer has no other way to recover them.

Without this, a follow-up like "what about since 2000?" would parse into a `local_query` of literally "what about since 2000?" and lose all topical context. With inheritance, it becomes something like "Arctic sea ice loss vegetation NDVI since 2000".

Routing decisions (`wants_fresh`, `requested_count`) are NOT made here — the intent classifier is the single source of truth, and the router stamps its verdicts onto the returned `ParsedQuery`.

### Quick sequence example

```
Turn 1 user: "What datasets can I use to study drought in Central Asia?"
  → intent=new_question → full RAG
  → state: recommended_dataset_ids=[cmr_mod13a1..., cmr_chirps..., ...],
           last_recommended_datasets=[{pos:1, MOD13A1}, {pos:2, CHIRPS}, ...]

Turn 2 user: "tell me about the second one"
  → intent=detail_followup, target_kind=null, target_ref="the second one"
  → parse_target_position → 2
  → matches last_recommended_datasets[pos=2] = CHIRPS
  → _answer_from_dataset with CHIRPS metadata + mention chunks
  → no retrieval, no evidence cache

Turn 3 user: "more datasets"
  → intent=re_recommend, rewritten_query="drought Central Asia vegetation..."
  → full RAG with exclude_dataset_ids=[cmr_mod13a1..., cmr_chirps..., ...]
  → user sees genuinely new recommendations

Turn 4 user: "第一篇 paper 用了什么方法"
  → intent=detail_followup, target_kind="paper", target_ref="第一篇 paper"
  → parse_target_position recognizes "第一" → 1
  → retrieve_chunks_for_paper(local_id of paper #1, question) → top-5 chunks
  → _answer_from_chunks
```

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
python -m app.ingestion.preprocess
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
│   ├── main.py                        # FastAPI app + HTTP endpoints (thin)
│   ├── router.py                      # Intent routing, session management, RAG orchestration
│   ├── core/
│   │   ├── config.py                  # Settings loader + OpenAI client
│   │   ├── schemas.py                 # All Pydantic models (GroundingReport, SessionState, etc.)
│   │   └── spatial_temporal_match.py  # Structured bbox overlap + date-range overlap
│   ├── clients/
│   │   ├── openalex_client.py         # OpenAlex search + single-work lookup
│   │   └── zenodo_client.py           # Zenodo dataset API
│   ├── ingestion/                     # One-time preprocessing (run before serving)
│   │   ├── preprocess.py              # Orchestrates steps 1–4 below
│   │   ├── paper_registry.py          # id_track.xlsx → paper_registry.jsonl + OpenAlex enrichment
│   │   ├── dataset_normalizer.py      # Unifies 4 metadata source schemas
│   │   ├── pdf_extractor.py           # PyMuPDF + pdfplumber
│   │   ├── chunker.py                 # Heading-aware + fixed-size fallback chunking
│   │   └── embedder.py                # sentence-transformer + ChromaDB (datasets/chunks)
│   └── pipeline/                      # Per-query RAG stages
│       ├── intent_classifier.py       # Pre-RAG routing (chitchat/new/re_recommend/followup/oos)
│       ├── query_parser.py            # OpenAI → ParsedQuery (+ region_bbox)
│       ├── paper_matcher.py           # OpenAlex ↔ local PDF fuzzy match
│       ├── chunk_retriever.py         # Chunk vector search (global + per-paper)
│       ├── dataset_retriever.py       # Local + Zenodo retrieval + must-have hard filter
│       ├── linker.py                  # Paper-dataset linker (chunk/abstract mention → literature_support)
│       ├── reranker.py                # Provenance-aware paper reranker + dataset reranker
│       ├── answer_generator.py        # Grounded prompt + JSON output + _verify_grounding
│       └── evidence_cache_writer.py   # Per-query evidence snapshot to disk
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

## Notes

- **Grounding guarantee**: every `[DS-N]/[P-N]/[C-N]` tag the LLM emits must resolve to the evidence block or it's flagged in `grounding_report.violations`. Typical queries achieve 100 % grounding rate.
- **Evidence cache**: every query dumps its full retrieval state to `generated/evidence_cache/<timestamp>_<qhash>/` — `parsed_query.json`, `openalex.jsonl`, `zenodo.jsonl`, `local_datasets.jsonl`, `chunks.jsonl`, `evidence_block.txt` (exactly what the LLM saw), `final_answer.json`, `grounding_report.json`, `manifest.json`. Enables reproducible evaluation and forensic debugging.
- **Local-first**: local PDFs are guaranteed 7 / 10 evidence-block paper slots; OpenAlex fills the remaining 3. The split is enforced by quota, and local papers' score ceiling (0.85) already exceeds the external ceiling (0.60) from the formula design, so even without the quota local would usually come first.
- **Evidence strength decoupling** (Phase 8): the UI label "high / medium / low" is multi-factor and considers source provenance, not just paper citation signal. Datasets from curated authorities (NASA CMR, STAC, Copernicus CDS, CDSE, Zenodo) with a DOI always reach at least "medium" even when no retrieved paper mentions them by name.
- **Zenodo embeddings are on-the-fly**: unlike the 2,500+ local dataset records (embedded once into ChromaDB during preprocessing), Zenodo's ~10 runtime records are embedded with the same model (`BAAI/bge-small-en-v1.5`) on each query so their cosine similarities are directly comparable to the ChromaDB distances.
- **Dataset deduplication is LLM-side**: the system does not do cross-source entity resolution between (e.g.) a canonical ERA5 and a Zenodo-hosted subset. Instead, the `answer_generator` prompt instructs the LLM to recommend only one version of each underlying resource, preferring the authoritative/canonical source unless the subset specifically matches the query scope.
- **Filename mislabeling in corpus**: some files have topic-incorrect filenames (e.g. `soil_moisture_guenther_2006.pdf` is actually the MEGAN isoprene paper). The system uses `id_track.xlsx.original_title` and actual PDF content — filename is only a last-resort year fallback, never a title source.
