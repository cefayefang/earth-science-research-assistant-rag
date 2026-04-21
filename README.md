# Earth Science Research Assistant

A hybrid RAG (Retrieval-Augmented Generation) system that helps researchers find relevant datasets and papers for Earth science questions. Built for Harvard EPS 210.

---

## What It Does

Given a natural-language research question, the system:

1. Parses the query into structured intent, variables, and region
2. Fetches recent and highly-cited papers from **OpenAlex**
3. Searches **Zenodo** for open datasets and code repositories
4. Retrieves semantically similar chunks from a **local PDF library** (82 papers)
5. Queries a **local dataset metadata index** (2,500+ records from NASA CMR, STAC, Copernicus CDS, CDSE)
6. Links datasets to supporting literature
7. Reranks everything with a deterministic weighted scoring formula
8. Generates a grounded answer via **Gemini 2.5 Flash**

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Parser   │  → Gemini LLM → structured ParsedQuery
│  (query_parser) │    (intent, variables, region, timescale)
└────────┬────────┘
         │
    ┌────┴─────────────────────────────────┐
    │                                      │
    ▼                                      ▼
┌──────────────┐                  ┌──────────────────┐
│   OpenAlex   │  runtime API     │     Zenodo        │  runtime API
│  (papers)    │──────────────┐   │  (datasets/code)  │──────────┐
└──────────────┘              │   └──────────────────┘          │
                              │                                  │
    ┌─────────────────────────┼──────────────────────────────────┘
    │                         │
    ▼                         ▼
┌──────────────────┐   ┌───────────────────┐
│  Local ChromaDB  │   │  Local ChromaDB   │
│  chunks          │   │  dataset metadata │
│  (3,700+ chunks) │   │  (2,500+ records) │
└────────┬─────────┘   └────────┬──────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────┐
         │    Reranker       │
         │  papers: 5-factor │
         │  datasets: 5-factor│
         └────────┬──────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Answer Generator │  → Gemini LLM → final response
         └──────────────────┘
```

### Local Data Sources

| Source | Type | Records |
|--------|------|---------|
| NASA CMR (expanded) | Dataset metadata | ~2,000 |
| STAC collections | Dataset metadata | ~135 |
| Copernicus CDS | Dataset metadata | ~133 |
| CDSE collections | Dataset metadata | ~270 |
| Local PDF library | Full-text papers | 82 papers / 3,700+ chunks |

### Reranking Weights

**Papers** (5 factors):
- 35% semantic similarity
- 25% chunk relevance (full-text evidence)
- 15% recency score
- 15% citation impact
- 10% full-text bonus

**Datasets** (5 factors):
- 35% semantic similarity
- 20% variable match
- 20% literature support
- 15% spatial match
- 10% temporal match

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (free education plan works)

### 2. Install dependencies

```bash
cd earth-science-research-assistant-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Run preprocessing (one-time)

This step builds the ChromaDB vector index from local PDFs and dataset metadata. Only needs to be re-run when you add new data.

```bash
source venv/bin/activate
python3 -m app.preprocess
```

Expected output:
- `generated/paper_registry.jsonl` — indexed paper metadata
- `generated/normalized_datasets.jsonl` — unified dataset records
- `generated/chunks.jsonl` — PDF text chunks
- `generated/chroma/` — ChromaDB vector store

### 5. Start the server

```bash
./start.sh
```

Or manually:

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at `http://localhost:8000`.

---

## Usage

### Option A — Pretty text output (recommended)

```bash
curl -s -X POST http://localhost:8000/query/pretty \
  -H "Content-Type: application/json" \
  -d '{"query": "What datasets can I use to study drought impacts on vegetation in Central Asia?"}'
```

Example output:
```
============================================================
QUERY: What datasets can I use to study drought impacts on vegetation in Central Asia?
MODE:  recommendation
============================================================

Based on the retrieved evidence, the following datasets are
well-suited for studying drought-vegetation interactions...

────────────────────────────────────────────────────────────
RECOMMENDED DATASETS
────────────────────────────────────────────────────────────
1. [NASA] MODIS Vegetation Indices (MOD13)
   Evidence: high | Score: 0.82
   DOI: 10.5067/MODIS/MOD13A2.061

2. [COPERNICUS] Drought Indices (SPI, SPEI)
   Evidence: medium | Score: 0.71
...

────────────────────────────────────────────────────────────
RECOMMENDED PAPERS
────────────────────────────────────────────────────────────
1. [✓ fulltext] Vegetation response to drought in Central Asia (2022)
2. [✓ fulltext] NDVI trends and climate variability (2021)
...
============================================================
```

### Option B — JSON output (for programmatic use)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What datasets can I use to study drought impacts on vegetation in Central Asia?"}' \
  | python3 -m json.tool
```

### Option C — Interactive docs

Open `http://localhost:8000/docs` in your browser for the Swagger UI.

---

## API Reference

### `POST /query`

Returns a structured JSON response.

**Request body:**
```json
{ "query": "your research question" }
```

**Response fields:**
| Field | Description |
|-------|-------------|
| `answer` | LLM-generated response text |
| `answer_mode` | `direct_answer`, `recommendation`, or `hybrid` |
| `recommended_datasets` | Up to 5 ranked datasets with scores and DOIs |
| `recommended_papers` | Up to 5 ranked papers with evidence levels |
| `methodology_hints` | Suggested analysis approaches |
| `uncertainty_notes` | Caveats about evidence quality |

### `POST /query/pretty`

Returns the same result as plain text, formatted for readability.

### `GET /health`

Returns `{"status": "ok"}` if the server is running.

---

## Project Structure

```
earth-science-research-assistant-rag/
├── app/
│   ├── main.py              # FastAPI endpoints
│   ├── schemas.py           # Pydantic data models
│   ├── config.py            # Settings loader
│   ├── query_parser.py      # LLM-based query parsing
│   ├── openalex_client.py   # OpenAlex API client
│   ├── zenodo_client.py     # Zenodo API client
│   ├── dataset_normalizer.py# Unifies 4 metadata sources
│   ├── embedder.py          # Sentence-transformer embeddings
│   ├── dataset_retriever.py # Vector search over datasets
│   ├── chunk_retriever.py   # Vector search over PDF chunks
│   ├── paper_matcher.py     # Matches OpenAlex papers to local PDFs
│   ├── linker.py            # Links datasets to literature
│   ├── reranker.py          # Weighted scoring for papers & datasets
│   ├── answer_generator.py  # Gemini LLM answer generation
│   ├── pdf_extractor.py     # PDF text extraction (PyMuPDF + pdfplumber)
│   ├── chunker.py           # Heading-aware text chunking
│   └── preprocess.py        # One-time indexing pipeline
├── config/
│   └── settings.yaml        # All configuration
├── local_database/
│   ├── dataset_metadata/    # 4 JSON metadata files (git-ignored)
│   ├── fulltext_paper/      # Local PDF library (git-ignored)
│   ├── fetch_metadata.py    # Script to re-fetch dataset metadata
│   ├── fetch_papers.py      # Script to re-fetch and download papers
│   ├── paper_manifest.json  # Index of all papers (status, DOI, filename)
│   └── manual_download_list.md  # Paywalled papers needing manual download
├── evaluation/
│   └── eval_set_v2.json     # Evaluation query set
├── generated/               # ChromaDB, chunks, registries (git-ignored)
├── requirements.txt
├── start.sh                 # One-command server startup
└── .env                     # API keys (git-ignored)
```

---

## Configuration

All settings are in `config/settings.yaml`. Key options:

```yaml
llm:
  default_model: gemini-2.5-flash   # Gemini model to use
  temperature: 0.1

retrieval:
  dataset_top_k: 15   # candidates before reranking
  chunk_top_k: 20

openalex:
  recent_top_k: 20    # most recent papers fetched
  impactful_top_k: 20 # most cited papers fetched

zenodo:
  top_k: 10           # Zenodo datasets fetched per query
```

---

## Notes

- **Rate limits**: Gemini 2.5 Flash education plan allows 5 requests/min and 20 requests/day. Space out queries accordingly.
- **Preprocessing**: Must be run once before starting the server. Re-run only when adding new PDFs or metadata files.
- **Evidence levels**: Papers with local full-text (`fulltext_supported`) provide stronger evidence than `metadata_only` matches.
- **Literature support tiers**: Zenodo DOI match (1.0) > chunk explicit mention (0.8) > abstract mention (0.5) > semantic similarity only (0.2).
