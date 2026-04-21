from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from google.genai import errors as genai_errors
from .schemas import QueryRequest, QueryResponse
from .query_parser import parse_query
from .openalex_client import fetch_openalex_papers
from .paper_matcher import match_papers
from .dataset_retriever import retrieve_datasets
from .chunk_retriever import retrieve_chunks
from .linker import build_links
from .reranker import rerank_papers, rerank_datasets
from .answer_generator import generate_answer

app = FastAPI(title="Earth Science Research Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_pipeline(user_query: str):
    parsed = parse_query(user_query)

    openalex_papers = []
    if parsed.openalex_query:
        openalex_papers = fetch_openalex_papers(parsed.openalex_query)

    openalex_dois = {p.doi.lower() for p in openalex_papers if p.doi}
    paper_matches = match_papers(openalex_papers)
    dataset_candidates = retrieve_datasets(parsed, openalex_dois)
    chunk_candidates = retrieve_chunks(parsed)
    build_links(dataset_candidates, chunk_candidates, openalex_papers)
    ranked_papers = rerank_papers(openalex_papers, paper_matches, chunk_candidates, parsed.local_query)
    ranked_datasets = rerank_datasets(dataset_candidates)
    answer = generate_answer(parsed, ranked_papers[:10], ranked_datasets[:10], chunk_candidates[:10])
    return parsed, ranked_papers, ranked_datasets, chunk_candidates, answer


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        _, ranked_papers, ranked_datasets, chunk_candidates, answer = _run_pipeline(request.query)
        return QueryResponse(
            query=request.query,
            answer=answer.final_text,
            answer_mode=answer.answer_mode,
            recommended_datasets=answer.recommended_datasets,
            recommended_papers=answer.recommended_papers,
            methodology_hints=answer.methodology_hints,
            uncertainty_notes=answer.uncertainty_notes,
        )
    except genai_errors.ClientError as e:
        code = e.code if hasattr(e, "code") else 0
        if code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(
                status_code=429,
                detail="Gemini API daily quota exceeded (free tier: 20 req/day). Please wait and try again later.",
            )
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _format_pretty(request: QueryRequest) -> str:
    _, ranked_papers, ranked_datasets, _, answer = _run_pipeline(request.query)

    lines = []
    lines.append("=" * 60)
    lines.append(f"QUERY: {request.query}")
    lines.append(f"MODE:  {answer.answer_mode}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(answer.final_text)
    lines.append("")

    if answer.recommended_datasets:
        lines.append("─" * 60)
        lines.append("RECOMMENDED DATASETS")
        lines.append("─" * 60)
        for i, d in enumerate(answer.recommended_datasets, 1):
            lines.append(f"{i}. [{d.source.upper()}] {d.dataset_name}")
            lines.append(f"   Evidence: {d.evidence_strength} | Score: {d.reason.split(';')[0].replace('Relevance score ','')}")
            if d.doi:
                lines.append(f"   DOI: {d.doi}")

    if answer.recommended_papers:
        lines.append("")
        lines.append("─" * 60)
        lines.append("RECOMMENDED PAPERS")
        lines.append("─" * 60)
        for i, p in enumerate(answer.recommended_papers, 1):
            tag = "✓ fulltext" if p.evidence_level == "fulltext_supported" else "  metadata"
            lines.append(f"{i}. [{tag}] {p.title} ({p.year})")

    if answer.uncertainty_notes:
        lines.append("")
        lines.append("─" * 60)
        lines.append("UNCERTAINTY NOTES")
        lines.append("─" * 60)
        for note in answer.uncertainty_notes:
            lines.append(f"• {note}")

    lines.append("=" * 60)
    return "\n".join(lines)


@app.post("/query/pretty", response_class=PlainTextResponse)
def query_pretty(request: QueryRequest):
    try:
        return _format_pretty(request)
    except genai_errors.ClientError as e:
        code = e.code if hasattr(e, "code") else 0
        if code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(
                status_code=429,
                detail="Gemini API daily quota exceeded (free tier: 20 req/day). Please wait and try again later.",
            )
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
