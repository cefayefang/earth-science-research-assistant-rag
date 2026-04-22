from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from openai import OpenAIError, RateLimitError
from .schemas import QueryRequest, QueryResponse
from .query_parser import parse_query
from .openalex_client import fetch_openalex_papers
from .paper_matcher import match_papers
from .dataset_retriever import retrieve_datasets
from .chunk_retriever import retrieve_chunks
from .linker import build_links
from .reranker import rerank_papers, rerank_datasets
from .answer_generator import generate_answer
from .evidence_cache_writer import write_evidence_cache

app = FastAPI(title="Earth Science Research Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_pipeline(
    user_query: str,
    history: list | None = None,
    exclude_paper_ids: list[str] | None = None,
    exclude_dataset_ids: list[str] | None = None,
):
    """Full pipeline. Returns a dict with everything needed for response + eval.

    - `history`: prior ConversationMessage dicts (most recent last). Used by the
      Stage-1 conversation analyzer inside parse_query.
    - `exclude_*_ids`: IDs from the immediately previous turn's recommendations.
      These are only ACTUALLY applied as filters when the conversation analyzer
      decides the current query is an EXPANSION follow-up
      (parsed.wants_fresh_recommendations == True). For focus shifts or
      drill-downs the excludes are ignored — the user likely still wants the
      previous items in context.
    """
    parsed, digest = parse_query(user_query, history=history)

    # Only honor the UI-supplied exclude lists when the analyzer confirms the
    # user wants fresh items. Otherwise a drill-down like "tell me about the
    # first paper" would wrongly exclude that paper from the candidate pool.
    effective_exclude_papers = (
        exclude_paper_ids if (parsed.wants_fresh_recommendations and exclude_paper_ids) else None
    )
    effective_exclude_datasets = (
        exclude_dataset_ids if (parsed.wants_fresh_recommendations and exclude_dataset_ids) else None
    )

    openalex_papers = []
    if parsed.openalex_query:
        openalex_papers = fetch_openalex_papers(parsed.openalex_query)

    openalex_dois = {p.doi.lower() for p in openalex_papers if p.doi}
    paper_matches = match_papers(openalex_papers)
    dataset_candidates, zenodo_records = retrieve_datasets(parsed, openalex_dois)
    chunk_candidates = retrieve_chunks(parsed)
    build_links(dataset_candidates, chunk_candidates, openalex_papers)
    ranked_papers = rerank_papers(
        openalex_papers, paper_matches, chunk_candidates, parsed.local_query,
        exclude_paper_ids=effective_exclude_papers,
    )
    ranked_datasets = rerank_datasets(
        dataset_candidates,
        exclude_dataset_ids=effective_exclude_datasets,
    )
    answer, evidence_block_text = generate_answer(
        parsed, ranked_papers[:10], ranked_datasets[:10], chunk_candidates[:10]
    )

    # Phase 3: snapshot the full evidence state to disk
    cache_dir = write_evidence_cache(
        query=user_query,
        parsed=parsed,
        openalex_papers=openalex_papers,
        zenodo_records=zenodo_records,
        local_dataset_candidates=ranked_datasets,
        chunk_candidates=chunk_candidates,
        evidence_block_text=evidence_block_text,
        final_answer=answer,
    )

    return {
        "parsed": parsed,
        "ranked_papers": ranked_papers,
        "ranked_datasets": ranked_datasets,
        "chunk_candidates": chunk_candidates,
        "openalex_papers": openalex_papers,
        "zenodo_records": zenodo_records,
        "answer": answer,
        "evidence_block_text": evidence_block_text,
        "cache_dir": str(cache_dir),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        history_dicts = (
            [m.model_dump() for m in request.history] if request.history else None
        )
        result = _run_pipeline(
            request.query,
            history=history_dicts,
            exclude_paper_ids=request.exclude_paper_ids,
            exclude_dataset_ids=request.exclude_dataset_ids,
        )
        answer = result["answer"]
        return QueryResponse(
            query=request.query,
            answer=answer.final_text,
            answer_mode=answer.answer_mode,
            recommended_datasets=answer.recommended_datasets,
            recommended_papers=answer.recommended_papers,
            methodology_hints=answer.methodology_hints,
            uncertainty_notes=answer.uncertainty_notes,
            grounding_report=answer.grounding_report,
        )
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=f"OpenAI API rate limit hit: {e}",
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _format_pretty(request: QueryRequest) -> str:
    history_dicts = (
        [m.model_dump() for m in request.history] if request.history else None
    )
    result = _run_pipeline(
        request.query,
        history=history_dicts,
        exclude_paper_ids=request.exclude_paper_ids,
        exclude_dataset_ids=request.exclude_dataset_ids,
    )
    answer = result["answer"]

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
            cit = f" [{', '.join(d.citations)}]" if d.citations else ""
            lines.append(f"{i}. [{d.source.upper()}] {d.dataset_name}{cit}")
            lines.append(f"   Evidence: {d.evidence_strength} | Reason: {d.reason}")
            if d.doi:
                lines.append(f"   DOI: {d.doi}")

    if answer.recommended_papers:
        lines.append("")
        lines.append("─" * 60)
        lines.append("RECOMMENDED PAPERS")
        lines.append("─" * 60)
        for i, p in enumerate(answer.recommended_papers, 1):
            tag = "✓ fulltext" if p.evidence_level == "fulltext_supported" else "  metadata"
            cit = f" [{', '.join(p.citations)}]" if p.citations else ""
            lines.append(f"{i}. [{tag}] {p.title} ({p.year}){cit}")
            lines.append(f"   Reason: {p.reason}")

    if answer.methodology_hints:
        lines.append("")
        lines.append("─" * 60)
        lines.append("METHODOLOGY HINTS")
        lines.append("─" * 60)
        for i, h in enumerate(answer.methodology_hints, 1):
            lines.append(f"{i}. {h.hint} [{', '.join(h.citations)}]")

    if answer.uncertainty_notes:
        lines.append("")
        lines.append("─" * 60)
        lines.append("UNCERTAINTY NOTES")
        lines.append("─" * 60)
        for note in answer.uncertainty_notes:
            lines.append(f"• {note}")

    if answer.grounding_report:
        lines.append("")
        lines.append("─" * 60)
        lines.append("GROUNDING REPORT")
        lines.append("─" * 60)
        gr = answer.grounding_report
        lines.append(f"OK: {gr.grounded_ok} | Rate: {gr.grounding_rate} | Tags: {gr.tags_found}/{gr.tags_total}")
        for v in gr.violations:
            lines.append(f"  ⚠ {v}")

    lines.append("=" * 60)
    lines.append(f"Cache: {result['cache_dir']}")
    return "\n".join(lines)


@app.post("/query/pretty", response_class=PlainTextResponse)
def query_pretty(request: QueryRequest):
    try:
        return _format_pretty(request)
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=f"OpenAI API rate limit hit: {e}",
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
