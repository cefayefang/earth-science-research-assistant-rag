from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from openai import OpenAIError, RateLimitError
from .core.schemas import QueryRequest, QueryResponse
from .router import _run_pipeline

app = FastAPI(title="Earth Science Research Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        history_dicts = (
            [m.model_dump() for m in request.history] if request.history else None
        )
        session_dict = request.session_state.model_dump() if request.session_state else None
        result = _run_pipeline(
            request.query,
            history=history_dicts,
            exclude_paper_ids=request.exclude_paper_ids,
            exclude_dataset_ids=request.exclude_dataset_ids,
            session_state=session_dict,
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
            session_state=result.get("session_state"),
            intent_type=result.get("intent_type"),
        )
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"OpenAI API rate limit hit: {e}")
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
    lines.append(f"INTENT: {result.get('intent_type', '—')}")
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
    cache_dir = result.get("cache_dir", "")
    if cache_dir:
        lines.append(f"Cache: {cache_dir}")
    return "\n".join(lines)


@app.post("/query/pretty", response_class=PlainTextResponse)
def query_pretty(request: QueryRequest):
    try:
        return _format_pretty(request)
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"OpenAI API rate limit hit: {e}")
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
