import json
from google import genai
from google.genai import types
from .config import get_settings, gemini_api_key, ROOT
from .schemas import (
    ParsedQuery, PaperCandidate, DatasetCandidate,
    ChunkCandidate, FinalAnswer, RecommendedDataset, RecommendedPaper,
)
from .dataset_normalizer import load_normalized_datasets

_PROMPT = """You are a grounded Earth science research assistant. Be concise — 3-5 sentences max.

User query: {query}
Answer mode: {answer_mode}

=== TOP DATASETS ===
{datasets}

=== TOP PAPERS ===
{papers}

=== EVIDENCE CHUNKS ===
{chunks}

Rules:
- Use ONLY information provided above. Never fabricate datasets, papers, or claims.
- direct_answer: 2-3 sentence factual answer only.
- recommendation: list datasets/papers with one-line reasons, no preamble.
- hybrid: 1-2 sentence answer, then bulleted recommendations.
- Flag uncertainty only if evidence is clearly weak.

Response:"""


def _format_datasets(datasets: list[DatasetCandidate], ds_lookup: dict) -> str:
    lines = []
    for i, d in enumerate(datasets[:5], 1):
        ds = ds_lookup.get(d.dataset_id)
        desc = ds.description[:200] if ds and ds.description else ""
        lines.append(f"{i}. [{d.source.upper()}] {d.title}\n   Score: {d.dataset_score} | Evidence: {d.literature_support}\n   {desc}")
    return "\n".join(lines) or "No datasets retrieved."


def _format_papers(papers: list[PaperCandidate]) -> str:
    lines = []
    for i, p in enumerate(papers[:5], 1):
        abstract_snippet = (p.abstract or "")[:200]
        lines.append(f"{i}. {p.title} ({p.year})\n   Score: {p.paper_score} | Evidence: {p.evidence_level}\n   {abstract_snippet}")
    return "\n".join(lines) or "No papers retrieved."


def _format_chunks(chunks: list[ChunkCandidate]) -> str:
    lines = []
    for i, c in enumerate(chunks[:5], 1):
        lines.append(f"{i}. [{c.section_guess or 'unknown section'}] (score: {c.chunk_score})\n   {c.text[:300]}")
    return "\n".join(lines) or "No evidence chunks retrieved."


def generate_answer(
    parsed_query: ParsedQuery,
    top_papers: list[PaperCandidate],
    top_datasets: list[DatasetCandidate],
    top_chunks: list[ChunkCandidate],
) -> FinalAnswer:
    cfg = get_settings()
    client = genai.Client(api_key=gemini_api_key())

    ds_lookup = {d.dataset_id: d for d in load_normalized_datasets()}

    prompt = _PROMPT.format(
        query=parsed_query.original_query,
        intent=parsed_query.intent,
        answer_mode=parsed_query.answer_mode,
        datasets=_format_datasets(top_datasets, ds_lookup),
        papers=_format_papers(top_papers),
        chunks=_format_chunks(top_chunks),
    )

    response = client.models.generate_content(
        model=cfg["llm"]["default_model"],
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=cfg["llm"]["temperature"],
            max_output_tokens=cfg["llm"]["max_output_tokens"],
        ),
    )
    final_text = response.text.strip()

    # build structured objects
    rec_datasets = []
    for d in top_datasets[:5]:
        ds = ds_lookup.get(d.dataset_id)
        strength = "high" if d.literature_support >= 0.8 else ("medium" if d.literature_support >= 0.5 else "low")
        rec_datasets.append(RecommendedDataset(
            dataset_id=d.dataset_id,
            dataset_name=d.title,
            source=d.source,
            reason=f"Relevance score {d.dataset_score:.2f}; evidence strength: {strength}.",
            evidence_strength=strength,
            doi=d.doi,
        ))

    rec_papers = []
    for p in top_papers[:5]:
        rec_papers.append(RecommendedPaper(
            openalex_id=p.openalex_id,
            local_id=p.local_id,
            title=p.title,
            year=p.year,
            reason=f"Relevance score {p.paper_score:.2f}; {p.evidence_level}.",
            evidence_level=p.evidence_level,
        ))

    uncertainty = []
    if any(p.evidence_level == "metadata_only" for p in top_papers[:3]):
        uncertainty.append("Some recommended papers are metadata-only and lack local full-text evidence.")
    if any(d.literature_support <= 0.2 for d in top_datasets[:3]):
        uncertainty.append("Some datasets are recommended based on semantic similarity only, without direct literature support.")

    answer = FinalAnswer(
        answer_mode=parsed_query.answer_mode,
        direct_answer=final_text if parsed_query.answer_mode == "direct_answer" else None,
        recommended_datasets=rec_datasets,
        recommended_papers=rec_papers,
        methodology_hints=[],
        uncertainty_notes=uncertainty,
        final_text=final_text,
    )

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_answer.json", "w") as f:
        f.write(answer.model_dump_json(indent=2))

    return answer
