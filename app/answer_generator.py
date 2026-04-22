"""
Grounded answer generator.

Design (matches EPS-210 rubric + Yu et al. 2025 Geo-RAG "verify" pillar):

  HARD grounding rules (absolute):
    H1. Every [DS-N], [P-N], [C-N] tag in output MUST exist in the evidence block.
    H2. recommended_datasets: only from DATASETS section.
    H3. recommended_papers:   only from PAPERS section.
    H4. methodology_hints:    each hint MUST cite at least one [C-N] chunk;
                              if no chunk supports → return [].
    H5. Dataset capabilities: only what the description states.
    H6. Paper findings:       only what abstract / chunk states.

  SOFT rules (definitional fallback only):
    S1. For basic definitional content in direct_answer/hybrid, if no chunk
        defines the term, LLM may use textbook knowledge but must add a flag
        to uncertainty_notes.
    S2. S1 does not extend to methodology / dataset / paper claims.

  Abstention:
    A1. recommendation mode with no relevant evidence → empty lists +
        uncertainty_notes += "No corpus evidence matched this query."
"""
import json
import re
from openai import OpenAI
from .config import get_settings, openai_api_key, ROOT
from .schemas import (
    ParsedQuery, PaperCandidate, DatasetCandidate, ChunkCandidate,
    FinalAnswer, RecommendedDataset, RecommendedPaper,
    MethodHint, GroundingReport,
)
from .dataset_normalizer import load_normalized_datasets


PROMPTS_DIR = ROOT / "prompts"

# Map ParsedQuery.intent → filename of the intent-specific prompt section.
_INTENT_TO_PROMPT_FILE = {
    "definition_or_explanation":  "intent_definition.md",
    "paper_specific_question":    "intent_paper_specific.md",
    "dataset_recommendation":     "intent_dataset_primary.md",
    "paper_recommendation":       "intent_paper_primary.md",
    "methodology_support":        "intent_methodology.md",
    "research_starter":           "intent_research_starter.md",
    "other":                      "intent_fallback.md",
}


def _load_prompt_file(name: str) -> str:
    """Read a prompt file from prompts/. Fallback to empty string if missing."""
    path = PROMPTS_DIR / name
    if not path.exists():
        return ""
    return path.read_text().strip()


def _assemble_prompt(
    intent: str,
    datasets_text: str,
    papers_text: str,
    chunks_text: str,
    query: str,
    answer_mode: str,
) -> str:
    """Compose final LLM prompt = base_rules + intent-specific + evidence + query."""
    base = _load_prompt_file("base_rules.md")
    intent_file = _INTENT_TO_PROMPT_FILE.get(intent, "intent_fallback.md")
    intent_specific = _load_prompt_file(intent_file)

    return (
        f"{base}\n\n"
        f"--- INTENT-SPECIFIC OUTPUT SHAPE ---\n{intent_specific}\n\n"
        f"--- EVIDENCE BLOCK ---\n"
        f"--- DATASETS (cite as [DS-N]) ---\n{datasets_text}\n\n"
        f"--- PAPERS (cite as [P-N]) ---\n{papers_text}\n\n"
        f"--- EVIDENCE CHUNKS (cite as [C-N]) ---\n{chunks_text}\n"
        f"--- END EVIDENCE BLOCK ---\n\n"
        f"User query: {query}\n"
        f"Answer mode: {answer_mode}\n"
    )


# ── Evidence block formatting ────────────────────────────────────────────────

MAX_DS = 10
MAX_P = 10
MAX_C = 10


def _format_datasets(datasets: list[DatasetCandidate], ds_lookup: dict) -> tuple[str, dict]:
    """Returns (formatted_text, id_map) where id_map['DS-1'] = dataset_id."""
    lines = []
    id_map = {}
    for i, d in enumerate(datasets[:MAX_DS], 1):
        ds = ds_lookup.get(d.dataset_id)
        desc = (ds.description[:300] if ds and ds.description else "").replace("\n", " ")
        spatial = ds.spatial_info if ds else None
        temporal = ds.temporal_info if ds else None
        tag = f"DS-{i}"
        id_map[tag] = d.dataset_id
        lines.append(
            f"[{tag} | id={d.dataset_id} | source={d.source}]\n"
            f"    {d.title}\n"
            f"    Spatial: {spatial or 'unspecified'} | Temporal: {temporal or 'unspecified'}\n"
            f"    Description: {desc}"
        )
    return ("\n\n".join(lines) or "(none)", id_map)


def _format_papers(papers: list[PaperCandidate]) -> tuple[str, dict]:
    lines = []
    id_map = {}
    for i, p in enumerate(papers[:MAX_P], 1):
        abstract = (p.abstract or "")[:400].replace("\n", " ")
        tag = f"P-{i}"
        id_map[tag] = {
            "openalex_id": p.openalex_id,
            "local_id": p.local_id,
        }
        lines.append(
            f"[{tag} | local_id={p.local_id or 'null'} | openalex_id={p.openalex_id or 'null'}]\n"
            f"    {p.title} ({p.year})\n"
            f"    Evidence level: {p.evidence_level}\n"
            f"    Abstract: {abstract}"
        )
    return ("\n\n".join(lines) or "(none)", id_map)


def _format_chunks(chunks: list[ChunkCandidate]) -> tuple[str, dict]:
    lines = []
    id_map = {}
    for i, c in enumerate(chunks[:MAX_C], 1):
        text = c.text[:600].replace("\n", " ")
        tag = f"C-{i}"
        id_map[tag] = c.chunk_id
        lines.append(
            f"[{tag} | chunk_id={c.chunk_id} | paper={c.local_id} | section={c.section_guess or 'unknown'}]\n"
            f"    {text}"
        )
    return ("\n\n".join(lines) or "(none)", id_map)


# ── Grounding validator ──────────────────────────────────────────────────────

_TAG_RE = re.compile(r"\[(DS|P|C)-(\d+)\]")


def _verify_grounding(
    answer_json: dict,
    ds_ids: set[str],
    p_ids: set[str],
    c_ids: set[str],
) -> GroundingReport:
    """Every tag the LLM emitted must resolve to something in the evidence block."""
    violations: list[str] = []
    tags_total = 0
    tags_found = 0

    def _check_tag(tag: str, where: str):
        nonlocal tags_total, tags_found
        tags_total += 1
        if tag in ds_ids or tag in p_ids or tag in c_ids:
            tags_found += 1
        else:
            violations.append(f"{where}: unknown tag [{tag}] not in evidence block")

    # recommended_datasets — ref must be a known DS tag
    for item in answer_json.get("recommended_datasets") or []:
        ref = item.get("ref")
        if ref:
            _check_tag(ref, "recommended_datasets.ref")
            if ref not in ds_ids:
                violations.append(f"recommended_datasets.ref={ref} is not a DATASET entry")
        for cit in item.get("citations") or []:
            _check_tag(cit, "recommended_datasets.citations")

    # recommended_papers — ref must be a known P tag
    for item in answer_json.get("recommended_papers") or []:
        ref = item.get("ref")
        if ref:
            _check_tag(ref, "recommended_papers.ref")
            if ref not in p_ids:
                violations.append(f"recommended_papers.ref={ref} is not a PAPER entry")
        for cit in item.get("citations") or []:
            _check_tag(cit, "recommended_papers.citations")

    # methodology_hints — every hint MUST cite at least one [C-N] chunk
    for item in answer_json.get("methodology_hints") or []:
        cits = item.get("citations") or []
        if not cits:
            violations.append("methodology_hint has no chunk citation (rule H4)")
        chunk_cits = [c for c in cits if c.startswith("C-")]
        if cits and not chunk_cits:
            violations.append(f"methodology_hint citations {cits} do not include any [C-N]")
        for cit in cits:
            _check_tag(cit, "methodology_hints.citations")

    # Scan direct_answer text for tags
    direct = answer_json.get("direct_answer") or ""
    for kind, n in _TAG_RE.findall(direct):
        _check_tag(f"{kind}-{n}", "direct_answer_text")

    grounding_rate = (tags_found / tags_total) if tags_total else 1.0
    return GroundingReport(
        grounded_ok=(len(violations) == 0),
        grounding_rate=round(grounding_rate, 4),
        violations=violations,
        tags_found=tags_found,
        tags_total=tags_total,
    )


# ── Main entry ───────────────────────────────────────────────────────────────

def _filter_by_relevance(
    top_papers: list[PaperCandidate],
    top_datasets: list[DatasetCandidate],
    top_chunks: list[ChunkCandidate],
) -> tuple[list, list, list, dict]:
    """Phase 9: drop items below similarity thresholds BEFORE exposing them
    to the LLM. This makes the evidence block genuinely sparse for OOS queries,
    so the abstention rule (A1) has a signal to act on.
    Returns (filtered_papers, filtered_datasets, filtered_chunks, drop_stats).
    """
    cfg = get_settings().get("retrieval", {})
    min_ds = cfg.get("min_dataset_similarity", 0.0)
    min_p = cfg.get("min_paper_similarity", 0.0)
    min_c = cfg.get("min_chunk_score", 0.0)

    kept_datasets = [d for d in top_datasets if d.metadata_similarity >= min_ds]
    # Papers use different relevance signals depending on provenance:
    #   local papers:    chunk_relevance (semantic_similarity is unused, always 0)
    #   external papers: semantic_similarity (chunk_relevance is 0)
    # Take max so either signal passing the threshold keeps the paper.
    kept_papers   = [
        p for p in top_papers
        if max(p.semantic_similarity, p.chunk_relevance) >= min_p
    ]
    kept_chunks   = [c for c in top_chunks   if c.chunk_score >= min_c]

    stats = {
        "datasets_dropped": len(top_datasets) - len(kept_datasets),
        "papers_dropped":   len(top_papers)   - len(kept_papers),
        "chunks_dropped":   len(top_chunks)   - len(kept_chunks),
        "datasets_kept":    len(kept_datasets),
        "papers_kept":      len(kept_papers),
        "chunks_kept":      len(kept_chunks),
    }
    return kept_papers, kept_datasets, kept_chunks, stats


def build_evidence_block(
    top_papers: list[PaperCandidate],
    top_datasets: list[DatasetCandidate],
    top_chunks: list[ChunkCandidate],
    ds_lookup: dict,
) -> tuple[str, dict, dict, dict, dict]:
    """Build the evidence block text + ID maps + filter stats.
    Phase 9: pre-filter by relevance thresholds so OOS queries get empty blocks."""
    top_papers, top_datasets, top_chunks, drop_stats = _filter_by_relevance(
        top_papers, top_datasets, top_chunks,
    )
    datasets_text, ds_map = _format_datasets(top_datasets, ds_lookup)
    papers_text,   p_map  = _format_papers(top_papers)
    chunks_text,   c_map  = _format_chunks(top_chunks)

    block = (
        "--- DATASETS (cite as [DS-N]) ---\n"
        f"{datasets_text}\n\n"
        "--- PAPERS (cite as [P-N]) ---\n"
        f"{papers_text}\n\n"
        "--- EVIDENCE CHUNKS (cite as [C-N]) ---\n"
        f"{chunks_text}"
    )
    return block, ds_map, p_map, c_map, drop_stats


def generate_answer(
    parsed_query: ParsedQuery,
    top_papers: list[PaperCandidate],
    top_datasets: list[DatasetCandidate],
    top_chunks: list[ChunkCandidate],
) -> tuple[FinalAnswer, str]:
    """Returns (final_answer, evidence_block_text)."""
    cfg = get_settings()
    client = OpenAI(api_key=openai_api_key())

    ds_lookup = {d.dataset_id: d for d in load_normalized_datasets()}

    evidence_block, ds_map, p_map, c_map, drop_stats = build_evidence_block(
        top_papers, top_datasets, top_chunks, ds_lookup,
    )
    # Re-derive the individual section texts from filtered lists (matching what's in the block)
    filtered_papers, filtered_datasets, filtered_chunks, _ = _filter_by_relevance(
        top_papers, top_datasets, top_chunks,
    )
    datasets_text, _ = _format_datasets(filtered_datasets, ds_lookup)
    papers_text,   _ = _format_papers(filtered_papers)
    chunks_text,   _ = _format_chunks(filtered_chunks)

    prompt = _assemble_prompt(
        intent=parsed_query.intent,
        datasets_text=datasets_text,
        papers_text=papers_text,
        chunks_text=chunks_text,
        query=parsed_query.original_query,
        answer_mode=parsed_query.answer_mode,
    )

    response = client.chat.completions.create(
        model=cfg["llm"]["default_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_output_tokens"],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()

    try:
        answer_json = json.loads(raw)
    except json.JSONDecodeError:
        # emergency fallback — wrap free text
        answer_json = {
            "direct_answer": raw,
            "recommended_datasets": [],
            "recommended_papers": [],
            "methodology_hints": [],
            "uncertainty_notes": ["LLM returned invalid JSON; falling back to free text."],
        }

    # Grounding validation
    report = _verify_grounding(
        answer_json,
        ds_ids=set(ds_map.keys()),
        p_ids=set(p_map.keys()),
        c_ids=set(c_map.keys()),
    )

    # ── Build structured recommendations ──────────────────────────────────────
    CURATED_SOURCES = {"nasa_cmr", "stac", "copernicus_cds", "cdse"}

    def _evidence_strength(cand: DatasetCandidate) -> str:
        """Phase 8: multi-factor evidence_strength.
        Combines literature_support (paper citation) + provenance (curated
        authority + DOI) into a single bucket for UI display.
        """
        lit = cand.literature_support
        # Strong paper backing
        if lit >= 0.8:
            return "high"
        # Moderate paper backing
        if lit >= 0.5:
            return "medium"
        # No strong paper backing — use provenance as fallback signal
        has_doi = bool(cand.doi)
        is_curated = cand.source in CURATED_SOURCES or cand.source == "zenodo"
        if has_doi and is_curated:
            return "medium"   # DOI-backed + curated authority
        if is_curated:
            return "medium"   # curated source alone still counts as provenance
        if has_doi:
            return "medium"   # any DOI-backed record
        return "low"

    rec_datasets: list[RecommendedDataset] = []
    for item in answer_json.get("recommended_datasets") or []:
        ref = item.get("ref", "")
        dataset_id = ds_map.get(ref)
        if not dataset_id:
            continue  # skip hallucinated refs
        cand = next((d for d in top_datasets if d.dataset_id == dataset_id), None)
        if not cand:
            continue
        rec_datasets.append(RecommendedDataset(
            dataset_id=dataset_id,
            dataset_name=cand.title,
            source=cand.source,
            reason=item.get("reason", ""),
            evidence_strength=_evidence_strength(cand),
            doi=cand.doi,
            citations=[str(c) for c in (item.get("citations") or [])],
        ))

    rec_papers: list[RecommendedPaper] = []
    for item in answer_json.get("recommended_papers") or []:
        ref = item.get("ref", "")
        info = p_map.get(ref)
        if not info:
            continue
        cand = next(
            (p for p in top_papers if p.openalex_id == info["openalex_id"]),
            None,
        )
        if not cand:
            continue
        rec_papers.append(RecommendedPaper(
            openalex_id=cand.openalex_id,
            local_id=cand.local_id,
            title=cand.title,
            year=cand.year,
            reason=item.get("reason", ""),
            evidence_level=cand.evidence_level,
            citations=[str(c) for c in (item.get("citations") or [])],
        ))

    # methodology_hints (H4: must have chunk citation)
    method_hints: list[MethodHint] = []
    for item in answer_json.get("methodology_hints") or []:
        cits = [str(c) for c in (item.get("citations") or [])]
        # Only keep hints with at least one valid C-N citation
        valid_cits = [c for c in cits if c.startswith("C-") and c in c_map]
        if not valid_cits:
            continue  # drop hints that violate H4
        method_hints.append(MethodHint(
            hint=item.get("hint", "").strip(),
            citations=valid_cits,
        ))

    uncertainty_notes = list(answer_json.get("uncertainty_notes") or [])

    # Phase 9: hard-enforced abstention when evidence block is sparse.
    # If all three evidence sections were filtered empty by the relevance threshold,
    # force empty recommendations regardless of what the LLM said.
    evidence_starved = (
        len(filtered_datasets) == 0
        and len(filtered_papers) == 0
        and len(filtered_chunks) == 0
    )
    if evidence_starved:
        rec_datasets = []
        rec_papers = []
        method_hints = []
        answer_json["direct_answer"] = (
            "Insufficient corpus evidence to answer this query. "
            "This topic appears out of scope for our Earth science literature corpus."
        )
        uncertainty_notes.append(
            "No corpus evidence matched this query — this query appears out of scope for our Earth science literature corpus. System abstained."
        )
    # Note: we used to append "Some recommended papers are metadata-only..." here,
    # but that note triggered on nearly every recommendation query (external
    # OpenAlex papers are always metadata-only by construction), so it provided
    # no new information — the paper cards in the side panel already show
    # "full-text" vs "metadata" per paper. Removed to reduce noise.

    # Phase 8: only warn when BOTH literature and provenance are weak
    def _truly_weak(d: DatasetCandidate) -> bool:
        if d.literature_support >= 0.5:
            return False
        has_doi = bool(d.doi)
        is_curated = d.source in CURATED_SOURCES or d.source == "zenodo"
        return not (has_doi or is_curated)

    if top_datasets and any(_truly_weak(d) for d in top_datasets[:3]):
        uncertainty_notes.append(
            "Some datasets are recommended based on semantic similarity only, without direct literature support or provenance signals."
        )
    if not report.grounded_ok:
        uncertainty_notes.append(
            f"Grounding violations detected: {len(report.violations)} issue(s) in LLM output."
        )

    # Build human-facing plain-text version
    final_text = _render_plain_text(
        answer_json=answer_json,
        ds_map=ds_map,
        p_map=p_map,
        rec_datasets=rec_datasets,
        rec_papers=rec_papers,
        method_hints=method_hints,
        uncertainty_notes=uncertainty_notes,
    )

    answer = FinalAnswer(
        answer_mode=parsed_query.answer_mode,
        direct_answer=(answer_json.get("direct_answer") or "").strip() or None,
        recommended_datasets=rec_datasets,
        recommended_papers=rec_papers,
        methodology_hints=method_hints,
        uncertainty_notes=uncertainty_notes,
        final_text=final_text,
        grounding_report=report,
    )

    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_answer.json", "w") as f:
        f.write(answer.model_dump_json(indent=2))
    with open(debug_dir / "last_llm_raw.json", "w") as f:
        f.write(raw)
    with open(debug_dir / "last_prompt.txt", "w") as f:
        f.write(prompt)

    return answer, evidence_block


# Matches [C-1], [DS-3], [P-4] as well as combined forms like [C-1, C-2, P-3]
_USER_FACING_TAG_RE = re.compile(
    r"\s*\[(?:DS|P|C)-\d+(?:\s*,\s*(?:DS|P|C)-\d+)*\]"
)


def _strip_internal_tags(text: str) -> str:
    """Remove internal evidence-block tags ([DS-N]/[P-N]/[C-N]) from
    user-visible text. Structured IDs remain in the JSON API response for
    programmatic consumers; the chatbot/UI just shouldn't expose them."""
    if not text:
        return text
    cleaned = _USER_FACING_TAG_RE.sub("", text)
    # collapse double spaces left behind by tag removal
    return re.sub(r"  +", " ", cleaned).strip()


_SOURCE_LABEL = {
    "nasa_cmr":       "NASA CMR",
    "stac":           "STAC",
    "copernicus_cds": "Copernicus CDS",
    "cdse":           "CDSE",
    "zenodo":         "Zenodo",
}


def _render_plain_text(
    answer_json: dict,
    ds_map: dict,
    p_map: dict,
    rec_datasets: list,
    rec_papers: list,
    method_hints: list,
    uncertainty_notes: list,
) -> str:
    """User-facing chatbot text.

    Design: the chatbot keeps a full conversation history, so each turn's
    message must stand on its own — the user should still be able to see
    what was recommended in turn N even after turn N+1 overwrites the side
    panel. We therefore include the recommendations in the chatbot text,
    but in a clean user-facing format:

      1. Framing / direct answer (prose)
      2. Recommended datasets (numbered list: name + source, no internal IDs)
      3. Recommended papers (numbered list: title + year + evidence level)
      4. Methodology hints — the "how" that cuts across papers
      5. Uncertainty / caveats (if any)

    All internal [DS-N]/[P-N]/[C-N] tags are stripped from user-facing text.
    The structured FinalAnswer JSON (returned over the API) still carries the
    tags for evaluation and programmatic use. The side panel shows the same
    items as richer cards (with DOI, evidence strength icons, etc.).
    """
    parts = []

    direct = _strip_internal_tags((answer_json.get("direct_answer") or "").strip())
    if direct:
        parts.append(direct)

    if rec_datasets:
        parts.append("\n**Recommended datasets:**")
        for i, d in enumerate(rec_datasets, 1):
            source = _SOURCE_LABEL.get(d.source, d.source.upper() if d.source else "")
            reason = _strip_internal_tags(d.reason) if d.reason else ""
            line = f"  {i}. **{d.dataset_name}**" + (f" _({source})_" if source else "")
            if reason:
                line += f" — {reason}"
            parts.append(line)

    if rec_papers:
        parts.append("\n**Recommended papers:**")
        for i, p in enumerate(rec_papers, 1):
            year_str = f"({p.year})" if p.year else ""
            tag = "✓ full-text" if p.evidence_level == "fulltext_supported" else "metadata"
            reason = _strip_internal_tags(p.reason) if p.reason else ""
            line = f"  {i}. **{p.title}** {year_str} _[{tag}]_".strip()
            if reason:
                line += f" — {reason}"
            parts.append(line)

    if method_hints:
        parts.append("\n**How researchers typically approach this:**")
        for i, h in enumerate(method_hints, 1):
            clean_hint = _strip_internal_tags(h.hint)
            parts.append(f"  {i}. {clean_hint}")

    if uncertainty_notes:
        parts.append("\n**Caveats:**")
        for n in uncertainty_notes:
            parts.append(f"  • {_strip_internal_tags(n)}")

    return "\n".join(parts) if parts else "(no answer)"
