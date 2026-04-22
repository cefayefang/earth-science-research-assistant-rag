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


_PROMPT = """You are an Earth science research assistant operating in retrieval-grounded mode.

=== EVIDENCE BLOCK ===
--- DATASETS (cite as [DS-N]) ---
{datasets}

--- PAPERS (cite as [P-N]) ---
{papers}

--- EVIDENCE CHUNKS (cite as [C-N]) ---
{chunks}
=== END EVIDENCE BLOCK ===

User query: {query}
Answer mode: {answer_mode}

HARD rules (absolute — never violate):
  H1. Every [DS-N], [P-N], [C-N] tag in your output MUST correspond to an entry in the EVIDENCE BLOCK above.
  H2. recommended_datasets: ONLY items from the DATASETS section above.
  H3. recommended_papers:   ONLY items from the PAPERS section above.
  H4. methodology_hints: each hint MUST cite at least one [C-N] chunk from above.
      If no chunk above supports a methodology claim, return [] (empty list).
      Do NOT use general methodology knowledge.
  H5. Dataset capabilities / variables / coverage: only what the description above states.
  H6. Paper findings: only what the abstract or a cited chunk above states.

SOFT rules (definitional fallback only):
  S1. For basic definitional content in direct_answer or hybrid mode
      (e.g. "What is NDVI?"), prefer quoting from evidence chunks via [C-N].
      If no chunk defines the term, you MAY use standard textbook knowledge,
      but MUST add to uncertainty_notes:
      "Definition provided from general knowledge; no corpus chunk directly defines this term."
  S2. S1 does NOT extend to methodology, dataset claims, or paper findings.

Abstention (CRITICAL — read carefully):
  A1. For recommendation mode, per-section abstention:
       - If the DATASETS section above is empty (shows "(none)") or contains
         only entries whose topical relevance to the query is clearly weak,
         return recommended_datasets = [].
       - Same rule for PAPERS → recommended_papers = [].
       - Same for CHUNKS → methodology_hints = [].
  A1b. Add the following note to uncertainty_notes IF AND ONLY IF all three
       of recommended_datasets, recommended_papers, AND methodology_hints
       are returned empty (i.e. the entire evidence cache was off-topic):
         "No corpus evidence matched this query — this may be out of scope
          for our Earth science literature corpus."
       If you returned at least one non-empty list, do NOT add this note.
  A2. For direct_answer mode, if neither chunks nor general knowledge can answer,
      say so explicitly in direct_answer.
  A3. IMPORTANT: do NOT invent relevance. If the DATASETS and PAPERS sections
      describe topics clearly unrelated to the query (e.g., the query is about
      earthquakes but the block contains only vegetation/drought items),
      this is a signal that the corpus does not cover this query. Abstain.

Answer-mode behavior:

  • direct_answer: put a concise 2-4 sentence factual answer in `direct_answer`,
    with [C-N], [P-N], or [DS-N] citations as appropriate to the question.
    Use [DS-N] when the question is about a specific dataset's properties
    (e.g., spatial resolution, coverage, variables). Use [C-N] / [P-N] for
    concept definitions or paper-specific questions.

  • recommendation: put a 3-5 sentence FRAMING PARAGRAPH in `direct_answer`
    (not a one-liner — readers should get context before the list). The
    framing should:
      - briefly describe what the research question entails (what kind of
        phenomenon, what variables typically matter, what methodological
        considerations come up);
      - orient the reader to the KINDS of datasets / papers recommended below
        (e.g., "The datasets below span atmospheric reanalysis and vegetation
        indices; the papers provide regional case studies applying these
        products.");
      - where natural, weave in specific [DS-N] / [P-N] citations for the
        most salient ones.
    Then populate recommended_datasets (up to 5) and recommended_papers (up
    to 5), each with a one-line `reason` grounded in the evidence block and
    `citations` pointing to supporting [C-N] chunks.

  • hybrid: same framing paragraph as recommendation mode, but lead with
    1-2 sentences of definitional/explanatory content (cited) before the
    orientation sentences. Then the structured lists.

When writing the framing paragraph, avoid mechanical phrasing like "below
are the recommended datasets". Write as a researcher briefing an early-career
colleague — natural, informative, and scoped to the question.

Dataset deduplication:
  • If multiple DATASETS entries refer to the same underlying resource (e.g.,
    a canonical product like "ERA5" and a regional/temporal subset like
    "ERA5 monthly mean over Central Asia 2000-2020", or equivalent
    reformulations of the same dataset with different DOIs/providers),
    recommend ONLY ONE — prefer the canonical/authoritative version from an
    official archive (NASA, Copernicus, NOAA, ESA, etc.) over a user-uploaded
    subset on Zenodo, unless the subset is specifically more appropriate for
    the query's scope (e.g., query explicitly asks for the subset's region).
  • Use the dataset titles, DOIs, and sources in the EVIDENCE BLOCK to make
    this judgment. When in doubt, pick the entry with the broader spatial/
    temporal coverage.

Return ONLY JSON matching this schema (use "ref" = "DS-1" / "P-1" / "C-1" short tags):
{{
  "direct_answer": "string or null",
  "recommended_datasets": [
    {{"ref": "DS-N", "reason": "string (must be grounded in above description)", "citations": ["C-N", ...]}}
  ],
  "recommended_papers": [
    {{"ref": "P-N", "reason": "string", "citations": ["C-N", ...]}}
  ],
  "methodology_hints": [
    {{"hint": "string (must be paraphrase of chunk content)", "citations": ["C-N", ...]}}
  ],
  "uncertainty_notes": ["..."]
}}
"""


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

    prompt = _PROMPT.format(
        query=parsed_query.original_query,
        answer_mode=parsed_query.answer_mode,
        datasets=datasets_text,
        papers=papers_text,
        chunks=chunks_text,
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

    Design: the side panel already renders recommended datasets and papers as
    structured cards. We do NOT repeat those lists here. Instead the chatbot
    message presents:
      1. Framing / direct answer (prose)  — the 3-5 sentence intro the LLM
         writes in direct_answer for recommendation/hybrid modes
      2. Methodology hints                — the "how" that cuts across papers
      3. Uncertainty / caveats            — evidence-quality flags

    All internal [DS-N]/[P-N]/[C-N] tags are stripped from user-facing text.
    The structured FinalAnswer JSON (returned over the API) still carries the
    tags for evaluation and programmatic use.
    """
    parts = []

    direct = _strip_internal_tags((answer_json.get("direct_answer") or "").strip())
    if direct:
        parts.append(direct)

    if method_hints:
        parts.append("\n**How researchers typically approach this:**")
        for i, h in enumerate(method_hints, 1):
            clean_hint = _strip_internal_tags(h.hint)
            parts.append(f"  {i}. {clean_hint}")

    if uncertainty_notes:
        parts.append("\n**Caveats:**")
        for n in uncertainty_notes:
            parts.append(f"  • {_strip_internal_tags(n)}")

    # Cue the side panel. Only if we actually have something there to show.
    has_side_panel_content = bool(rec_datasets or rec_papers)
    if has_side_panel_content:
        cue_bits = []
        if rec_datasets:
            cue_bits.append(f"{len(rec_datasets)} recommended dataset(s)")
        if rec_papers:
            cue_bits.append(f"{len(rec_papers)} recommended paper(s)")
        parts.append(f"\n_See the side panel for {' and '.join(cue_bits)}._")

    return "\n".join(parts) if parts else "(no answer)"
