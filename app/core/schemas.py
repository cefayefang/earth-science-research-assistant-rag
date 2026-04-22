from pydantic import BaseModel, Field
from typing import Optional, Any


# ── Intent classification ─────────────────────────────────────────────────────

class IntentClassification(BaseModel):
    intent_type: str  # chitchat | new_question | re_recommend | detail_followup | out_of_scope
    confidence: float = 0.8
    target_ref: Optional[str] = None      # for detail_followup: which item
    # For detail_followup: WHICH KIND of item the user is asking about.
    # "paper" / "dataset" / null (ambiguous — e.g. "the first one", "tell me more").
    # Ambiguous cases fall back to a paper-first lookup in _handle_detail_followup.
    target_kind: Optional[str] = None
    rewritten_query: Optional[str] = None  # for re_recommend: expanded retrieval query
    # User-specified count like "推两个 dataset" / "give me 3 papers". Null when unspecified.
    # The target names which output list the count applies to. When the user names a
    # kind ("two datasets"), target is set; when the user says a bare number with no
    # kind, target is null and the downstream layer falls back to the intent's primary
    # output.
    requested_count: Optional[int] = None
    requested_count_target: Optional[str] = None  # "datasets" | "papers" | "methodology"


class CachedChunk(BaseModel):
    """Serializable snapshot of a chunk for session state storage."""
    chunk_id: str
    local_id: str
    text: str
    section_guess: Optional[str] = None


class SessionPaper(BaseModel):
    """Minimal paper info stored per-turn for detail_followup resolution."""
    position: int       # 1-indexed rank in the recommendation list
    title: str
    local_id: Optional[str] = None
    openalex_id: Optional[str] = None


class SessionDataset(BaseModel):
    position: int
    title: str
    dataset_id: str


class SessionDatasetMetadata(BaseModel):
    """Serializable slice of NormalizedDataset for datasets whose source is
    EPHEMERAL (currently Zenodo — fetched fresh each query, not written to
    normalized_datasets.jsonl). Persisting this per-turn lets
    `_answer_from_dataset` answer detail follow-ups about Zenodo records
    without having to re-hit the Zenodo API. `raw_metadata` is intentionally
    excluded to keep session state small.
    """
    dataset_id: str
    display_name: str
    source: str
    provider: Optional[str] = None
    doi: Optional[str] = None
    description: Optional[str] = None
    variables: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    spatial_info: Optional[str] = None
    temporal_info: Optional[str] = None


class SessionState(BaseModel):
    """Client-side session state. Sent with every request; returned updated."""
    # Accumulated IDs across turns — used for re_recommend exclusion
    recommended_paper_ids: list[str] = Field(default_factory=list)
    recommended_dataset_ids: list[str] = Field(default_factory=list)
    # Positional info from the last RAG turn — for detail_followup resolution
    last_recommended_papers: list[SessionPaper] = Field(default_factory=list)
    last_recommended_datasets: list[SessionDataset] = Field(default_factory=list)
    # Chunks retrieved in the last RAG turn
    last_turn_chunks: list[CachedChunk] = Field(default_factory=list)
    # Metadata for last-turn-recommended datasets whose source is ephemeral
    # (currently Zenodo). Baked sources (nasa_cmr / stac / copernicus_cds /
    # cdse) do NOT appear here — they're re-looked-up from
    # load_normalized_datasets().
    last_turn_ephemeral_dataset_metadata: list[SessionDatasetMetadata] = Field(default_factory=list)
    turn_count: int = 0


# ── Query parsing ─────────────────────────────────────────────────────────────

class MustHaveConstraints(BaseModel):
    region: bool = False
    timescale: bool = False


class ParsedQuery(BaseModel):
    original_query: str
    intent: str   # one of: definition_or_explanation | paper_specific_question |
                  #        dataset_recommendation | paper_recommendation |
                  #        methodology_support | research_starter | other
    answer_mode: str  # direct_answer | recommendation | hybrid
    phenomenon: Optional[str] = None
    variables: list[str] = Field(default_factory=list)
    region: Optional[str] = None
    timescale: Optional[str] = None
    local_query: str
    openalex_query: Optional[str] = None
    zenodo_query: Optional[str] = None
    must_have_constraints: MustHaveConstraints = Field(default_factory=MustHaveConstraints)
    # Phase 4: structured spatial/temporal matching
    region_bbox: Optional[list[float]] = None        # [min_lon, min_lat, max_lon, max_lat]
    parsed_timescale: Optional[list[str]] = None     # [ISO_start, ISO_end]
    # True only for EXPANSION follow-ups ("more papers", "different ones"). Set
    # authoritatively by the caller based on intent_classifier's `re_recommend`
    # verdict. When True, the reranker will filter out exclude_paper_ids /
    # exclude_dataset_ids so new items surface.
    wants_fresh_recommendations: bool = False
    # User-specified count for the primary output list, threaded down from
    # IntentClassification. See IntentClassification for field semantics.
    requested_count: Optional[int] = None
    requested_count_target: Optional[str] = None  # "datasets" | "papers" | "methodology"


# ── Paper registry ────────────────────────────────────────────────────────────

class PaperRecord(BaseModel):
    local_id: str
    openalex_id: Optional[str] = None
    original_title: str
    filename: str
    pdf_path: str
    # OpenAlex-enriched fields populated at preprocessing time. These ensure
    # local papers have year/abstract/doi/cited_by_count even when the current
    # query's OpenAlex search does not happen to return them.
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    cited_by_count: int = 0


# ── Dataset ───────────────────────────────────────────────────────────────────

class NormalizedDataset(BaseModel):
    dataset_id: str
    source: str  # nasa_cmr | stac | copernicus_cds | cdse | zenodo
    source_raw_id: str
    source_title: str
    display_name: str
    description: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    provider: Optional[str] = None
    spatial_info: Optional[str] = None
    temporal_info: Optional[str] = None
    doi: Optional[str] = None
    retrieval_text: str = ""
    raw_metadata: dict = Field(default_factory=dict)


class DatasetCandidate(BaseModel):
    dataset_id: str
    source: str
    title: str
    doi: Optional[str] = None
    metadata_similarity: float = 0.0
    variable_match: float = 0.0
    spatial_match: float = 0.5
    temporal_match: float = 0.5
    # Default matches the `semantic_only` baseline in settings.yaml. The
    # retrieve_datasets pipeline overwrites this with has_doi (0.7) for
    # DOI-bearing candidates, and linker.build_links may upgrade further
    # to abstract_mention (0.85) or chunk_explicit_mention (1.0).
    literature_support: float = 0.5
    dataset_score: float = 0.0


# ── PDF chunks ────────────────────────────────────────────────────────────────

class PageText(BaseModel):
    page_num: int
    raw_text: str
    cleaned_text: str


class ParsedPaper(BaseModel):
    local_id: str
    openalex_id: Optional[str] = None
    filename: str
    pages: list[PageText]
    full_cleaned_text: str


class Chunk(BaseModel):
    chunk_id: str
    local_id: str
    openalex_id: Optional[str] = None
    filename: str
    page_range: list[int]
    section_guess: Optional[str] = None
    text: str


class ChunkCandidate(BaseModel):
    chunk_id: str
    local_id: str
    openalex_id: Optional[str] = None
    section_guess: Optional[str] = None
    chunk_score: float
    text: str


# ── OpenAlex ──────────────────────────────────────────────────────────────────

class OpenAlexPaper(BaseModel):
    openalex_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    cited_by_count: int = 0
    bucket: str  # recent | impactful
    oa_url: Optional[str] = None


class PaperMatch(BaseModel):
    openalex_id: str
    local_id: Optional[str] = None
    evidence_level: str  # fulltext_supported | metadata_only


class PaperCandidate(BaseModel):
    openalex_id: str
    local_id: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    cited_by_count: int = 0
    evidence_level: str = "metadata_only"
    semantic_similarity: float = 0.0
    chunk_relevance: float = 0.0
    recency_score: float = 0.0
    impact_score: float = 0.0
    fulltext_bonus: float = 0.0
    paper_score: float = 0.0


# ── Linking ───────────────────────────────────────────────────────────────────

class DatasetLink(BaseModel):
    dataset_id: str
    local_id: Optional[str] = None
    openalex_id: Optional[str] = None
    evidence_source: str  # chunk | abstract | zenodo_doi | semantic
    confidence: str       # high | medium | low
    evidence_text: Optional[str] = None


# ── Final answer ──────────────────────────────────────────────────────────────

class RecommendedDataset(BaseModel):
    dataset_id: str
    dataset_name: str
    source: str
    reason: str
    evidence_strength: str  # high | medium | low
    doi: Optional[str] = None
    citations: list[str] = Field(default_factory=list)   # e.g. ["C-3", "C-7"]


class RecommendedPaper(BaseModel):
    openalex_id: Optional[str] = None
    local_id: Optional[str] = None
    title: str
    year: Optional[int] = None
    reason: str
    evidence_level: str  # fulltext_supported | metadata_only
    citations: list[str] = Field(default_factory=list)


class MethodHint(BaseModel):
    hint: str
    citations: list[str] = Field(default_factory=list)   # chunk tags like "C-3"


class GroundingReport(BaseModel):
    grounded_ok: bool
    grounding_rate: float
    violations: list[str] = Field(default_factory=list)
    tags_found: int = 0
    tags_total: int = 0


class FinalAnswer(BaseModel):
    answer_mode: str
    direct_answer: Optional[str] = None
    recommended_datasets: list[RecommendedDataset] = Field(default_factory=list)
    recommended_papers: list[RecommendedPaper] = Field(default_factory=list)
    methodology_hints: list[MethodHint] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    final_text: str
    grounding_report: Optional[GroundingReport] = None


# ── API request/response ──────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    history: list[ConversationMessage] | None = None
    # Kept for backwards compat; new clients should rely on session_state instead.
    exclude_paper_ids: list[str] | None = None
    exclude_dataset_ids: list[str] | None = None
    # Full session state from the previous turn (None on first turn).
    session_state: Optional[SessionState] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    answer_mode: str
    recommended_datasets: list[RecommendedDataset] = Field(default_factory=list)
    recommended_papers: list[RecommendedPaper] = Field(default_factory=list)
    methodology_hints: list[MethodHint] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    grounding_report: Optional[GroundingReport] = None
    # Updated session state to store client-side and send back next turn.
    session_state: Optional[SessionState] = None
    intent_type: Optional[str] = None
