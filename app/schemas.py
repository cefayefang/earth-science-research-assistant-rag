from pydantic import BaseModel, Field
from typing import Optional


# ── Query parsing ─────────────────────────────────────────────────────────────

class MustHaveConstraints(BaseModel):
    region: bool = False
    timescale: bool = False


class ParsedQuery(BaseModel):
    original_query: str
    intent: str
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

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    answer_mode: str
    recommended_datasets: list[RecommendedDataset] = Field(default_factory=list)
    recommended_papers: list[RecommendedPaper] = Field(default_factory=list)
    methodology_hints: list[MethodHint] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    grounding_report: Optional[GroundingReport] = None
