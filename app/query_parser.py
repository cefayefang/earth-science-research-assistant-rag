import json
from openai import OpenAI
from .config import get_settings, openai_api_key, ROOT
from .schemas import ParsedQuery, MustHaveConstraints

_PROMPT = """You are an Earth science research assistant. Parse the user's query into structured JSON.

User query: {query}

Return ONLY valid JSON with this exact structure:
{{
  "original_query": "<the user's query>",
  "intent": "<one of: definition_or_explanation | paper_specific_question | dataset_and_paper_recommendation | dataset_recommendation | paper_recommendation | methodology_support | other>",
  "answer_mode": "<one of: direct_answer | recommendation | hybrid>",
  "phenomenon": "<main Earth science phenomenon or null>",
  "variables": ["<list of relevant scientific variables>"],
  "region": "<geographic region or null>",
  "timescale": "<time period as free text or null>",
  "local_query": "<enriched retrieval string for local search, include variables and region>",
  "openalex_query": "<concise query string for OpenAlex paper search or null>",
  "zenodo_query": "<concise query string for Zenodo dataset search or null>",
  "must_have_constraints": {{
    "region": <true if region is essential, false otherwise>,
    "timescale": <true if timescale is essential, false otherwise>
  }},
  "region_bbox": <[min_lon, min_lat, max_lon, max_lat] or null>,
  "parsed_timescale": <["YYYY-MM-DD", "YYYY-MM-DD"] or null>
}}

Rules:
- answer_mode=direct_answer for definitions, explanations, and paper-specific questions
- answer_mode=recommendation for dataset/paper/research starting point requests
- answer_mode=hybrid for questions needing both explanation and recommendations
- local_query should be rich: combine phenomenon + variables + region + timescale
- openalex_query and zenodo_query should be concise keyword strings
- Return null for fields that are not present in the query

For region_bbox:
- If region is specified, provide a best-effort bounding box in decimal degrees.
- Use standard geographic knowledge. Examples:
    "Central Asia"      → [40, 30, 90, 55]
    "Tibetan Plateau"   → [73, 25, 104, 40]
    "Arctic"            → [-180, 66.5, 180, 90]
    "Northern California" → [-124, 37, -119, 42]
    "Sahel"             → [-20, 10, 40, 20]
    "global" or no region → null
- If the query spans crossing the antimeridian, use the western bbox only.
- If region is vague or null, return null.

For parsed_timescale:
- Convert timescale to ISO dates [start, end].
- For open-ended "to present" or "ongoing", use today's date: "2026-04-21".
- For phrases like "past two decades" → compute start = today - 20 years.
- For "satellite era" (Earth-observation context) → start at 1972-01-01 (Landsat-1).
- If no timescale is specified, return null.
"""


def parse_query(user_query: str) -> ParsedQuery:
    cfg = get_settings()
    client = OpenAI(api_key=openai_api_key())

    prompt = _PROMPT.format(query=user_query)
    response = client.chat.completions.create(
        model=cfg["llm"]["default_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_output_tokens"],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()

    data = json.loads(text)

    constraints = data.get("must_have_constraints") or {}
    parsed = ParsedQuery(
        original_query=data.get("original_query", user_query),
        intent=data.get("intent", "other"),
        answer_mode=data.get("answer_mode", "hybrid"),
        phenomenon=data.get("phenomenon"),
        variables=data.get("variables") or [],
        region=data.get("region"),
        timescale=data.get("timescale"),
        local_query=data.get("local_query", user_query),
        openalex_query=data.get("openalex_query"),
        zenodo_query=data.get("zenodo_query"),
        must_have_constraints=MustHaveConstraints(
            region=bool(constraints.get("region", False)),
            timescale=bool(constraints.get("timescale", False)),
        ),
        region_bbox=_coerce_bbox(data.get("region_bbox")),
        parsed_timescale=_coerce_timescale(data.get("parsed_timescale")),
    )

    # save debug output
    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_parsed_query.json", "w") as f:
        f.write(parsed.model_dump_json(indent=2))

    return parsed


def _coerce_bbox(val) -> list[float] | None:
    if val is None:
        return None
    if isinstance(val, list) and len(val) == 4:
        try:
            return [float(x) for x in val]
        except (ValueError, TypeError):
            return None
    return None


def _coerce_timescale(val) -> list[str] | None:
    if val is None:
        return None
    if isinstance(val, list) and len(val) == 2:
        return [str(v) for v in val]
    return None
