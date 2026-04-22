"""
Single-stage query parser.

One LLM call reads the conversation history (if any) together with the
current user query and returns a fully structured ParsedQuery.

This replaces the earlier two-stage design (a separate
`_analyze_conversation` call producing a ConversationDigest, followed by a
`parse_query` LLM call consuming that digest). The two stages always ran on
the same inputs and their outputs always fed each other, so keeping them
split just doubled the pre-RAG latency. The merged prompt instructs the LLM
to reason about multi-turn continuity internally before emitting the
structured fields, without needing to externalize an intermediate summary.

Routing decisions (is this an expansion follow-up? did the user ask for a
specific count?) are still NOT made here — intent_classifier is the single
source of truth for those, and main.py threads them in via kwargs
(`wants_fresh`, `requested_count`, `requested_count_target`). This module
stamps them onto the returned ParsedQuery unchanged.

On LLM/JSON failure we fall back gracefully: the returned ParsedQuery
carries `original_query`/`local_query` = user_query and sensible defaults,
so the rest of the pipeline can continue without crashing.
"""
import json
from typing import Optional
from ..core.config import get_settings, get_openai_client, ROOT
from ..core.schemas import ParsedQuery, MustHaveConstraints


# ── History prep ────────────────────────────────────────────────────────────

_MAX_VERBATIM_TURNS = 3
_MAX_ASSISTANT_CHARS = 400


def _pair_into_turns(history: list[dict] | None) -> list[dict]:
    """Fold a flat list of {role, content} messages into user/assistant pairs."""
    if not history:
        return []
    turns: list[dict] = []
    i = 0
    while i < len(history):
        msg = history[i]
        role = (msg.get("role") if isinstance(msg, dict) else msg.role) or ""
        content = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
        if role != "user":
            i += 1
            continue
        turn = {"user": content, "assistant": ""}
        if i + 1 < len(history):
            nxt = history[i + 1]
            nxt_role = (nxt.get("role") if isinstance(nxt, dict) else nxt.role) or ""
            nxt_content = (nxt.get("content") if isinstance(nxt, dict) else nxt.content) or ""
            if nxt_role == "assistant":
                turn["assistant"] = nxt_content
                i += 2
            else:
                i += 1
        else:
            i += 1
        turns.append(turn)
    return turns


def _format_history_block(turns: list[dict]) -> str:
    """Keep last N verbatim turns; truncate long assistant replies."""
    recent = turns[-_MAX_VERBATIM_TURNS:]
    parts = []
    for t in recent:
        a = (t.get("assistant") or "")
        if len(a) > _MAX_ASSISTANT_CHARS:
            a = a[:_MAX_ASSISTANT_CHARS].rstrip() + "…"
        parts.append(f"User: {t['user']}\nAssistant: {a}")
    return "\n\n".join(parts)


# ── Merged prompt ───────────────────────────────────────────────────────────

_MERGED_PROMPT = """You are an Earth science research assistant. Parse the user's query into a structured JSON retrieval plan.

{history_section}User query: {query}

{multi_turn_guidance}Return ONLY valid JSON with this EXACT structure:
{{
  "original_query": "<user's query>",
  "intent": "<one of: definition_or_explanation | paper_specific_question | dataset_recommendation | paper_recommendation | methodology_support | research_starter | other>",
  "answer_mode": "<direct_answer | recommendation | hybrid>",
  "phenomenon": "<main phenomenon or null>",
  "variables": ["<scientific variables>"],
  "region": "<geographic region or null>",
  "timescale": "<time period or null>",
  "local_query": "<enriched retrieval string; combine phenomenon + variables + region + timescale>",
  "openalex_query": "<concise keyword string or null>",
  "zenodo_query": "<concise keyword string or null>",
  "must_have_constraints": {{"region": <bool>, "timescale": <bool>}},
  "region_bbox": <[min_lon, min_lat, max_lon, max_lat] or null>,
  "parsed_timescale": <["YYYY-MM-DD", "YYYY-MM-DD"] or null>
}}

Answer-mode mapping:
- direct_answer   ← definition_or_explanation, paper_specific_question
- recommendation  ← dataset_recommendation, paper_recommendation, methodology_support, research_starter
- hybrid          ← questions that mix explanation + recommendation

Rules:
- Return null for fields the query (+ any prior-turn context) does not specify.
- Always produce a non-null local_query — for short follow-ups, inherit
  phenomenon, variables, region, timescale from prior turns so downstream
  retrieval has something to search on.

For region_bbox, use standard Earth-science bounding boxes, e.g.
  Central Asia [40, 30, 90, 55]; Arctic [-180, 66.5, 180, 90]
Return null for vague/global/unspecified regions.

For parsed_timescale, convert to ISO dates. "to present" → today's date.
"past N years" → today - N years. "satellite era" → 1972-01-01 onwards.
Return null if no timescale given.
"""


_MULTI_TURN_GUIDANCE = (
    "Because prior turns are shown, first think (silently) about how the "
    "CURRENT query relates to them — continuation, focus shift, or new "
    "topic. Then let that understanding flow into the fields below:\n"
    "- `intent`: pick the intent for the CURRENT query. If the user has "
    "  shifted focus (e.g. prior turns recommended datasets, current query "
    "  asks 'what methodology'), classify under the shifted intent.\n"
    "- `local_query` / `openalex_query` / `zenodo_query`: write the COMPLETE "
    "  retrieval string even for short follow-ups. Inherit the established "
    "  phenomenon, variables, region, and timescale from prior turns; the "
    "  retrieval layer has no other way to recover them.\n"
    "- `phenomenon` / `variables` / `region` / `timescale`: same — inherit "
    "  from prior turns unless the user has clearly changed topic.\n\n"
)


def _build_prompt(user_query: str, turns: list[dict]) -> str:
    if turns:
        history_text = _format_history_block(turns)
        history_section = (
            "Previous turns (most recent last):\n"
            f"{history_text}\n\n"
        )
        guidance = _MULTI_TURN_GUIDANCE
    else:
        history_section = ""
        guidance = ""
    return _MERGED_PROMPT.format(
        history_section=history_section,
        query=user_query,
        multi_turn_guidance=guidance,
    )


# ── Main entry ──────────────────────────────────────────────────────────────

def parse_query(
    user_query: str,
    history: list[dict] | None = None,
    wants_fresh: bool = False,
    requested_count: int | None = None,
    requested_count_target: str | None = None,
) -> ParsedQuery:
    """Parse the user's query into a ParsedQuery using a single LLM call.

    The LLM sees (history, current query) together and returns every field
    needed by the downstream retrieval + answer pipeline. Multi-turn reasoning
    happens inside that one call.

    The caller — typically main._run_pipeline — is responsible for deciding
    whether this turn is an expansion follow-up (`wants_fresh`) and whether
    the user specified a count (`requested_count` / `requested_count_target`).
    Those come from intent_classifier on the ORIGINAL user message (which may
    have been rewritten before being passed here) and are stamped directly
    onto the returned ParsedQuery.
    """
    cfg = get_settings()
    turns = _pair_into_turns(history)
    prompt = _build_prompt(user_query, turns)

    data: dict = {}
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=cfg["llm"]["default_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_output_tokens"],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
    except Exception as e:
        print(f"  [query_parser warn] merged parse failed: {e}")
        # Fall through with data={}, sensible defaults below.

    constraints = data.get("must_have_constraints") or {}
    # `or <fallback>` (instead of .get(key, fallback)) so explicit JSON null
    # also falls back. Important for short follow-ups where the LLM may leave
    # local_query null, and for the error-fallback path where data={}.
    parsed = ParsedQuery(
        original_query=(data.get("original_query") or user_query),
        intent=(data.get("intent") or "other"),
        answer_mode=(data.get("answer_mode") or "hybrid"),
        phenomenon=data.get("phenomenon"),
        variables=data.get("variables") or [],
        region=data.get("region"),
        timescale=data.get("timescale"),
        local_query=(data.get("local_query") or user_query),
        openalex_query=data.get("openalex_query"),
        zenodo_query=data.get("zenodo_query"),
        must_have_constraints=MustHaveConstraints(
            region=bool(constraints.get("region", False)),
            timescale=bool(constraints.get("timescale", False)),
        ),
        region_bbox=_coerce_bbox(data.get("region_bbox")),
        parsed_timescale=_coerce_timescale(data.get("parsed_timescale")),
        # Caller-stamped fields — intent_classifier is the source of truth.
        wants_fresh_recommendations=bool(wants_fresh),
        requested_count=requested_count,
        requested_count_target=requested_count_target,
    )

    # Debug dump (single file — no more separate digest file)
    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_parsed_query.json", "w") as f:
        f.write(parsed.model_dump_json(indent=2))

    return parsed


def _coerce_bbox(val) -> Optional[list[float]]:
    if val is None:
        return None
    if isinstance(val, list) and len(val) == 4:
        try:
            return [float(x) for x in val]
        except (ValueError, TypeError):
            return None
    return None


def _coerce_timescale(val) -> Optional[list[str]]:
    if val is None:
        return None
    if isinstance(val, list) and len(val) == 2:
        return [str(v) for v in val]
    return None
