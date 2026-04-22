"""
Two-stage query parsing:

  Stage 1 (optional): conversation_analyzer
    Runs ONLY when `history` is non-empty. Reads prior turns + current query
    and produces a ConversationDigest with:
      - a 2–3 sentence topical summary
      - wants_fresh_recommendations flag (true only for EXPANSION follow-ups)
      - optional intent_shift_hint (e.g. "methodology_support" when the
        follow-up shifts focus)

  Stage 2: parse_query (always runs)
    Takes the current query PLUS the Stage-1 summary (if any) and produces
    a structured ParsedQuery. The prompt is kept short because Stage 2 no
    longer has to reason about multi-turn logic — that's Stage 1's job.

If Stage 1 fails (bad JSON, API hiccup), we fall back gracefully: no
summary, wants_fresh=False, no intent_shift_hint. The Stage 2 parse then
runs as if it were a standalone query.
"""
import json
from typing import Optional
from openai import OpenAI
from .config import get_settings, openai_api_key, ROOT
from .schemas import ParsedQuery, MustHaveConstraints, ConversationDigest


# ── History prep (same as before but much simpler now) ──────────────────────

_MAX_VERBATIM_TURNS = 2
_MAX_ASSISTANT_CHARS = 400


def _pair_into_turns(history: list[dict] | None) -> list[dict]:
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


def _format_history_for_analyzer(turns: list[dict]) -> str:
    """Keep last N verbatim; truncate long assistant replies."""
    recent = turns[-_MAX_VERBATIM_TURNS:]
    parts = []
    for t in recent:
        a = (t.get("assistant") or "")
        if len(a) > _MAX_ASSISTANT_CHARS:
            a = a[:_MAX_ASSISTANT_CHARS].rstrip() + "…"
        parts.append(f"User: {t['user']}\nAssistant: {a}")
    return "\n\n".join(parts)


# ── Stage 1: conversation analyzer ───────────────────────────────────────────

_ANALYZER_PROMPT = """You are analyzing a multi-turn Earth science research conversation.

Previous turns (most recent last):
{history}

Current user query: {query}

Produce JSON with this EXACT shape:
{{
  "summary": "<2–3 sentences: what topic was discussed, what was recommended (if anything), how the current query relates>",
  "wants_fresh_recommendations": <true|false>,
  "intent_shift_hint": "<one of: definition_or_explanation | paper_specific_question | dataset_recommendation | paper_recommendation | methodology_support | research_starter | other | null>"
}}

Rules for wants_fresh_recommendations:
- TRUE only for EXPANSION follow-ups asking for items BEYOND what was already shown:
    "more papers", "give me different ones", "other datasets", "any others",
    "再给几个", "换一批", "what else is there"
- FALSE for every other case:
    - Focus shifts ("what methodology", "what about datasets", "how do they compare")
    - Drill-downs ("tell me about the first one", "what does paper A say")
    - Completely unrelated new topics
    - Clarifications ("what does that mean?")

Rules for intent_shift_hint:
- Set to one of the intent names if the CURRENT query clearly shifts focus to a
  different kind of output than prior turns. Examples:
    prior: datasets for X → current: "what methodology is used" → "methodology_support"
    prior: methodology for X → current: "recommend some papers" → "paper_recommendation"
- Set to null when the current query is the same intent type as prior turns
  (e.g., "more papers" following a paper recommendation).
- Set to null when there is no prior context to compare to.
"""


def _analyze_conversation(
    history: list[dict],
    current_query: str,
    cfg: dict,
) -> ConversationDigest:
    """Stage 1: small LLM call that distills history + current query."""
    turns = _pair_into_turns(history)
    if not turns:
        return ConversationDigest(summary="", wants_fresh_recommendations=False)

    history_text = _format_history_for_analyzer(turns)
    prompt = _ANALYZER_PROMPT.format(history=history_text, query=current_query)

    try:
        client = OpenAI(api_key=openai_api_key())
        response = client.chat.completions.create(
            model=cfg["llm"]["default_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content.strip())
        hint = data.get("intent_shift_hint")
        if hint == "null" or hint == "":
            hint = None
        return ConversationDigest(
            summary=(data.get("summary") or "").strip(),
            wants_fresh_recommendations=bool(data.get("wants_fresh_recommendations", False)),
            intent_shift_hint=hint,
        )
    except Exception as e:
        print(f"  [query_parser warn] conversation analysis failed: {e}")
        # graceful degradation: no summary, no routing hints
        return ConversationDigest(summary="", wants_fresh_recommendations=False)


# ── Stage 2: single-turn parse (shorter, focused) ────────────────────────────

_PARSE_PROMPT = """You are an Earth science research assistant. Parse the user's query into structured JSON.
{context_section}
User query: {query}

Return ONLY valid JSON with this exact structure:
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

Intent examples (for reference):
- "What is NDVI?"                                     → definition_or_explanation
- "What does the Smith 2022 paper say about drought?" → paper_specific_question
- "Which datasets for drought in Central Asia?"       → dataset_recommendation
- "Recommend papers on sea ice decline"               → paper_recommendation
- "How do I compute SPEI?"                            → methodology_support
- "How can I study drought impacts on vegetation?"    → research_starter
- Out-of-scope / unclear                              → other

Answer-mode mapping:
- direct_answer   ← definition_or_explanation, paper_specific_question
- recommendation  ← dataset_recommendation, paper_recommendation, methodology_support, research_starter
- hybrid          ← questions that mix explanation + recommendation
- (other) decide by query content

Rules:
- If conversation context is present, INHERIT the topic / region / timescale
  from it. Produce a COMPLETE local_query / openalex_query / zenodo_query even
  when the current query is a short follow-up.
- If the context indicates intent shifted (e.g. shift hint provided), use the
  shifted intent as the classification.
- Return null for fields the query (+context) does not specify.

For region_bbox, use standard Earth-science bounding boxes. Examples:
  Central Asia [40, 30, 90, 55]; Tibetan Plateau [73, 25, 104, 40];
  Arctic [-180, 66.5, 180, 90]; Sahel [-20, 10, 40, 20]
Return null for vague/global/unspecified regions.

For parsed_timescale, convert to ISO dates. "to present" → today's date.
"past N years" → today - N years. "satellite era" → 1972-01-01 onwards.
Return null if no timescale given.
"""


def parse_query(
    user_query: str,
    history: list[dict] | None = None,
) -> tuple[ParsedQuery, ConversationDigest]:
    """Parse the user's query.

    Returns (ParsedQuery, ConversationDigest). The digest is meaningful only
    when `history` was non-empty; otherwise it has summary="" and default flags.
    The caller is responsible for threading digest.wants_fresh_recommendations
    down into the reranker.
    """
    cfg = get_settings()
    client = OpenAI(api_key=openai_api_key())

    # Stage 1 — only when history is present
    digest = _analyze_conversation(history or [], user_query, cfg)

    # Stage 2 — parse, optionally with the Stage-1 summary as context
    if digest.summary or digest.intent_shift_hint:
        shift_note = (
            f" [Intent may have shifted to: {digest.intent_shift_hint}]"
            if digest.intent_shift_hint else ""
        )
        context_section = (
            "\nConversation context:\n"
            f"{digest.summary}{shift_note}\n"
        )
    else:
        context_section = ""

    prompt = _PARSE_PROMPT.format(query=user_query, context_section=context_section)

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
    # `or <fallback>` (instead of .get(key, fallback)) so explicit JSON null
    # also falls back. Important for short follow-ups where LLM may leave
    # local_query null.
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
        wants_fresh_recommendations=bool(digest.wants_fresh_recommendations),
    )

    # debug output
    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_parsed_query.json", "w") as f:
        f.write(parsed.model_dump_json(indent=2))
    with open(debug_dir / "last_conversation_digest.json", "w") as f:
        f.write(digest.model_dump_json(indent=2))

    return parsed, digest


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
