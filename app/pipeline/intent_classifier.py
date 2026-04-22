"""
Pre-RAG intent classifier.

One fast LLM call before the main pipeline determines routing:

  chitchat       — greetings, thanks, small talk; skip retrieval
  new_question   — fresh earth-science query; full RAG pipeline
  re_recommend   — "more papers / other datasets"; RAG with exclusion list
  detail_followup — "tell me about paper 1"; prefer cached chunks
  out_of_scope   — not earth science; polite refusal
"""
import json
import re
from ..core.config import get_settings, get_openai_client
from ..core.schemas import IntentClassification

_MAX_HISTORY_CHARS = 400

_PROMPT = """\
You are a routing agent for an Earth Science Research Assistant.
Given the conversation history and the user's latest message, classify the intent.

Recent conversation (most recent last):
{history}

User's latest message: {query}

Output JSON with this exact structure:
{{
  "intent_type": "<chitchat|new_question|re_recommend|detail_followup|out_of_scope>",
  "confidence": <float 0.0-1.0>,
  "target_ref": "<string identifying the item the user asks about, or null>",
  "target_kind": "<paper|dataset, or null>",
  "rewritten_query": "<full retrieval query string capturing the topic, or null>",
  "requested_count": <integer, or null>,
  "requested_count_target": "<datasets|papers|methodology, or null>"
}}

Definitions:
- chitchat: greetings, thanks, small talk. ("hi", "thanks!")

- new_question: a fresh earth-science research question. Requires earth-science
  context (datasets, papers, climate, remote sensing, hydrology, etc.).

- re_recommend: user wants MORE items on the same topic beyond what was shown.
  Signals: "more", "other", "another", "different", "any others?".
  → set rewritten_query to a full retrieval string capturing the established topic + variables.

- detail_followup: user wants details about a SPECIFIC item previously mentioned.
  ("tell me about the first paper", "what variables does that dataset cover?")
  → set target_ref to identify which item (e.g. "paper 1", "second dataset").
  → set target_kind = "paper" / "dataset" when the user names the kind;
    leave null only when genuinely ambiguous ("the first one").

- out_of_scope: clearly NOT earth-science. ("weather today", "cook pasta")

Decision rules:
- If ambiguous between new_question and detail_followup, prefer new_question.
- Confidence: 0.95 obvious, 0.7–0.85 moderately clear, 0.5–0.6 ambiguous.

requested_count / requested_count_target (only for new_question / re_recommend;
null for all other intents):
- Set requested_count ONLY when the user explicitly gave a number. Recognize
  digits ("2", "3") and English words ("two", "three", "four", "five").
- Set requested_count_target only when the user named the kind
  (datasets / papers / methodology). Bare numbers → leave target null.
- Examples:
    "give me 3 papers"  → count=3, target="papers"
    "more datasets"     → count=null  (no number)
"""


def classify_intent(
    user_query: str,
    history: list[dict] | None,
    cfg: dict,
) -> IntentClassification:
    """Fast pre-pipeline intent classification. Falls back to new_question on error."""
    history_text = _format_history(history)
    prompt = _PROMPT.format(history=history_text or "(no history)", query=user_query)

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=cfg["llm"]["default_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content.strip())
        intent_type = data.get("intent_type", "new_question")
        valid = {"chitchat", "new_question", "re_recommend", "detail_followup", "out_of_scope"}
        if intent_type not in valid:
            intent_type = "new_question"

        # Coerce requested_count: must be a positive int or null
        raw_count = data.get("requested_count")
        requested_count: int | None = None
        if isinstance(raw_count, int) and raw_count > 0:
            requested_count = raw_count
        elif isinstance(raw_count, str) and raw_count.strip().isdigit():
            n = int(raw_count.strip())
            if n > 0:
                requested_count = n

        # Coerce target: must be one of the known primary-list names or null
        raw_target = data.get("requested_count_target")
        valid_targets = {"datasets", "papers", "methodology"}
        requested_count_target = (
            raw_target if (isinstance(raw_target, str) and raw_target in valid_targets) else None
        )

        # Coerce target_kind for detail_followup
        raw_kind = data.get("target_kind")
        valid_kinds = {"paper", "dataset"}
        target_kind = (
            raw_kind if (isinstance(raw_kind, str) and raw_kind in valid_kinds) else None
        )

        return IntentClassification(
            intent_type=intent_type,
            confidence=float(data.get("confidence", 0.8)),
            target_ref=data.get("target_ref") or None,
            target_kind=target_kind,
            rewritten_query=data.get("rewritten_query") or None,
            requested_count=requested_count,
            requested_count_target=requested_count_target,
        )
    except Exception as e:
        print(f"  [intent_classifier warn] {e}")
        return IntentClassification(intent_type="new_question", confidence=0.5)


# Shared ordinal vocabulary — public so other modules can reuse if needed.
_ORDINAL_TOKENS = {
    "1": 1, "first": 1, "1st": 1, "第一": 1,
    "2": 2, "second": 2, "2nd": 2, "第二": 2,
    "3": 3, "third": 3, "3rd": 3, "第三": 3,
    "4": 4, "fourth": 4, "4th": 4, "第四": 4,
    "5": 5, "fifth": 5, "5th": 5, "第五": 5,
}


def parse_target_position(target_ref: str | None) -> int | None:
    """Extract a 1-indexed position from phrases like "paper 1", "the second one",
    "第三篇 paper", "dataset 2", etc. Returns None when no ordinal is found.

    Priority:
      1. Explicit "paper N" / "dataset N" wins (highest-signal form).
      2. Otherwise any known ordinal token anywhere in the string.
    """
    if not target_ref:
        return None
    ref = target_ref.lower()
    m = re.search(r'(?:paper|dataset)\s*(\d+)', ref)
    if m:
        try:
            n = int(m.group(1))
            if n > 0:
                return n
        except ValueError:
            pass
    for token, pos in _ORDINAL_TOKENS.items():
        if token in ref:
            return pos
    return None


def find_chunks_for_target(session, target_ref: str | None) -> list:
    """Return cached chunks relevant to the follow-up target.

    Tries ordinal matching first ("paper 1", "the second one"), then falls back
    to keyword matching against chunk text. Returns at most 3 chunks. Used as
    a LAST-RESORT fallback now — the primary path for paper detail questions
    goes through chunk_retriever.retrieve_chunks_for_paper, which searches the
    entire paper instead of just last_turn_chunks.
    """
    if not session.last_turn_chunks or not target_ref:
        return []

    position = parse_target_position(target_ref)
    if position:
        for p in session.last_recommended_papers:
            if p.position == position and p.local_id:
                hits = [c for c in session.last_turn_chunks if c.local_id == p.local_id]
                if hits:
                    return hits[:3]

    # Keyword fallback
    ref_lower = target_ref.lower()
    keywords = [w for w in ref_lower.split() if len(w) > 3]
    if keywords:
        matched = [
            c for c in session.last_turn_chunks
            if any(kw in c.text.lower() for kw in keywords)
        ]
        return matched[:2]

    return []


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    parts = []
    for msg in history[-6:]:
        role = (msg.get("role") or "").capitalize()
        content = msg.get("content") or ""
        if not content or content == "⏳ Thinking…":
            continue
        if len(content) > _MAX_HISTORY_CHARS:
            content = content[:_MAX_HISTORY_CHARS] + "…"
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
