import json
from google import genai
from google.genai import types
from .config import get_settings, gemini_api_key, ROOT
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
  "timescale": "<time period or null>",
  "local_query": "<enriched retrieval string for local search, include variables and region>",
  "openalex_query": "<concise query string for OpenAlex paper search or null>",
  "zenodo_query": "<concise query string for Zenodo dataset search or null>",
  "must_have_constraints": {{
    "region": <true if region is essential, false otherwise>,
    "timescale": <true if timescale is essential, false otherwise>
  }}
}}

Rules:
- answer_mode=direct_answer for definitions, explanations, and paper-specific questions
- answer_mode=recommendation for dataset/paper/research starting point requests
- answer_mode=hybrid for questions needing both explanation and recommendations
- local_query should be rich: combine phenomenon + variables + region + timescale
- openalex_query and zenodo_query should be concise keyword strings
- Return null for fields that are not present in the query
"""


def parse_query(user_query: str) -> ParsedQuery:
    cfg = get_settings()
    client = genai.Client(api_key=gemini_api_key())

    prompt = _PROMPT.format(query=user_query)
    response = client.models.generate_content(
        model=cfg["llm"]["default_model"],
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=cfg["llm"]["temperature"],
            max_output_tokens=cfg["llm"]["max_output_tokens"],
        ),
    )
    text = response.text.strip()

    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    data = json.loads(text)

    constraints = data.get("must_have_constraints", {})
    parsed = ParsedQuery(
        original_query=data.get("original_query", user_query),
        intent=data.get("intent", "other"),
        answer_mode=data.get("answer_mode", "hybrid"),
        phenomenon=data.get("phenomenon"),
        variables=data.get("variables", []),
        region=data.get("region"),
        timescale=data.get("timescale"),
        local_query=data.get("local_query", user_query),
        openalex_query=data.get("openalex_query"),
        zenodo_query=data.get("zenodo_query"),
        must_have_constraints=MustHaveConstraints(
            region=constraints.get("region", False),
            timescale=constraints.get("timescale", False),
        ),
    )

    # save debug output
    debug_dir = ROOT / "generated" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "last_parsed_query.json", "w") as f:
        f.write(parsed.model_dump_json(indent=2))

    return parsed
