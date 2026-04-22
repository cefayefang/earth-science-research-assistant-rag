"""
Pipeline router: intent routing, session management, and all handler logic.

main.py imports only `_run_pipeline` from here. Everything else in this
module is internal to the pipeline execution path.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from .core.schemas import (
    FinalAnswer, SessionState, SessionPaper, SessionDataset,
    SessionDatasetMetadata, CachedChunk,
)
from .pipeline.query_parser import parse_query
from .clients.openalex_client import fetch_openalex_papers
from .pipeline.paper_matcher import match_papers
from .pipeline.dataset_retriever import retrieve_datasets
from .pipeline.chunk_retriever import retrieve_chunks, retrieve_chunks_for_paper
from .pipeline.linker import build_links
from .pipeline.reranker import rerank_papers, rerank_datasets
from .pipeline.answer_generator import generate_answer
from .pipeline.evidence_cache_writer import write_evidence_cache
from .pipeline.intent_classifier import classify_intent, find_chunks_for_target, parse_target_position
from .ingestion.dataset_normalizer import load_normalized_datasets
from .core.config import get_settings, get_openai_client


# ── Simple answer helpers ────────────────────────────────────────────────────

def _simple_answer(text: str, mode: str = "direct_answer") -> FinalAnswer:
    return FinalAnswer(answer_mode=mode, direct_answer=text, final_text=text)


# ── Special-case handlers ────────────────────────────────────────────────────

def _handle_chitchat(user_query: str, history: list, cfg: dict) -> dict:
    """Friendly reply without any retrieval."""
    client = get_openai_client()
    messages: list[dict] = []
    for m in (history or [])[-6:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if content and content != "⏳ Thinking…" and role in ("user", "assistant"):
            messages.append({"role": role, "content": content[:400]})
    messages.append({
        "role": "user",
        "content": (
            "You are a friendly Earth science research assistant. "
            "Respond conversationally and briefly.\n\n" + user_query
        ),
    })
    resp = client.chat.completions.create(
        model=cfg["llm"]["default_model"],
        messages=messages,
        temperature=0.7,
        max_tokens=200,
    )
    text = resp.choices[0].message.content.strip()
    return {
        "answer": _simple_answer(text),
        "ranked_papers": [], "ranked_datasets": [], "chunk_candidates": [],
        "openalex_papers": [], "zenodo_records": [],
        "evidence_block_text": "", "cache_dir": "",
    }


def _handle_out_of_scope() -> dict:
    text = (
        "That question is outside the scope of this Earth science research assistant. "
        "I can help you find datasets, papers, and methodologies related to Earth science topics "
        "such as climate, hydrology, remote sensing, and more."
    )
    return {
        "answer": _simple_answer(text),
        "ranked_papers": [], "ranked_datasets": [], "chunk_candidates": [],
        "openalex_papers": [], "zenodo_records": [],
        "evidence_block_text": "", "cache_dir": "",
    }


def _answer_from_chunks(
    user_query: str,
    chunks: list,
    history: list,
    cfg: dict,
    paper_title: str | None = None,
) -> dict:
    """Compose a detail-followup answer from a concrete set of chunks, without
    touching the full RAG pipeline. Shared by every cache-hit path in
    `_handle_detail_followup`.
    """
    client = get_openai_client()
    chunk_text = "\n\n".join(
        f"[{c.local_id} | {c.section_guess or 'unknown'}]\n{c.text}"
        for c in chunks[:5]
    )
    context_intro = (
        f'Excerpts from the paper "{paper_title}":'
        if paper_title
        else "Excerpts:"
    )
    messages: list[dict] = []
    for m in (history or [])[-4:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if content and content != "⏳ Thinking…" and role in ("user", "assistant"):
            messages.append({"role": role, "content": content[:400]})
    messages.append({
        "role": "user",
        "content": (
            "You are an Earth science research assistant. Answer the user's question "
            "using ONLY the text excerpts below. If the excerpts do not contain enough "
            "information to answer, say so explicitly — do not speculate.\n\n"
            f"{context_intro}\n{chunk_text}\n\n"
            f"Question: {user_query}\n\n"
            "Be concise and accurate. Do not add information not present in the excerpts."
        ),
    })
    resp = client.chat.completions.create(
        model=cfg["llm"]["default_model"],
        messages=messages,
        temperature=0.1,
        max_tokens=500,
    )
    text = resp.choices[0].message.content.strip()
    return {
        "answer": _simple_answer(text),
        "ranked_papers": [], "ranked_datasets": [],
        "chunk_candidates": [
            type("C", (), {
                "chunk_id": c.chunk_id,
                "local_id": c.local_id,
                "text": c.text,
                "section_guess": c.section_guess,
                "openalex_id": getattr(c, "openalex_id", None),
                "chunk_score": getattr(c, "chunk_score", 1.0),
            })()
            for c in chunks
        ],
        "openalex_papers": [], "zenodo_records": [],
        "evidence_block_text": "", "cache_dir": "",
    }


def _answer_from_dataset(
    user_query: str,
    target_dataset: SessionDataset,
    session: SessionState,
    history: list,
    cfg: dict,
) -> dict:
    """Answer a detail question about a specific recommended dataset."""
    all_datasets = load_normalized_datasets()
    ds_meta = next(
        (d for d in all_datasets if d.dataset_id == target_dataset.dataset_id),
        None,
    )
    if ds_meta is None:
        ds_meta = next(
            (e for e in (session.last_turn_ephemeral_dataset_metadata or [])
             if e.dataset_id == target_dataset.dataset_id),
            None,
        )

    mention_chunks: list = []
    if ds_meta and session.last_turn_chunks:
        name_tokens = [t for t in (ds_meta.display_name or "").lower().split() if len(t) > 3]
        kw_tokens = [k.lower() for k in (ds_meta.keywords or [])[:5] if k and len(k) > 3]
        probe_terms = list({*name_tokens, *kw_tokens})
        if probe_terms:
            for c in session.last_turn_chunks:
                txt_lower = c.text.lower()
                if any(term in txt_lower for term in probe_terms):
                    mention_chunks.append(c)
        mention_chunks = mention_chunks[:3]

    meta_lines = [f"Name: {target_dataset.title}"]
    if ds_meta:
        if ds_meta.source:
            meta_lines.append(f"Source: {ds_meta.source}")
        if ds_meta.provider:
            meta_lines.append(f"Provider: {ds_meta.provider}")
        if ds_meta.doi:
            meta_lines.append(f"DOI: {ds_meta.doi}")
        if ds_meta.variables:
            meta_lines.append(f"Variables: {', '.join(ds_meta.variables[:10])}")
        if ds_meta.keywords:
            meta_lines.append(f"Keywords: {', '.join(ds_meta.keywords[:10])}")
        if ds_meta.spatial_info:
            meta_lines.append(f"Spatial coverage: {ds_meta.spatial_info}")
        if ds_meta.temporal_info:
            meta_lines.append(f"Temporal coverage: {ds_meta.temporal_info}")
        if ds_meta.description:
            meta_lines.append(f"Description:\n{ds_meta.description[:1500]}")
    else:
        meta_lines.append(
            "(Normalized metadata not found for this dataset_id — answer from title only.)"
        )
    metadata_text = "\n".join(meta_lines)

    excerpts_text = ""
    if mention_chunks:
        excerpts_text = "\n\n".join(
            f"[{c.local_id} | {c.section_guess or 'unknown'}]\n{c.text[:500]}"
            for c in mention_chunks
        )

    messages: list[dict] = []
    for m in (history or [])[-4:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if content and content != "⏳ Thinking…" and role in ("user", "assistant"):
            messages.append({"role": role, "content": content[:400]})

    evidence_parts = [f"Dataset metadata (authoritative):\n{metadata_text}"]
    if excerpts_text:
        evidence_parts.append(
            "Literature excerpts that mention this dataset (for usage context, "
            "NOT for claims about dataset capabilities):\n" + excerpts_text
        )
    evidence = "\n\n".join(evidence_parts)

    messages.append({
        "role": "user",
        "content": (
            "You are an Earth science research assistant. Answer the user's "
            "question about a specific dataset using ONLY the evidence below. "
            "Dataset capabilities (variables, coverage, provider) must come from "
            "the metadata section. The literature excerpts may be used to "
            "describe how papers USE the dataset, but do NOT invent capabilities "
            "or facts about the dataset itself. If the evidence does not contain "
            "enough information to answer, say so explicitly.\n\n"
            f"{evidence}\n\n"
            f"Question: {user_query}\n\n"
            "Be concise and accurate."
        ),
    })

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=cfg["llm"]["default_model"],
        messages=messages,
        temperature=0.1,
        max_tokens=500,
    )
    text = resp.choices[0].message.content.strip()

    return {
        "answer": _simple_answer(text),
        "ranked_papers": [], "ranked_datasets": [],
        "chunk_candidates": [
            type("C", (), {
                "chunk_id": c.chunk_id,
                "local_id": c.local_id,
                "text": c.text,
                "section_guess": c.section_guess,
                "openalex_id": getattr(c, "openalex_id", None),
                "chunk_score": getattr(c, "chunk_score", 1.0),
            })()
            for c in mention_chunks
        ],
        "openalex_papers": [], "zenodo_records": [],
        "evidence_block_text": "", "cache_dir": "",
    }


def _handle_detail_followup(
    user_query: str,
    target_ref: str | None,
    target_kind: str | None,
    history: list,
    cfg: dict,
    session: SessionState,
) -> dict:
    """Route detail follow-up by (target_kind, position)."""
    position = parse_target_position(target_ref)

    if target_kind == "dataset":
        if position:
            target_dataset = next(
                (d for d in session.last_recommended_datasets if d.position == position),
                None,
            )
            if target_dataset:
                return _answer_from_dataset(
                    user_query=user_query,
                    target_dataset=target_dataset,
                    session=session,
                    history=history,
                    cfg=cfg,
                )
        return _run_full_rag(user_query, history=history)

    target_paper = None
    if position:
        target_paper = next(
            (p for p in session.last_recommended_papers if p.position == position),
            None,
        )

    if target_kind is None and not target_paper and position:
        target_dataset = next(
            (d for d in session.last_recommended_datasets if d.position == position),
            None,
        )
        if target_dataset:
            return _answer_from_dataset(
                user_query=user_query,
                target_dataset=target_dataset,
                session=session,
                history=history,
                cfg=cfg,
            )

    # Path 1: local paper — per-paper semantic retrieval
    if target_paper and target_paper.local_id:
        chunks = retrieve_chunks_for_paper(
            target_paper.local_id,
            user_query,
            top_k=5,
        )
        if chunks:
            return _answer_from_chunks(
                user_query=user_query,
                chunks=chunks,
                history=history,
                cfg=cfg,
                paper_title=target_paper.title,
            )

    # Path 2: external paper (no local_id) — enriched full RAG
    if target_paper and not target_paper.local_id:
        enriched = f'{user_query} — regarding the paper titled "{target_paper.title}"'
        return _run_full_rag(enriched, history=history)

    # Path 3a: legacy keyword fallback on last_turn_chunks
    kw_chunks = find_chunks_for_target(session, target_ref)
    if kw_chunks:
        return _answer_from_chunks(
            user_query=user_query,
            chunks=kw_chunks,
            history=history,
            cfg=cfg,
            paper_title=target_paper.title if target_paper else None,
        )

    # Path 3b: nothing matched — plain / enriched full RAG
    if target_paper:
        enriched = f'{user_query} — regarding the paper titled "{target_paper.title}"'
        return _run_full_rag(enriched, history=history)
    return _run_full_rag(user_query, history=history)


# ── Full RAG pipeline ────────────────────────────────────────────────────────

def _run_full_rag(
    user_query: str,
    history: list | None = None,
    exclude_paper_ids: list[str] | None = None,
    exclude_dataset_ids: list[str] | None = None,
    wants_fresh: bool = False,
    requested_count: int | None = None,
    requested_count_target: str | None = None,
) -> dict:
    parsed = parse_query(
        user_query,
        history=history,
        wants_fresh=wants_fresh,
        requested_count=requested_count,
        requested_count_target=requested_count_target,
    )

    effective_exclude_papers = exclude_paper_ids if (wants_fresh and exclude_paper_ids) else None
    effective_exclude_datasets = exclude_dataset_ids if (wants_fresh and exclude_dataset_ids) else None

    # Parallelize OpenAlex HTTP fetch and local chunk retrieval — both only need `parsed`.
    with ThreadPoolExecutor(max_workers=2) as executor:
        openalex_future = (
            executor.submit(fetch_openalex_papers, parsed.openalex_query)
            if parsed.openalex_query else None
        )
        chunks_future = executor.submit(retrieve_chunks, parsed)
        openalex_papers = openalex_future.result() if openalex_future else []
        chunk_candidates = chunks_future.result()

    openalex_dois = {p.doi.lower() for p in openalex_papers if p.doi}
    paper_matches = match_papers(openalex_papers)
    dataset_candidates, zenodo_records = retrieve_datasets(parsed, openalex_dois)
    build_links(dataset_candidates, chunk_candidates, openalex_papers)
    ranked_papers = rerank_papers(
        openalex_papers, paper_matches, chunk_candidates, parsed.local_query,
        exclude_paper_ids=effective_exclude_papers,
    )
    ranked_datasets = rerank_datasets(
        dataset_candidates,
        exclude_dataset_ids=effective_exclude_datasets,
    )
    answer, evidence_block_text = generate_answer(
        parsed, ranked_papers[:10], ranked_datasets[:10], chunk_candidates[:10]
    )

    # Write evidence cache in background — don't block the response.
    cache_dir = ""
    threading.Thread(
        target=write_evidence_cache,
        kwargs=dict(
            query=user_query,
            parsed=parsed,
            openalex_papers=openalex_papers,
            zenodo_records=zenodo_records,
            local_dataset_candidates=ranked_datasets,
            chunk_candidates=chunk_candidates,
            evidence_block_text=evidence_block_text,
            final_answer=answer,
        ),
        daemon=True,
    ).start()

    return {
        "parsed": parsed,
        "ranked_papers": ranked_papers,
        "ranked_datasets": ranked_datasets,
        "chunk_candidates": chunk_candidates,
        "openalex_papers": openalex_papers,
        "zenodo_records": zenodo_records,
        "answer": answer,
        "evidence_block_text": evidence_block_text,
        "cache_dir": str(cache_dir),
    }


# ── Session state management ─────────────────────────────────────────────────

def _update_session(
    session: SessionState,
    result: dict,
    intent_type: str,
) -> SessionState:
    """Build updated session state after a RAG turn."""
    answer = result.get("answer")
    chunk_candidates = result.get("chunk_candidates", [])

    new_paper_ids: list[str] = []
    new_dataset_ids: list[str] = []
    new_session_papers: list[SessionPaper] = []
    new_session_datasets: list[SessionDataset] = []

    if answer:
        for i, p in enumerate(answer.recommended_papers or [], 1):
            pid = p.openalex_id or p.local_id
            if pid:
                new_paper_ids.append(pid)
            new_session_papers.append(SessionPaper(
                position=i,
                title=p.title,
                local_id=p.local_id,
                openalex_id=p.openalex_id,
            ))
        for i, d in enumerate(answer.recommended_datasets or [], 1):
            if d.dataset_id:
                new_dataset_ids.append(d.dataset_id)
            new_session_datasets.append(SessionDataset(
                position=i,
                title=d.dataset_name,
                dataset_id=d.dataset_id,
            ))

    new_chunks = [
        CachedChunk(
            chunk_id=c.chunk_id,
            local_id=c.local_id,
            text=c.text,
            section_guess=c.section_guess,
        )
        for c in (chunk_candidates or [])[:10]
    ]

    zenodo_records = result.get("zenodo_records") or []
    recommended_ds_ids_set = set(new_dataset_ids)
    new_ephemeral_ds_meta = [
        SessionDatasetMetadata(
            dataset_id=r.dataset_id,
            display_name=r.display_name,
            source=r.source,
            provider=r.provider,
            doi=r.doi,
            description=r.description,
            variables=r.variables or [],
            keywords=r.keywords or [],
            spatial_info=r.spatial_info,
            temporal_info=r.temporal_info,
        )
        for r in zenodo_records
        if r.dataset_id in recommended_ds_ids_set
    ]

    if intent_type == "re_recommend":
        all_paper_ids = list(dict.fromkeys(session.recommended_paper_ids + new_paper_ids))
        all_dataset_ids = list(dict.fromkeys(session.recommended_dataset_ids + new_dataset_ids))
    else:
        all_paper_ids = new_paper_ids
        all_dataset_ids = new_dataset_ids

    return SessionState(
        recommended_paper_ids=all_paper_ids,
        recommended_dataset_ids=all_dataset_ids,
        last_recommended_papers=new_session_papers,
        last_recommended_datasets=new_session_datasets,
        last_turn_chunks=new_chunks,
        last_turn_ephemeral_dataset_metadata=new_ephemeral_ds_meta,
        turn_count=session.turn_count + 1,
    )


# ── Main entry point ─────────────────────────────────────────────────────────

def _run_pipeline(
    user_query: str,
    history: list | None = None,
    exclude_paper_ids: list[str] | None = None,
    exclude_dataset_ids: list[str] | None = None,
    session_state: dict | None = None,
) -> dict:
    """Route the query by intent, run the appropriate handler, update session."""
    cfg = get_settings()
    session = SessionState(**(session_state or {})) if session_state else SessionState()

    history_dicts = history or []
    intent = classify_intent(user_query, history_dicts, cfg)
    print(f"  [intent] {intent.intent_type} (conf={intent.confidence:.2f})")

    if intent.intent_type == "chitchat":
        result = _handle_chitchat(user_query, history_dicts, cfg)

    elif intent.intent_type == "out_of_scope":
        result = _handle_out_of_scope()

    elif intent.intent_type == "detail_followup":
        result = _handle_detail_followup(
            user_query,
            intent.target_ref,
            intent.target_kind,
            history_dicts,
            cfg,
            session,
        )

    else:
        # new_question or re_recommend → full RAG
        if intent.intent_type == "re_recommend":
            eff_query = intent.rewritten_query or user_query
            eff_excl_papers = session.recommended_paper_ids or None
            eff_excl_datasets = session.recommended_dataset_ids or None
            wants_fresh = True
        else:
            eff_query = user_query
            eff_excl_papers = exclude_paper_ids
            eff_excl_datasets = exclude_dataset_ids
            wants_fresh = False

        result = _run_full_rag(
            eff_query,
            history_dicts,
            eff_excl_papers,
            eff_excl_datasets,
            wants_fresh=wants_fresh,
            requested_count=intent.requested_count,
            requested_count_target=intent.requested_count_target,
        )

    result["intent_type"] = intent.intent_type

    if intent.intent_type in ("new_question", "re_recommend"):
        result["session_state"] = _update_session(session, result, intent.intent_type)
    elif intent.intent_type == "detail_followup":
        chunk_candidates = result.get("chunk_candidates", [])
        new_chunks = [
            CachedChunk(
                chunk_id=c.chunk_id,
                local_id=c.local_id,
                text=c.text,
                section_guess=c.section_guess,
            )
            for c in chunk_candidates[:10]
        ]
        result["session_state"] = SessionState(
            recommended_paper_ids=session.recommended_paper_ids,
            recommended_dataset_ids=session.recommended_dataset_ids,
            last_recommended_papers=session.last_recommended_papers,
            last_recommended_datasets=session.last_recommended_datasets,
            last_turn_chunks=new_chunks if new_chunks else session.last_turn_chunks,
            last_turn_ephemeral_dataset_metadata=session.last_turn_ephemeral_dataset_metadata,
            turn_count=session.turn_count + 1,
        )
    else:  # chitchat / out_of_scope
        result["session_state"] = session

    return result
