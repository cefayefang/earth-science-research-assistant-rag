import gradio as gr
import httpx

API_URL = "http://localhost:8000"

CSS = """
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
h1 { font-size: 1.6rem !important; font-weight: 700 !important; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; margin-top: 0.8em !important; }
.label-wrap { font-weight: 600 !important; font-size: 0.85rem !important; color: #555 !important; }
.chatbot .message { font-size: 0.95rem !important; line-height: 1.6 !important; }
/* Side panel: never cap height, always show all listed recommendations. */
.side-panel {
    font-size: 0.88rem !important;
    line-height: 1.55 !important;
    max-height: none !important;
    overflow: visible !important;
    height: auto !important;
}
.side-panel > *, .side-panel .prose, .side-panel .markdown-body {
    max-height: none !important;
    overflow: visible !important;
    height: auto !important;
}
.gr-button-primary { font-weight: 600 !important; }
"""

BADGE = {
    "nasa_cmr": "🛰 NASA",
    "copernicus_cds": "🌍 CDS",
    "cdse": "🛸 CDSE",
    "zenodo": "📦 Zenodo",
    "stac": "🗂 STAC",
}

STRENGTH_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def _flatten_content(content) -> str:
    """Normalize Gradio chatbot message content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts).strip()
    return str(content)


def check_health() -> str:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        return "● Online" if r.status_code == 200 else "● Server error"
    except Exception:
        return "● Offline — run ./start.sh first"


def _fmt_datasets(items: list[dict]) -> str:
    if not items:
        return "_No datasets retrieved._"
    lines = []
    for i, d in enumerate(items[:5], 1):
        badge = BADGE.get(d.get("source", ""), d.get("source", "").upper())
        icon = STRENGTH_ICON.get(d.get("evidence_strength", "low"), "⚪")
        name = d.get("dataset_name", "—")
        doi = d.get("doi", "")
        doi_str = f"  `{doi}`" if doi else ""
        lines.append(f"**{i}. {name}**  {badge}  {icon}{doi_str}")
    return "\n\n".join(lines)


def _fmt_papers(items: list[dict]) -> str:
    if not items:
        return "_No papers retrieved._"
    lines = []
    for i, p in enumerate(items[:5], 1):
        title = p.get("title", "—")
        year = p.get("year", "—")
        tag = "✅ full-text" if p.get("evidence_level") == "fulltext_supported" else "📄 metadata"
        lines.append(f"**{i}. {title}** ({year})  {tag}")
    return "\n\n".join(lines)


def ask(query: str, history: list, session_state: dict):
    """Main chat handler.

    session_state is a gr.State dict that mirrors the backend SessionState:
      recommended_paper_ids, recommended_dataset_ids,
      last_recommended_papers, last_recommended_datasets,
      last_turn_chunks, turn_count
    """
    if not query.strip():
        yield history, _fmt_datasets([]), _fmt_papers([]), "", session_state
        return

    pending = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "⏳ Thinking…"},
    ]
    yield pending, _fmt_datasets([]), _fmt_papers([]), "", session_state

    # Build conversation history for the API (last 10 turns = 20 messages, excluding pending)
    prior = []
    for m in history[-6:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _flatten_content(m.get("content"))
        if not content or content == "⏳ Thinking…":
            continue
        prior.append({"role": role, "content": content})

    try:
        r = httpx.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "history": prior or None,
                "session_state": session_state or None,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.TimeoutException:
        msg = "Request timed out. Please try again."
        yield (
            history + [{"role": "user", "content": query}, {"role": "assistant", "content": msg}],
            _fmt_datasets([]), _fmt_papers([]), "", session_state,
        )
        return
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        msg = f"Error: {detail}"
        yield (
            history + [{"role": "user", "content": query}, {"role": "assistant", "content": msg}],
            _fmt_datasets([]), _fmt_papers([]), "", session_state,
        )
        return
    except Exception as e:
        msg = f"Error: {e}"
        yield (
            history + [{"role": "user", "content": query}, {"role": "assistant", "content": msg}],
            _fmt_datasets([]), _fmt_papers([]), "", session_state,
        )
        return

    answer = data.get("answer", "")
    notes = data.get("uncertainty_notes", [])
    notes_md = "\n".join(f"⚠️ {n}" for n in notes) if notes else ""

    # Receive updated session state from backend
    updated_session = data.get("session_state") or session_state or {}

    final_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    yield (
        final_history,
        _fmt_datasets(data.get("recommended_datasets", [])),
        _fmt_papers(data.get("recommended_papers", [])),
        notes_md,
        updated_session,
    )


with gr.Blocks(title="Earth Science Research Assistant") as demo:
    gr.Markdown("# 🌍 Earth Science Research Assistant\nDataset and paper recommendations grounded in Earth science literature.")

    # Full session state — replaces the old last_ids_state
    session_state_var = gr.State({})

    with gr.Row():
        health_box = gr.Textbox(value=check_health(), label="Status", interactive=False, max_lines=1, scale=1)

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=600)
            with gr.Row():
                query_box = gr.Textbox(
                    placeholder="e.g. What datasets can I use to study drought in Central Asia?",
                    label="Question",
                    lines=2,
                    scale=5,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    ["What datasets can I use to study drought impacts on vegetation in Central Asia?"],
                    ["Which papers study sea surface temperature changes in the North Atlantic?"],
                    ["What is the NDVI and how is it used in remote sensing?"],
                    ["Recommend datasets for studying Arctic sea ice loss since 2000."],
                    ["What methods are used to downscale climate model outputs?"],
                ],
                inputs=query_box,
            )

        with gr.Column(scale=2, elem_classes=["side-panel"]):
            datasets_box = gr.Markdown(label="📊 Datasets", value="_Ask a question to see recommendations._")
            gr.Markdown("---")
            papers_box = gr.Markdown(label="📄 Papers", value="")
            notes_box = gr.Markdown(label="⚠️ Notes", value="")

    submit_btn.click(
        fn=ask,
        inputs=[query_box, chatbot, session_state_var],
        outputs=[chatbot, datasets_box, papers_box, notes_box, session_state_var],
    )
    query_box.submit(
        fn=ask,
        inputs=[query_box, chatbot, session_state_var],
        outputs=[chatbot, datasets_box, papers_box, notes_box, session_state_var],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=CSS)
