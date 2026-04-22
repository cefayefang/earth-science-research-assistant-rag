"""
Run the v2 evaluation set against two variants:

  V0 — no-RAG baseline: direct LLM call with the query, no evidence
  V1 — full system:     parse → OpenAlex + Zenodo + local retrieval →
                        rerank → linker → grounded answer generator

Outputs under evaluation/results/:
  per_sample_v0.jsonl
  per_sample_v1.jsonl
  summary.csv
  comparison_table.md

Run from the project root:
  python -m evaluation.run_eval
"""
import json
import csv
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict

# Make sure we can import app.* when running from project root
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from app.core.config import get_settings, openai_api_key
from app.router import _run_pipeline
from evaluation.metrics import (
    recall_at_k, recall_from_pool, mrr,
    precision_at_k, f1_at_k,
    parse_citation_tags, compute_grounding_rate,
    collect_emitted_tags, citation_coverage,
    is_abstention_correct, methodology_cites_chunks_rate,
    rouge_l_score,
)


EVAL_FILE = HERE / "eval_set_v2.json"
RESULTS_DIR = HERE / "results"


# ── V0: no-RAG baseline ──────────────────────────────────────────────────────

_V0_PROMPT = """You are an Earth science research assistant. Answer the user's query.

For recommendation queries, list concrete datasets and papers.
For definition queries, provide a concise direct answer.

User query: {query}

Return JSON:
{{
  "direct_answer": "string or null",
  "recommended_datasets": [{{"dataset_name": "...", "source": "..."}}],
  "recommended_papers": [{{"title": "...", "year": 2020}}],
  "uncertainty_notes": ["..."]
}}
"""


def run_v0(query: str, client: OpenAI, model: str) -> dict:
    """Direct LLM call with no retrieval, no evidence cache."""
    prompt = _V0_PROMPT.format(query=query)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    latency = time.time() - t0
    raw = resp.choices[0].message.content.strip()
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {"direct_answer": raw, "recommended_datasets": [], "recommended_papers": [], "uncertainty_notes": []}
    out["_latency_sec"] = round(latency, 3)
    out["_raw"] = raw
    return out


# ── V1: full system ──────────────────────────────────────────────────────────

def run_v1(query: str) -> dict:
    t0 = time.time()
    result = _run_pipeline(query)
    latency = time.time() - t0
    answer = result["answer"]

    top_paper_ids_local = [p.local_id for p in result["ranked_papers"][:10] if p.local_id]
    top_paper_ids_openalex = [p.openalex_id for p in result["ranked_papers"][:10] if p.openalex_id]
    top_dataset_ids = [d.dataset_id for d in result["ranked_datasets"][:10]]
    top_chunk_paper_ids = [c.local_id for c in result["chunk_candidates"][:10] if c.local_id]

    return {
        "intent_type": result.get("intent_type"),
        "answer_mode": answer.answer_mode,
        "direct_answer": answer.direct_answer,
        "recommended_datasets": [d.model_dump() for d in answer.recommended_datasets],
        "recommended_papers": [p.model_dump() for p in answer.recommended_papers],
        "methodology_hints": [h.model_dump() for h in answer.methodology_hints],
        "uncertainty_notes": answer.uncertainty_notes,
        "grounding_report": answer.grounding_report.model_dump() if answer.grounding_report else None,
        "top_paper_local_ids": top_paper_ids_local,
        "top_paper_openalex_ids": top_paper_ids_openalex,
        "top_dataset_ids": top_dataset_ids,
        "top_chunk_paper_ids": top_chunk_paper_ids,
        "cache_dir": result["cache_dir"],
        "_latency_sec": round(latency, 3),
    }


# ── Per-sample metric computation ────────────────────────────────────────────

def score_v1_sample(sample: dict, v1: dict) -> dict:
    gold_papers = sample.get("gold_paper_local_ids", [])
    gold_datasets = sample.get("gold_dataset_ids", [])

    # ── Paper retrieval ──────────────────────────────────────────────────────
    # Recall: union of top-k from the main channel and chunk channel independently.
    # The two channels rank items by different signals; appending one list after
    # the other and slicing at k would make chunk IDs invisible when the main
    # list already fills k. Instead, take the set union of top-k from each.
    top5_main  = set(v1["top_paper_local_ids"][:5])
    top10_main = set(v1["top_paper_local_ids"][:10])
    top5_chunk  = set(v1["top_chunk_paper_ids"][:5])
    top10_chunk = set(v1["top_chunk_paper_ids"][:10])
    pool5  = top5_main  | top5_chunk
    pool10 = top10_main | top10_chunk

    paper_recall_5  = recall_from_pool(pool5,  gold_papers)
    paper_recall_10 = recall_from_pool(pool10, gold_papers)

    # Precision and MRR operate on the main ranked list only — ordering matters.
    paper_precision_5 = precision_at_k(v1["top_paper_local_ids"], gold_papers, k=5)
    paper_f1_5        = f1_at_k(v1["top_paper_local_ids"], gold_papers, k=5)
    paper_mrr_val     = mrr(v1["top_paper_local_ids"], gold_papers)

    # ── Dataset retrieval ─────────────────────────────────────────────────────
    dataset_recall_5    = recall_at_k(v1["top_dataset_ids"], gold_datasets, k=5)
    dataset_recall_10   = recall_at_k(v1["top_dataset_ids"], gold_datasets, k=10)
    dataset_precision_5 = precision_at_k(v1["top_dataset_ids"], gold_datasets, k=5)
    dataset_f1_5        = f1_at_k(v1["top_dataset_ids"], gold_datasets, k=5)
    dataset_mrr_val     = mrr(v1["top_dataset_ids"], gold_datasets)

    # ── Grounding ─────────────────────────────────────────────────────────────
    grounding_rep = v1.get("grounding_report") or {}
    tags_total = grounding_rep.get("tags_total", 0)
    raw_grounding_rate = grounding_rep.get("grounding_rate")
    # Pipeline reports grounding_rate=1.0 when no tags are emitted (vacuously
    # true — the model didn't hallucinate citations it never made). In eval,
    # convert to None (N/A) so these cases don't inflate category averages.
    grounding_rate = None if tags_total == 0 else raw_grounding_rate
    grounded_ok = grounding_rep.get("grounded_ok")

    # ── Citation coverage ──────────────────────────────────────────────────────
    all_emitted = collect_emitted_tags({
        "direct_answer": v1.get("direct_answer"),
        "recommended_datasets": v1.get("recommended_datasets"),
        "recommended_papers": v1.get("recommended_papers"),
        "methodology_hints": v1.get("methodology_hints"),
    })
    cov = citation_coverage(all_emitted)

    # ── Methodology fidelity ───────────────────────────────────────────────────
    meth_rate = methodology_cites_chunks_rate(v1["methodology_hints"])

    # ── Abstention (oos category only) ────────────────────────────────────────
    abst = None
    if sample.get("category") == "oos":
        abst = is_abstention_correct(
            v1["recommended_datasets"],
            v1["recommended_papers"],
            v1["uncertainty_notes"],
        )

    # ── Answer quality (direct_answer + reference available) ──────────────────
    rouge_l = None
    if sample.get("category") == "direct_answer" and sample.get("gold_reference_answer"):
        rouge_l = rouge_l_score(
            v1.get("direct_answer") or "",
            sample["gold_reference_answer"],
        )

    return {
        "paper_recall@5":            paper_recall_5,
        "paper_recall@10":           paper_recall_10,
        "paper_precision@5":         paper_precision_5,
        "paper_f1@5":                paper_f1_5,
        "paper_mrr":                 paper_mrr_val,
        "dataset_recall@5":          dataset_recall_5,
        "dataset_recall@10":         dataset_recall_10,
        "dataset_precision@5":       dataset_precision_5,
        "dataset_f1@5":              dataset_f1_5,
        "dataset_mrr":               dataset_mrr_val,
        "grounding_rate":            grounding_rate,
        "grounded_ok":               grounded_ok,
        "has_any_citation":          cov["has_any_citation"],
        "unique_sources_cited":      cov["unique_sources_cited"],
        "methodology_cites_chunks_rate": meth_rate,
        "abstention_correct":        abst,
        "rouge_l":                   rouge_l,
        "latency_sec":               v1["_latency_sec"],
    }


def score_v0_sample(sample: dict, v0: dict) -> dict:
    """V0 has no retrieval so retrieval metrics are N/A.
    Grounding is N/A for V0 (no evidence block exists) — previously this was
    set to 0.0 when recommendations existed and None otherwise, which created
    an asymmetry vs V1 (where no-tag cases became 1.0). Both are now None.
    """
    abst = None
    if sample.get("category") == "oos":
        abst = is_abstention_correct(
            v0.get("recommended_datasets") or [],
            v0.get("recommended_papers") or [],
            v0.get("uncertainty_notes") or [],
        )

    num_fabricated_datasets = len(v0.get("recommended_datasets") or [])
    num_fabricated_papers = len(v0.get("recommended_papers") or [])

    rouge_l = None
    if sample.get("category") == "direct_answer" and sample.get("gold_reference_answer"):
        rouge_l = rouge_l_score(
            v0.get("direct_answer") or "",
            sample["gold_reference_answer"],
        )

    return {
        "paper_recall@5":            None,
        "paper_recall@10":           None,
        "paper_precision@5":         None,
        "paper_f1@5":                None,
        "paper_mrr":                 None,
        "dataset_recall@5":          None,
        "dataset_recall@10":         None,
        "dataset_precision@5":       None,
        "dataset_f1@5":              None,
        "dataset_mrr":               None,
        "grounding_rate":            None,
        "grounded_ok":               None,
        "has_any_citation":          None,
        "unique_sources_cited":      None,
        "methodology_cites_chunks_rate": None,
        "abstention_correct":        abst,
        "rouge_l":                   rouge_l,
        "latency_sec":               v0.get("_latency_sec"),
        "num_fabricated_datasets":   num_fabricated_datasets,
        "num_fabricated_papers":     num_fabricated_papers,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(per_sample: list[dict], variant: str) -> list[dict]:
    """Return aggregate rows grouped by category (plus overall)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in per_sample:
        groups[row["category"]].append(row["metrics"])
        groups["_all"].append(row["metrics"])

    out = []
    for cat, rows in groups.items():
        agg = {"variant": variant, "category": cat, "n": len(rows)}

        numeric_fields = [
            "paper_recall@5", "paper_recall@10", "paper_precision@5", "paper_f1@5", "paper_mrr",
            "dataset_recall@5", "dataset_recall@10", "dataset_precision@5", "dataset_f1@5", "dataset_mrr",
            "grounding_rate", "methodology_cites_chunks_rate",
            "unique_sources_cited", "rouge_l", "latency_sec",
        ]
        for f in numeric_fields:
            vals = [r.get(f) for r in rows if r.get(f) is not None]
            agg[f] = round(sum(vals) / len(vals), 4) if vals else None

        bool_fields = ["grounded_ok", "abstention_correct", "has_any_citation"]
        for f in bool_fields:
            vals = [r.get(f) for r in rows if r.get(f) is not None]
            if vals:
                agg[f + "_rate"] = round(sum(1 for v in vals if v) / len(vals), 4)
            else:
                agg[f + "_rate"] = None

        # Latency P95 (useful for tail-latency profiling of recommendation_hard)
        latencies = sorted(r["latency_sec"] for r in rows if r.get("latency_sec") is not None)
        agg["latency_p95"] = round(latencies[max(0, int(0.95 * len(latencies)) - 1)], 3) if latencies else None

        out.append(agg)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main(limit: int | None = None, skip_v0: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_FILE) as f:
        eval_set = json.load(f)

    samples = eval_set["samples"]
    if limit:
        samples = samples[:limit]

    cfg = get_settings()
    client = OpenAI(api_key=openai_api_key())
    model = cfg["llm"]["default_model"]

    per_sample_v0: list[dict] = []
    per_sample_v1: list[dict] = []

    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {sample['sample_id']} ({sample['category']})")
        print(f"  Q: {sample['query'][:100]}")

        # V0 — no-RAG baseline
        if not skip_v0:
            try:
                print(f"  Running V0 (no-RAG)...")
                v0 = run_v0(sample["query"], client, model)
                v0_metrics = score_v0_sample(sample, v0)
                per_sample_v0.append({
                    "sample_id": sample["sample_id"],
                    "category": sample["category"],
                    "answer_mode": sample["answer_mode"],
                    "query": sample["query"],
                    "output": {k: v for k, v in v0.items() if not k.startswith("_")},
                    "metrics": v0_metrics,
                })
                print(f"  V0 latency={v0['_latency_sec']}s")
            except Exception as e:
                print(f"  V0 FAILED: {e}")
                traceback.print_exc()
                per_sample_v0.append({
                    "sample_id": sample["sample_id"],
                    "category": sample["category"],
                    "answer_mode": sample["answer_mode"],
                    "query": sample["query"],
                    "error": str(e),
                })

        # V1 — full system
        try:
            print(f"  Running V1 (full RAG)...")
            v1 = run_v1(sample["query"])
            v1_metrics = score_v1_sample(sample, v1)
            per_sample_v1.append({
                "sample_id": sample["sample_id"],
                "category": sample["category"],
                "answer_mode": sample["answer_mode"],
                "query": sample["query"],
                "output": v1,
                "metrics": v1_metrics,
            })
            lat = v1["_latency_sec"]
            gr = v1_metrics.get("grounding_rate")
            cit = v1_metrics.get("unique_sources_cited")
            print(f"  V1 latency={lat}s, grounding={gr}, sources_cited={cit}")
        except Exception as e:
            print(f"  V1 FAILED: {e}")
            traceback.print_exc()
            per_sample_v1.append({
                "sample_id": sample["sample_id"],
                "category": sample["category"],
                "answer_mode": sample["answer_mode"],
                "query": sample["query"],
                "error": str(e),
            })

    # ── Write per-sample files ────────────────────────────────────────────────
    with open(RESULTS_DIR / "per_sample_v0.jsonl", "w") as f:
        for row in per_sample_v0:
            f.write(json.dumps(row, default=str) + "\n")
    with open(RESULTS_DIR / "per_sample_v1.jsonl", "w") as f:
        for row in per_sample_v1:
            f.write(json.dumps(row, default=str) + "\n")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    v0_agg = aggregate([r for r in per_sample_v0 if "metrics" in r], "V0")
    v1_agg = aggregate([r for r in per_sample_v1 if "metrics" in r], "V1")
    all_agg = v0_agg + v1_agg

    if all_agg:
        fieldnames = sorted({k for r in all_agg for k in r.keys()})
        for pri in ["n", "category", "variant"]:
            if pri in fieldnames:
                fieldnames.remove(pri)
                fieldnames.insert(0, pri)
        with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in all_agg:
                w.writerow(row)

    # ── Markdown comparison table ─────────────────────────────────────────────
    _write_markdown_summary(v0_agg, v1_agg, RESULTS_DIR / "comparison_table.md")

    print(f"\n✓ Done. Results in {RESULTS_DIR}/")
    print(f"   - per_sample_v0.jsonl ({len(per_sample_v0)} rows)")
    print(f"   - per_sample_v1.jsonl ({len(per_sample_v1)} rows)")
    print(f"   - summary.csv ({len(all_agg)} aggregate rows)")
    print(f"   - comparison_table.md")


def _write_markdown_summary(v0_agg: list[dict], v1_agg: list[dict], out_path: Path):
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = [
        "# Evaluation Results — V0 (no-RAG) vs V1 (full system)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Retrieval metrics",
        "",
        "| Variant | Category | n | Paper R@5 | Paper P@5 | Paper F1@5 | Paper MRR | DS R@5 | DS P@5 | DS F1@5 | DS MRR |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in v0_agg + v1_agg:
        lines.append(
            f"| {row['variant']} | {row['category']} | {row['n']} | "
            f"{fmt(row.get('paper_recall@5'))} | {fmt(row.get('paper_precision@5'))} | "
            f"{fmt(row.get('paper_f1@5'))} | {fmt(row.get('paper_mrr'))} | "
            f"{fmt(row.get('dataset_recall@5'))} | {fmt(row.get('dataset_precision@5'))} | "
            f"{fmt(row.get('dataset_f1@5'))} | {fmt(row.get('dataset_mrr'))} |"
        )

    lines += [
        "",
        "## Grounding & citation metrics",
        "",
        "| Variant | Category | n | Grounding Rate | Cited Anything | Unique Sources | Meth Chunks | ROUGE-L |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in v0_agg + v1_agg:
        lines.append(
            f"| {row['variant']} | {row['category']} | {row['n']} | "
            f"{fmt(row.get('grounding_rate'))} | {fmt(row.get('has_any_citation_rate'))} | "
            f"{fmt(row.get('unique_sources_cited'))} | {fmt(row.get('methodology_cites_chunks_rate'))} | "
            f"{fmt(row.get('rouge_l'))} |"
        )

    lines += [
        "",
        "## Abstention & latency",
        "",
        "| Variant | Category | n | Abstention Rate | Latency mean (s) | Latency P95 (s) |",
        "|---|---|---|---|---|---|",
    ]
    for row in v0_agg + v1_agg:
        lines.append(
            f"| {row['variant']} | {row['category']} | {row['n']} | "
            f"{fmt(row.get('abstention_correct_rate'))} | "
            f"{fmt(row.get('latency_sec'))} | {fmt(row.get('latency_p95'))} |"
        )

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only first N samples")
    parser.add_argument("--skip-v0", action="store_true", help="Skip V0 baseline")
    args = parser.parse_args()
    main(limit=args.limit, skip_v0=args.skip_v0)
