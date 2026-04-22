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
from app.config import get_settings, openai_api_key
from app.main import _run_pipeline
from evaluation.metrics import (
    recall_at_k, mrr, parse_citation_tags, compute_grounding_rate,
    collect_emitted_tags, is_abstention_correct, methodology_cites_chunks_rate,
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

    # IDs of top-10 retrieved for recall metrics
    top_paper_ids_local = [p.local_id for p in result["ranked_papers"][:10] if p.local_id]
    top_paper_ids_openalex = [p.openalex_id for p in result["ranked_papers"][:10]]
    top_dataset_ids = [d.dataset_id for d in result["ranked_datasets"][:10]]
    top_chunk_paper_ids = [c.local_id for c in result["chunk_candidates"][:10] if c.local_id]

    return {
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

    # Retrieval recall — papers (by local_id)
    retrieved_papers_in_order = v1["top_paper_local_ids"]
    # include chunk-paper ids as additional retrieval channel (fulltext evidence)
    retrieved_papers_union = retrieved_papers_in_order + [
        lid for lid in v1["top_chunk_paper_ids"] if lid not in retrieved_papers_in_order
    ]
    paper_recall_5 = recall_at_k(retrieved_papers_union, gold_papers, k=5)
    paper_recall_10 = recall_at_k(retrieved_papers_union, gold_papers, k=10)
    paper_mrr = mrr(retrieved_papers_union, gold_papers)

    # Retrieval recall — datasets
    dataset_recall_5 = recall_at_k(v1["top_dataset_ids"], gold_datasets, k=5)
    dataset_recall_10 = recall_at_k(v1["top_dataset_ids"], gold_datasets, k=10)
    dataset_mrr = mrr(v1["top_dataset_ids"], gold_datasets)

    # Grounding: report came from the pipeline; re-compute as well from emitted tags
    grounding_rep = v1.get("grounding_report") or {}
    grounding_rate = grounding_rep.get("grounding_rate")
    grounded_ok = grounding_rep.get("grounded_ok")

    # Methodology fidelity
    meth_rate = methodology_cites_chunks_rate(v1["methodology_hints"])

    # Abstention (only for oos category)
    abst = None
    if sample.get("category") == "oos":
        abst = is_abstention_correct(
            v1["recommended_datasets"],
            v1["recommended_papers"],
            v1["uncertainty_notes"],
        )

    return {
        "paper_recall@5": paper_recall_5,
        "paper_recall@10": paper_recall_10,
        "paper_mrr": paper_mrr,
        "dataset_recall@5": dataset_recall_5,
        "dataset_recall@10": dataset_recall_10,
        "dataset_mrr": dataset_mrr,
        "grounding_rate": grounding_rate,
        "grounded_ok": grounded_ok,
        "methodology_cites_chunks_rate": meth_rate,
        "abstention_correct": abst,
        "latency_sec": v1["_latency_sec"],
    }


def score_v0_sample(sample: dict, v0: dict) -> dict:
    """V0 has no retrieval so most retrieval metrics are N/A. Grounding = 0 (no evidence to cite)."""
    abst = None
    if sample.get("category") == "oos":
        abst = is_abstention_correct(
            v0.get("recommended_datasets") or [],
            v0.get("recommended_papers") or [],
            v0.get("uncertainty_notes") or [],
        )

    # V0 has no evidence block, so any citation it produces is fabricated.
    # Since the baseline prompt doesn't ask for citations, most will be empty.
    # Count any recommendation at all — these are fabricated datasets / papers.
    num_fabricated_datasets = len(v0.get("recommended_datasets") or [])
    num_fabricated_papers = len(v0.get("recommended_papers") or [])

    return {
        "paper_recall@5": None,
        "paper_recall@10": None,
        "paper_mrr": None,
        "dataset_recall@5": None,
        "dataset_recall@10": None,
        "dataset_mrr": None,
        "grounding_rate": 0.0 if (num_fabricated_datasets or num_fabricated_papers) else None,
        "grounded_ok": False if (num_fabricated_datasets or num_fabricated_papers) else None,
        "methodology_cites_chunks_rate": None,
        "abstention_correct": abst,
        "latency_sec": v0.get("_latency_sec"),
        "num_fabricated_datasets": num_fabricated_datasets,
        "num_fabricated_papers": num_fabricated_papers,
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
            "paper_recall@5", "paper_recall@10", "paper_mrr",
            "dataset_recall@5", "dataset_recall@10", "dataset_mrr",
            "grounding_rate", "methodology_cites_chunks_rate", "latency_sec",
        ]
        for f in numeric_fields:
            vals = [r.get(f) for r in rows if r.get(f) is not None]
            agg[f] = round(sum(vals) / len(vals), 4) if vals else None

        bool_fields = ["grounded_ok", "abstention_correct"]
        for f in bool_fields:
            vals = [r.get(f) for r in rows if r.get(f) is not None]
            if vals:
                agg[f + "_rate"] = round(sum(1 for v in vals if v) / len(vals), 4)
            else:
                agg[f + "_rate"] = None

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
            print(f"  V1 latency={lat}s, grounding={gr}")
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
        # put variant, category, n first
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
    lines = [
        "# Evaluation Results — V0 (no-RAG) vs V1 (full system)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Aggregate metrics by category",
        "",
        "| Variant | Category | n | Paper Recall@5 | Dataset Recall@5 | Paper MRR | Dataset MRR | Grounding Rate | Methodology Cites Chunks | Abstention Rate | Latency (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    for row in v0_agg + v1_agg:
        lines.append(
            f"| {row['variant']} | {row['category']} | {row['n']} | "
            f"{fmt(row.get('paper_recall@5'))} | {fmt(row.get('dataset_recall@5'))} | "
            f"{fmt(row.get('paper_mrr'))} | {fmt(row.get('dataset_mrr'))} | "
            f"{fmt(row.get('grounding_rate'))} | {fmt(row.get('methodology_cites_chunks_rate'))} | "
            f"{fmt(row.get('abstention_correct_rate'))} | {fmt(row.get('latency_sec'))} |"
        )

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only first N samples")
    parser.add_argument("--skip-v0", action="store_true", help="Skip V0 baseline")
    args = parser.parse_args()
    main(limit=args.limit, skip_v0=args.skip_v0)
