"""Retrieval metrics — recall@k and MRR against the golden set.

Ground truth: each golden question lists ``source_files`` (the vault note(s) that
contain the answer). A retriever returns ranked chunks, each carrying its note
name in ``metadata["source"]``. We score how well the right notes surface:

  * recall@k -- fraction of a question's ground-truth notes present in the top-k
    retrieved chunks (averaged over questions). "Is the answer even in context?"
  * MRR      -- reciprocal rank of the FIRST relevant chunk (1/rank), averaged.
    "How highly ranked was the right note?"

Unanswerable questions (no source_files) are excluded — refusal is measured
separately with RAGAS. The harness is retriever-agnostic: pass any
``retriever_fn(question, k) -> list[Document]`` so v1..v4 run identically.

Usage:
    python -m eval.retrieval_metrics            # evaluate v1-naive
    python -m eval.retrieval_metrics --k 5
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from statistics import mean

from src import config

GOLDEN_PATH = config.ROOT / "eval" / "golden_set.jsonl"


def load_golden(answerable_only: bool = True) -> list[dict]:
    rows = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["answerable"]] if answerable_only else rows


def evaluate(retriever_fn, k: int = 5) -> dict:
    """Run every answerable golden question through retriever_fn and score it."""
    rows = load_golden()
    per_query = []
    for r in rows:
        truth = set(r["source_files"])
        t0 = time.perf_counter()
        docs = retriever_fn(r["question"], k)
        latency = time.perf_counter() - t0

        ranked_notes = [d.metadata.get("source") for d in docs][:k]

        # recall@k: fraction of ground-truth notes present in the top-k
        recall = len(truth & set(ranked_notes)) / len(truth) if truth else 0.0
        # reciprocal rank: 1/rank of the first relevant note (0 if none)
        rr = next(
            (1.0 / i for i, note in enumerate(ranked_notes, 1) if note in truth),
            0.0,
        )
        per_query.append(
            {
                "category": r["category"],
                "question": r["question"],
                "recall": recall,
                "rr": rr,
                "latency": latency,
                "hit": recall > 0,
            }
        )
    return {"k": k, "per_query": per_query}


def _summary(per_query: list[dict], k: int) -> dict:
    return {
        f"recall@{k}": mean(q["recall"] for q in per_query),
        "mrr": mean(q["rr"] for q in per_query),
        "hit_rate": mean(q["hit"] for q in per_query),
        "avg_latency_s": mean(q["latency"] for q in per_query),
        "n": len(per_query),
    }


def print_report(name: str, result: dict) -> None:
    pq, k = result["per_query"], result["k"]

    print(f"\n=== {name} — retrieval metrics (k={k}) ===\n")
    print(f"{'category':<20}{'recall':>8}{'rr':>8}{'lat(s)':>9}  question")
    print("-" * 90)
    for q in pq:
        print(
            f"{q['category']:<20}{q['recall']:>8.2f}{q['rr']:>8.2f}"
            f"{q['latency']:>9.3f}  {q['question'][:44]}"
        )

    print("\n--- per category ---")
    by_cat: dict[str, list] = defaultdict(list)
    for q in pq:
        by_cat[q["category"]].append(q)
    for cat, qs in sorted(by_cat.items()):
        print(
            f"  {cat:<20} recall@{k}={mean(x['recall'] for x in qs):.2f}  "
            f"mrr={mean(x['rr'] for x in qs):.2f}  (n={len(qs)})"
        )

    s = _summary(pq, k)
    print("\n--- overall ---")
    print(
        f"  recall@{k}={s[f'recall@{k}']:.3f}  mrr={s['mrr']:.3f}  "
        f"hit_rate={s['hit_rate']:.3f}  avg_latency={s['avg_latency_s']:.3f}s  "
        f"(n={s['n']} answerable questions)"
    )


def v1_retriever_fn():
    """v1-naive: plain cosine similarity search over pgvector."""
    from src.retrievers.v1_naive import get_store

    store = get_store()
    return lambda question, k: store.similarity_search(question, k=k)


def v2_retriever_fn():
    """v2-hybrid: BM25 + vector fused with RRF."""
    from src.retrievers.v2_hybrid import get_hybrid_retriever

    return get_hybrid_retriever()


def v3_retriever_fn():
    """v3-reranked: vector retrieve-20 -> cross-encoder rerank-5."""
    from src.retrievers.v3_reranked import get_reranking_retriever

    return get_reranking_retriever()


def v4_retriever_fn():
    """v4-metadata: LLM-extracted metadata filter -> filtered vector search."""
    from src.retrievers.v4_metadata import get_metadata_retriever

    return get_metadata_retriever()


# Registry so later versions plug in with no harness changes.
RETRIEVERS = {
    "v1-naive": v1_retriever_fn,
    "v2-hybrid": v2_retriever_fn,
    "v3-reranked": v3_retriever_fn,
    "v4-metadata": v4_retriever_fn,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", default="v1-naive", choices=RETRIEVERS)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    retriever_fn = RETRIEVERS[args.retriever]()
    result = evaluate(retriever_fn, k=args.k)
    print_report(args.retriever, result)


if __name__ == "__main__":
    main()
