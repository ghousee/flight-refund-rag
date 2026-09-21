"""Generation metrics — RAGAS faithfulness + refusal rate for the RAG answers.

Retrieval metrics (``eval/retrieval_metrics.py``) only ask "was the right note in
context?". This harness scores what the model *did* with that context:

  * faithfulness -- RAGAS. The judge decomposes the answer into atomic
    statements and checks each one against the retrieved chunks. 1.0 means every
    claim is supported by the context; anything lower means the model asserted
    something the policies did not say. This is the metric that catches "right
    retrieval, wrong synthesis".
  * refusal rate  -- on the unanswerable questions, did the model decline rather
    than invent an answer? (Shared with ``eval/refusal_test.py``.)

Judge model: the same local Ollama model used for generation, so the eval needs
no API key. Small models cannot reliably emit RAGAS's structured output when
asked politely, so ``SchemaGuidedLLM`` below constrains decoding to the JSON
schema RAGAS already states in its prompt. See that class for details.

Answers are cached in ``eval/results/`` so scoring can be re-run without paying
for generation again.

Usage:
    python -m eval.generation_metrics                # generate + score
    python -m eval.generation_metrics --regenerate   # ignore the answer cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import typing as t

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig

from eval.refusal_test import is_refusal
from eval.retrieval_metrics import load_golden
from src import config

RESULTS_DIR = config.ROOT / "eval" / "results"
SCHEMA_MARKER = "JSON Schema:\n"


class SchemaGuidedLLM(LangchainLLMWrapper):
    """RAGAS LLM wrapper that pins Ollama's decoding to the expected schema.

    RAGAS asks for structured output in prose ("return JSON complying with the
    following schema: {...}") and parses the reply with a pydantic parser.
    llama3.2:3b fails that two ways: unconstrained it answers with Python code
    that would produce the JSON, and with ``format="json"`` it echoes the schema
    back instead of an instance of it. Either way RAGAS's parser raises and the
    metric never scores.

    Ollama supports schema-constrained decoding: pass a JSON Schema as ``format``
    and the sampler can only emit a conforming instance. RAGAS already embeds
    that exact schema in its prompt, so we lift it back out and hand it to the
    model. Nothing about the judging prompt or the metric changes -- we only
    stop the small model from replying in the wrong shape.
    """

    def _schema_from_prompt(self, prompt) -> dict | None:
        text = prompt.to_string()
        start = text.find(SCHEMA_MARKER)
        if start == -1:
            return None
        try:
            schema, _ = json.JSONDecoder().raw_decode(text, start + len(SCHEMA_MARKER))
        except ValueError:
            return None
        return schema if isinstance(schema, dict) else None

    async def agenerate_text(self, prompt, *args, **kwargs):
        schema = self._schema_from_prompt(prompt)
        previous = getattr(self.langchain_llm, "format", None)
        if schema is not None:
            self.langchain_llm.format = schema
        try:
            return await super().agenerate_text(prompt, *args, **kwargs)
        finally:
            self.langchain_llm.format = previous


def build_judge(timeout: int) -> Faithfulness:
    from langchain_ollama import ChatOllama

    run_config = RunConfig(max_workers=1, timeout=timeout)
    judge = SchemaGuidedLLM(
        ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
        ),
        run_config=run_config,
    )
    return Faithfulness(llm=judge)


def generate_answers(k: int) -> list[dict]:
    """Run the full v1 pipeline over every golden question, answerable or not."""
    from src.retrievers.v1_naive import generate, get_llm, get_store

    store, llm = get_store(), get_llm()
    rows = load_golden(answerable_only=False)
    out = []
    for i, r in enumerate(rows, 1):
        t0 = time.perf_counter()
        answer, docs = generate(r["question"], store, llm, k)
        print(f"  [{i}/{len(rows)}] {time.perf_counter() - t0:5.1f}s  {r['question'][:58]}")
        out.append(
            {
                **r,
                "answer": answer,
                "contexts": [d.page_content for d in docs],
                "retrieved_notes": [d.metadata.get("source") for d in docs],
                "gen_latency_s": time.perf_counter() - t0,
            }
        )
    return out


def score_faithfulness(records: list[dict], metric: Faithfulness) -> None:
    """Attach a RAGAS faithfulness score to each answerable record, in place."""
    answerable = [r for r in records if r["answerable"]]
    for i, r in enumerate(answerable, 1):
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
        )
        t0 = time.perf_counter()
        try:
            r["faithfulness"] = float(asyncio.run(metric.single_turn_ascore(sample)))
            r["faithfulness_error"] = None
        except Exception as exc:  # judge failures are reported, never silently 0
            r["faithfulness"] = None
            r["faithfulness_error"] = f"{type(exc).__name__}: {exc}"[:200]
        shown = "FAILED" if r["faithfulness"] is None else f"{r['faithfulness']:.2f}"
        print(
            f"  [{i}/{len(answerable)}] faithfulness={shown:>6}  "
            f"({time.perf_counter() - t0:5.1f}s)  {r['question'][:46]}"
        )


def print_report(records: list[dict], k: int) -> None:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    scored = [r for r in answerable if r["faithfulness"] is not None]
    failed = [r for r in answerable if r["faithfulness"] is None]

    print(f"\n=== v1-naive — generation metrics (k={k}) ===\n")
    print(f"{'faith':>7}  {'category':<20} question")
    print("-" * 90)
    for r in sorted(answerable, key=lambda x: (x["faithfulness"] is None, x["faithfulness"] or 0)):
        shown = "FAILED" if r["faithfulness"] is None else f"{r['faithfulness']:.2f}"
        print(f"{shown:>7}  {r['category']:<20} {r['question'][:52]}")

    if scored:
        by_cat: dict[str, list[float]] = {}
        for r in scored:
            by_cat.setdefault(r["category"], []).append(r["faithfulness"])
        print("\n--- faithfulness per category ---")
        for cat, vals in sorted(by_cat.items()):
            print(f"  {cat:<20} {sum(vals) / len(vals):.3f}  (n={len(vals)})")

    refused = sum(is_refusal(r["answer"]) for r in unanswerable)
    print("\n--- overall ---")
    if scored:
        mean_f = sum(r["faithfulness"] for r in scored) / len(scored)
        print(f"  faithfulness = {mean_f:.3f}  (n={len(scored)} answerable questions scored)")
    if failed:
        print(f"  judge failed on {len(failed)}/{len(answerable)} questions (excluded from the mean)")
    print(f"  refusal rate = {refused}/{len(unanswerable)} = {refused / len(unanswerable):.0%}")
    gen = [r["gen_latency_s"] for r in records]
    print(f"  avg generation latency = {sum(gen) / len(gen):.1f}s/question  (local llama3.2:3b, CPU)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--regenerate", action="store_true", help="ignore the answer cache")
    parser.add_argument("--judge-timeout", type=int, default=600)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = RESULTS_DIR / f"v1_generations_k{args.k}.json"

    if cache.exists() and not args.regenerate:
        print(f"Reusing cached answers from {cache.name} (--regenerate to redo).")
        records = json.loads(cache.read_text(encoding="utf-8"))
    else:
        print("Generating answers for every golden question (local llama3.2:3b)...")
        records = generate_answers(args.k)
        cache.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print("\nScoring faithfulness with RAGAS (local judge, one question at a time)...")
    score_faithfulness(records, build_judge(args.judge_timeout))
    cache.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print_report(records, args.k)
    out = RESULTS_DIR / f"v1_generation_metrics_k{args.k}.json"
    out.write_text(
        json.dumps(
            [
                {
                    key: r[key]
                    for key in ("question", "category", "answerable", "answer",
                                "retrieved_notes", "source_files", "faithfulness",
                                "faithfulness_error")
                    if key in r
                }
                for r in records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nPer-question results written to {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
