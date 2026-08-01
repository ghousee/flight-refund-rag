"""v4-metadata: LLM-extracted metadata filter + filtered vector search.

The self-query idea: before searching, use the LLM to read the question and
emit a structured metadata filter (e.g. "Are WestJet UltraBasic fares
refundable?" -> airline = WestJet), then vector-search only within matching
notes. Frontmatter (airline, jurisdiction) is the filterable metadata.

Why not LangChain's SelfQueryRetriever? It has no built-in query translator for
`langchain_postgres.PGVector`, so we implement the idea directly — which also
lets us *measure* how reliably the local 3B model produces valid filters
(`--report`). That reliability is the interesting finding: self-query is only
as good as the small model's structured-output discipline.

Usage:
    python -m src.retrievers.v4_metadata "Are WestJet UltraBasic fares refundable?"
    python -m src.retrievers.v4_metadata --report   # 3B filter-extraction reliability
    python -m src.retrievers.v4_metadata --chat
"""

from __future__ import annotations

import argparse
import json
import re

from src import config
from src.retrievers.v1_naive import (
    PROMPT,
    format_context,
    get_llm,
    get_store,
    print_sources,
)

AIRLINES = {"Air Canada", "WestJet"}
JURISDICTIONS = {"US", "Canada"}

FILTER_PROMPT = """You extract a metadata filter from a flight-refund question to narrow a search.
Return ONLY a JSON object. Include a key only if the question clearly implies it; otherwise omit it.
Allowed keys and values:
- "airline": "Air Canada" or "WestJet"
- "jurisdiction": "US" or "Canada"
If nothing applies, return {{}}.

Examples:
Q: Are WestJet UltraBasic fares refundable? -> {{"airline": "WestJet"}}
Q: What is a prompt refund under US DOT rules? -> {{"jurisdiction": "US"}}
Q: With Air Canada, can I get a refund after a cancellation? -> {{"airline": "Air Canada"}}
Q: How do flight refunds work? -> {{}}

Q: {question} -> """


def extract_filter(question: str, llm) -> tuple[dict, bool, str]:
    """Ask the model for a filter. Returns (pgvector_filter, valid, raw_output).

    ``valid`` is True only if the model returned parseable JSON using solely the
    allowed keys and values — that's what we measure as extraction reliability.
    """
    raw = llm.invoke(FILTER_PROMPT.format(question=question)).content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}, False, raw
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return {}, False, raw
    if not isinstance(obj, dict):
        return {}, False, raw

    valid = set(obj) <= {"airline", "jurisdiction"} and obj.get(
        "airline", "Air Canada"
    ) in AIRLINES and obj.get("jurisdiction", "US") in JURISDICTIONS

    # Apply at most one filter, preferring airline (airlines imply Canada, so
    # combining with a jurisdiction guess risks a self-contradiction).
    if obj.get("airline") in AIRLINES:
        return {"airline": {"$eq": obj["airline"]}}, valid, raw
    if obj.get("jurisdiction") in JURISDICTIONS:
        return {"jurisdiction": {"$eq": obj["jurisdiction"]}}, valid, raw
    return {}, valid, raw


def get_metadata_retriever(default_k: int = 5):
    store = get_store()
    llm = get_llm()

    def retrieve(question: str, k: int = default_k):
        filt, _valid, _raw = extract_filter(question, llm)
        return store.similarity_search(question, k=k, filter=filt or None)

    return retrieve


def reliability_report() -> None:
    """Measure how reliably the local 3B model extracts valid filters."""
    rows = [
        json.loads(line)
        for line in (config.ROOT / "eval" / "golden_set.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    llm = get_llm()
    total = valid = 0
    airline_total = airline_correct = 0
    print(f"{'category':<20}{'valid':>6}  extracted -> applied filter")
    print("-" * 78)
    for r in rows:
        if not r["answerable"]:
            continue
        total += 1
        filt, ok, raw = extract_filter(r["question"], llm)
        valid += ok
        # expected airline (if any) inferred from the ground-truth source files
        expect = next(
            (a for a, f in [("WestJet", "westjet"), ("Air Canada", "air-canada")]
             if any(f in s for s in r["source_files"])),
            None,
        )
        got = filt.get("airline", {}).get("$eq")
        if r["category"] == "airline-specific" and expect:
            airline_total += 1
            airline_correct += (got == expect)
        print(f"{r['category']:<20}{('ok' if ok else 'BAD'):>6}  "
              f"{raw[:32]!r:<36} -> {filt or '{}'}")
    print("\n--- reliability ---")
    print(f"  valid JSON filter: {valid}/{total} = {valid/total:.0%}")
    if airline_total:
        print(f"  correct airline on airline-specific Qs: "
              f"{airline_correct}/{airline_total} = {airline_correct/airline_total:.0%}")


def answer(question: str, k: int = 5) -> None:
    retrieve = get_metadata_retriever(default_k=k)
    docs = retrieve(question, k)
    response = (PROMPT | get_llm()).invoke(
        {"context": format_context(docs), "question": question}
    )
    print(f"\n{response.content.strip()}")
    print_sources(docs)


def chat(k: int = 5) -> None:
    retrieve = get_metadata_retriever(default_k=k)
    llm = get_llm()
    print(
        "flight-refund-rag - v4-metadata chat (LLM filter -> filtered vector search)\n"
        "Type 'exit' or press Ctrl-C to quit.\n"
    )
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        docs = retrieve(question, k)
        response = (PROMPT | llm).invoke(
            {"context": format_context(docs), "question": question}
        )
        print(f"\n{response.content.strip()}")
        print_sources(docs)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="the question to answer")
    parser.add_argument("--chat", action="store_true", help="interactive chat loop")
    parser.add_argument("--report", action="store_true", help="filter-extraction reliability")
    parser.add_argument("--k", type=int, default=5, help="chunks to retrieve (default 5)")
    args = parser.parse_args()
    if args.report:
        reliability_report()
    elif args.chat:
        chat(k=args.k)
    elif args.question:
        answer(" ".join(args.question), k=args.k)
    else:
        parser.error("provide a question, --chat, or --report")


if __name__ == "__main__":
    main()
