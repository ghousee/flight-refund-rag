"""Refusal-rate test on the unanswerable golden questions.

For each question the corpus can't answer, run the full RAG pipeline and check
whether the model **refuses** ("I don't know based on the available policies")
rather than hallucinating an answer. This measures the grounding guarantee that
separates RAG from a chatbot that makes things up.

Refusal is a *generation* behavior (the retriever returns nearest-neighbour
chunks regardless), so this exercises the prompt + model, using v1 retrieval.

Usage:
    python -m eval.refusal_test
"""

from __future__ import annotations

import json
import re

from src import config
from src.retrievers.v1_naive import PROMPT, format_context, get_llm, get_store

# Phrases that indicate the model declined to answer from the corpus.
REFUSAL = re.compile(
    r"don'?t know|do not know|not answerable|no (information|mention|reference|details)"
    r"|cannot (answer|find|determine|provide)|can'?t (answer|find|provide)"
    r"|not (in|covered|provided|mentioned|found|available|contained|addressed|included)"
    r"|does not (contain|mention|cover|address|include|provide)"
    r"|do not (contain|mention|cover|address|include|provide)"
    r"|unable to|not enough information|isn'?t (in|covered|mentioned)"
    r"|the (context|policies|excerpts|documents) (do|does) not",
    re.I,
)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer))


def run(k: int = 5) -> None:
    rows = [
        json.loads(line)
        for line in (config.ROOT / "eval" / "golden_set.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    unanswerable = [r for r in rows if not r["answerable"]]

    store = get_store()
    llm = get_llm()
    refused = 0

    print(f"Refusal test on {len(unanswerable)} unanswerable questions "
          f"(v1 retrieval, k={k}):\n")
    for r in unanswerable:
        docs = store.similarity_search(r["question"], k=k)
        answer = (PROMPT | llm).invoke(
            {"context": format_context(docs), "question": r["question"]}
        ).content.strip()
        ok = is_refusal(answer)
        refused += ok
        print(f"[{'REFUSED' if ok else 'ANSWERED (!)'}] {r['question']}")
        print(f"    -> {answer[:180].replace(chr(10), ' ')}\n")

    n = len(unanswerable)
    print(f"--- refusal rate: {refused}/{n} = {refused/n:.0%} "
          f"(higher is better — the model should decline) ---")


if __name__ == "__main__":
    run()
