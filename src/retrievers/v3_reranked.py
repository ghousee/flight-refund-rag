"""v3-reranked retriever: wide vector retrieval + cross-encoder reranking.

The bi-encoder (bge-small) that powers v1/v2 encodes the question and each chunk
*separately*, so it can rank the truly-best chunk too low. A cross-encoder
(bge-reranker-v2-m3) scores the question and a chunk *together*, which is far
more accurate at judging relevance — but too slow to run over the whole corpus.

So v3 uses the standard retrieve-then-rerank pattern:

    vector search (cheap)  ->  top FETCH_K candidates
    cross-encoder (accurate) -> re-score those candidates -> keep top k

This targets the cases where the right chunk *is* retrieved but ranked below the
top 5 (our multi-hop and casual-vs-legalese weak spots).

Reranking over the *vector* base (not v2-hybrid), since hybrid measured worse.

Usage:
    python -m src.retrievers.v3_reranked "Are WestJet UltraBasic fares refundable?"
    python -m src.retrievers.v3_reranked --chat
"""

from __future__ import annotations

import argparse

from sentence_transformers import CrossEncoder

from src import config
from src.retrievers.v1_naive import (
    PROMPT,
    format_context,
    get_llm,
    get_store,
    print_sources,
)

FETCH_K = 20  # candidates pulled by vector search before reranking


def get_reranking_retriever(fetch_k: int = FETCH_K, default_k: int = 5):
    """Build vector store + cross-encoder once; return retrieve(question, k)."""
    store = get_store()
    reranker = CrossEncoder(config.RERANKER_MODEL)

    def retrieve(question: str, k: int = default_k):
        candidates = store.similarity_search(question, k=fetch_k)
        if not candidates:
            return []
        scores = reranker.predict([(question, d.page_content) for d in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda sc: sc[0], reverse=True)
        return [doc for _, doc in ranked][:k]

    return retrieve


def answer(question: str, k: int = 5) -> None:
    retrieve = get_reranking_retriever(default_k=k)
    docs = retrieve(question, k)
    response = (PROMPT | get_llm()).invoke(
        {"context": format_context(docs), "question": question}
    )
    print(f"\n{response.content.strip()}")
    print_sources(docs)


def chat(k: int = 5) -> None:
    retrieve = get_reranking_retriever(default_k=k)
    llm = get_llm()
    print(
        "flight-refund-rag - v3-reranked chat (vector retrieve-20 -> rerank-5)\n"
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
    parser.add_argument("--k", type=int, default=5, help="chunks to keep (default 5)")
    args = parser.parse_args()
    if args.chat:
        chat(k=args.k)
    elif args.question:
        answer(" ".join(args.question), k=args.k)
    else:
        parser.error("provide a question, or use --chat for interactive mode")


if __name__ == "__main__":
    main()
