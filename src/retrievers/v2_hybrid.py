"""v2-hybrid retriever: BM25 (lexical) + vector (semantic), fused with RRF.

v1 used vector similarity alone. v2 adds a BM25 keyword retriever and fuses the
two ranked lists with Reciprocal Rank Fusion via LangChain's EnsembleRetriever:

    score(chunk) = sum over retrievers of 1 / (c + rank_in_that_retriever)

Chunks that either retriever ranks highly rise to the top — vector catches
meaning, BM25 catches exact terms (rule names, citations, legalese).

The BM25 index is built in-memory from the same vault chunks that were embedded
into pgvector, so both retrievers see an identical corpus.

Usage:
    python -m src.retrievers.v2_hybrid "Are WestJet UltraBasic fares refundable?"
    python -m src.retrievers.v2_hybrid --chat
"""

from __future__ import annotations

import argparse

# In LangChain 1.x, EnsembleRetriever lives in the langchain_classic package.
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from src.ingest.index import chunk, load_notes
from src.retrievers.v1_naive import (
    PROMPT,
    format_context,
    get_llm,
    get_store,
    print_sources,
)


def get_hybrid_retriever(default_k: int = 5):
    """Build BM25 + vector once; return a retrieve(question, k) closure."""
    chunks = chunk(load_notes())
    bm25 = BM25Retriever.from_documents(chunks)
    vector = get_store().as_retriever(search_kwargs={"k": default_k})
    ensemble = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])

    def retrieve(question: str, k: int = default_k):
        bm25.k = k
        vector.search_kwargs["k"] = k
        # EnsembleRetriever returns the fused, de-duplicated ranking; take top-k.
        return ensemble.invoke(question)[:k]

    return retrieve


def answer(question: str, k: int = 5) -> None:
    retrieve = get_hybrid_retriever(default_k=k)
    docs = retrieve(question, k)
    response = (PROMPT | get_llm()).invoke(
        {"context": format_context(docs), "question": question}
    )
    print(f"\n{response.content.strip()}")
    print_sources(docs)


def chat(k: int = 5) -> None:
    retrieve = get_hybrid_retriever(default_k=k)
    llm = get_llm()
    print(
        "flight-refund-rag - v2-hybrid chat (BM25 + vector, RRF)\n"
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
    parser.add_argument("--k", type=int, default=5, help="chunks to retrieve (default 5)")
    args = parser.parse_args()
    if args.chat:
        chat(k=args.k)
    elif args.question:
        answer(" ".join(args.question), k=args.k)
    else:
        parser.error("provide a question, or use --chat for interactive mode")


if __name__ == "__main__":
    main()
