"""v1-naive retriever: fixed-size chunks + cosine similarity + grounded answer.

The baseline of the four retriever versions. The query flow:

    question -> embed -> pgvector cosine top-k -> grounded prompt -> LLM
             -> answer WITH source citations (or an honest "I don't know")

Generation runs on local Ollama (llama3.2:3b) by default so no API key is
needed. If ANTHROPIC_API_KEY is set, it upgrades to Claude.

Usage:
    python -m src.retrievers.v1_naive "Can I get a cash refund for a cancelled flight?"
    python -m src.retrievers.v1_naive --k 8 "What is a 'prompt refund'?"
"""

from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

from src import config
from src.ingest.index import get_embeddings

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful assistant answering questions about flight "
            "cancellation refunds, using ONLY the numbered policy excerpts "
            "provided.\n"
            "Rules:\n"
            "- Answer strictly from the context. If the answer is not in the "
            "context, say you don't know based on the available policies — do "
            "not use outside knowledge.\n"
            "- Cite the excerpts you rely on inline using their [n] markers.\n"
            "- Quote only short phrases; do not reproduce long passages.\n"
            "- Be concise and practical.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with [n] citations):",
        ),
    ]
)


def get_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        connection=config.DATABASE_URL,
        use_jsonb=True,
    )


def get_llm():
    """Ollama by default; Claude if an Anthropic key is present."""
    if config.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.ANTHROPIC_MODEL, temperature=0)
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0
    )


def _label(meta: dict) -> str:
    """Human-readable citation label for a chunk."""
    return meta.get("citation") or meta.get("title") or meta.get("source", "unknown")


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_url", "")
        blocks.append(f"[{i}] ({_label(d.metadata)} — {src})\n{d.page_content.strip()}")
    return "\n\n".join(blocks)


def print_sources(docs: list[Document]) -> None:
    print("\nSources:")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        juris = m.get("jurisdiction", "")
        print(f"  [{i}] {_label(m)} ({juris})")
        if m.get("source_url"):
            print(f"      {m['source_url']}")


def answer_once(question: str, store: PGVector, llm, k: int = 5) -> None:
    """Answer a single question against an already-loaded store + model."""
    docs = store.similarity_search(question, k=k)
    response = (PROMPT | llm).invoke(
        {"context": format_context(docs), "question": question}
    )
    print(f"\n{response.content.strip()}")
    print_sources(docs)


def answer(question: str, k: int = 5) -> None:
    """One-shot: load everything, answer once (used by the CLI)."""
    answer_once(question, get_store(), get_llm(), k)


def chat(k: int = 5) -> None:
    """Interactive REPL. Loads the store + model once, then loops on input."""
    store, llm = get_store(), get_llm()
    print(
        "flight-refund-rag - v1-naive chat\n"
        "Ask a flight-refund question. Answers are grounded in the policy corpus "
        "and cited.\nType 'exit' or press Ctrl-C to quit.\n"
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
        answer_once(question, store, llm, k)
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
