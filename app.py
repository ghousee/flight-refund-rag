"""Minimal Gradio web chat for the flight-refund-rag v1-naive retriever.

A browser chat box wrapping the exact same retrieve -> ground -> answer pipeline
as the CLI, with streaming responses and clickable source citations.

    python app.py       # then open http://localhost:7860

Requires the Docker stack up (Ollama for generation) and the vault indexed.
"""

from __future__ import annotations

import gradio as gr

from src.retrievers.v1_naive import PROMPT, _label, format_context, get_llm, get_store

K = 5

# Load the vector store and model once at startup (not per message).
STORE = get_store()
LLM = get_llm()


def _sources_md(docs) -> str:
    """Render the retrieved chunks as a numbered markdown source list.

    Numbering matches the [n] markers the model cites in its answer.
    """
    lines = ["\n\n---\n**Sources**"]
    for i, d in enumerate(docs, 1):
        m = d.metadata
        label, url, juris = _label(m), m.get("source_url", ""), m.get("jurisdiction", "")
        lines.append(f"{i}. [{label}]({url}) ({juris})" if url else f"{i}. {label} ({juris})")
    return "\n".join(lines)


def respond(message: str, history):
    """Stream a grounded answer, then append the source list."""
    docs = STORE.similarity_search(message, k=K)
    context = format_context(docs)

    partial = ""
    for chunk in (PROMPT | LLM).stream({"context": context, "question": message}):
        partial += chunk.content or ""
        yield partial
    yield partial + _sources_md(docs)


demo = gr.ChatInterface(
    respond,
    title="✈️ flight-refund-rag — v1-naive",
    description=(
        "Ask about flight cancellation refunds. Answers are grounded in real "
        "airline & government policy (Air Canada, WestJet, APPR, US DOT) and cited. "
        "Baseline retriever — it may occasionally mis-synthesize; that's what the "
        "evaluation phase measures."
    ),
    examples=[
        "Can I get a cash refund if my flight is cancelled?",
        "How soon must a US airline pay a refund?",
        "Are WestJet UltraBasic fares refundable?",
        "If the airline offers a voucher instead of a refund, do I have to accept it?",
    ],
)

if __name__ == "__main__":
    demo.launch()
