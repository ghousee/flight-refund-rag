# flight-refund-rag

A retrieval-augmented generation (RAG) system that answers customer questions
about **flight cancellation refunds**, grounded in real airline and regulatory
policy: Air Canada & WestJet contracts of carriage, Canada's Air Passenger
Protection Regulations (APPR), and US DOT refund rules.

Built as a learning project and portfolio piece — the interesting part isn't
"a chatbot," it's **measuring how retrieval quality changes the answers.** The
repo ships four progressively better retrievers and evaluates each against a
hand-authored golden set.

> **Runs with zero API keys.** Postgres/pgvector and the LLM (Ollama
> `llama3.2:3b`) both run locally via Docker. Clone → `docker compose up` →
> one ingest command → working system. Set `ANTHROPIC_API_KEY` to upgrade
> generation to Claude.

## Architecture

```
INGEST (offline):  PDFs ──► markdown vault ──► chunks ──► embeddings ──► pgvector
QUERY  (online):   question ──► embed ──► retrieve ──► LLM ──► grounded answer + citations
```

- **Embeddings:** local `BAAI/bge-small-en-v1.5` (deterministic, no key).
- **Vector store:** PostgreSQL + pgvector.
- **Corpus:** an [Obsidian](https://obsidian.md) vault of markdown policy notes
  with YAML frontmatter (`airline`, `jurisdiction`, `doc_type`, `topic`, …),
  which becomes searchable metadata.
- **Generation:** Ollama `llama3.2:3b` by default; Claude if a key is set.

## Retriever versions

| Tag | Retriever | Technique |
|-----|-----------|-----------|
| `v1-naive`    | Vector similarity      | Fixed-size chunks + cosine search |
| `v2-hybrid`   | Hybrid                 | BM25 + vector, fused with reciprocal rank fusion |
| `v3-reranked` | Hybrid + rerank        | v2 + cross-encoder reranking |
| `v4-metadata` | Metadata-filtered      | Self-query filtering on frontmatter |

_Evaluation results table — coming soon._

## Status

🚧 Early construction. Weekend 1: repo scaffold, local infra, ingest pipeline,
and the `v1-naive` retriever.

## License / data

Source policy PDFs are **not** committed to this repository. They are downloaded
from official sources by `src/ingest/fetch_data.py` for local, personal,
educational use. All policies belong to their respective airlines and
governments.
