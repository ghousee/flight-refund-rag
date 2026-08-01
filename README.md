<div align="center">

# ✈️ flight-refund-rag

**A retrieval-augmented generation system that answers flight-cancellation refund
questions — grounded in real airline and government policy, with every answer cited.**

Air Canada · WestJet · Canada APPR · US DOT

<br>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-orchestration-1C3C3C?logo=langchain&logoColor=white)
![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2:3b-000000?logo=ollama&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![No API keys](https://img.shields.io/badge/API_keys-none_required-success)

</div>

---

## Why this project

Anyone can wire an LLM to a vector database. The interesting engineering question
is **how much retrieval quality actually matters** — so this repo ships **four
progressively smarter retrievers** and measures each against a hand-authored
golden set. The headline isn't "a refund chatbot," it's a reproducible,
apples-to-apples comparison of naive vector search → hybrid → reranked →
metadata-filtered retrieval.

> [!NOTE]
> **Runs with zero API keys and zero accounts.** PostgreSQL/pgvector and the LLM
> (Ollama `llama3.2:3b`) both run locally in Docker. Clone → `docker compose up`
> → one ingest command → working system. Set `ANTHROPIC_API_KEY` to optionally
> upgrade generation to Claude.

## Architecture

```mermaid
flowchart LR
    subgraph INGEST["🛠️  Ingest — offline, run once"]
        direction LR
        A[Official policy<br/>PDFs] -->|fetch_data.py| B[data/raw/]
        B -->|parse_pdfs.py<br/>PyMuPDF| C[Obsidian vault<br/>markdown + YAML<br/>frontmatter]
        C -->|index.py<br/>chunk + embed| D[(PostgreSQL<br/>+ pgvector)]
    end

    subgraph QUERY["💬  Query — online, per question"]
        direction LR
        Q[User question] --> E[Embed]
        E --> R[Retrieve]
        R --> L[LLM]
        L --> ANS[Grounded answer<br/>+ citations]
    end

    D -.provides context.-> R

    style INGEST fill:#0d1b2a20,stroke:#4169E1
    style QUERY fill:#1b2a0d20,stroke:#2e8b57
```

- **Embeddings** — local `BAAI/bge-small-en-v1.5` (384-dim, deterministic, no key).
- **Vector store** — PostgreSQL + pgvector (`vector` column type, `<=>` cosine distance).
- **Corpus** — an [Obsidian](https://obsidian.md) vault of markdown policy notes.
  YAML frontmatter (`airline`, `jurisdiction`, `doc_type`, `topic`, …) becomes
  searchable metadata on every chunk.
- **Generation** — Ollama `llama3.2:3b` by default; Claude if a key is set.

## The four retrievers

| Tag | Retriever | Technique | What it adds |
|-----|-----------|-----------|--------------|
| `v1-naive`    | Vector similarity   | Fixed-size chunks + cosine search               | Baseline |
| `v2-hybrid`   | Hybrid              | BM25 + vector, fused via reciprocal rank fusion | Keyword + semantic recall |
| `v3-reranked` | Hybrid + rerank     | v2 + cross-encoder reranking                    | Precision at the top |
| `v4-metadata` | Metadata-filtered   | Self-query filtering on frontmatter             | Scoped, structured retrieval |

Each version is a git tag. Evaluation compares them on the same golden set.

### Results

> [!NOTE]
> **Preliminary — golden set is n=20 answerable questions (+5 unanswerable),
> growing toward ~50.** Treat as directional. RAGAS answer-quality metrics and
> the generator comparison are in progress.

**Retrieval quality** (20 answerable questions; local Docker, CPU):

| Retriever | recall@5 | MRR | hit-rate | latency/query |
|-----------|:--------:|:---:|:--------:|:-------------:|
| `v1-naive` (vector)              | **0.67** | **0.65** | **0.80** | 0.08s |
| `v2-hybrid` (BM25 + vector, RRF) | 0.60 | 0.61 | 0.70 | 0.09s |
| `v3-reranked` | _in progress_ | | | |
| `v4-metadata` | _in progress_ | | | |

**recall@5 by question category** (n=5 each):

| Category | `v1-naive` | `v2-hybrid` | Notes |
|----------|:----------:|:-----------:|-------|
| airline-specific    | 1.00 | 1.00 | Vector nails clean-vocabulary lookups |
| simple-lookup       | 0.70 | 0.50 | Hybrid *hurts* — lexical fusion adds noise |
| casual-vs-legalese  | 0.50 | 0.50 | Casual wording ≠ legalese; a semantic gap |
| multi-hop           | 0.47 | 0.40 | **Hardest** — needs 2+ notes in the top 5 |

### Findings so far

1. **Right retrieval, wrong reasoning (the headline).** `v1` retrieves the
   correct source (e.g. 14 CFR §260.6, which *guarantees* a cash refund for a
   cancelled flight) but the 3B model sometimes **contradicts its own citation**
   ("you cannot get a cash refund"). Failure localizes to *generation*, not
   retrieval — which is why the next experiment holds the retriever constant and
   swaps the generator (llama3.2:3b → Claude) to measure how much of the error
   budget is small-model synthesis.

2. **Hybrid made retrieval *worse* here — and sample size mattered.** At n=9,
   `v2-hybrid` looked like a tie with `v1`; at n=20 it's clearly behind
   (recall@5 0.67 → 0.60, hit-rate 0.80 → 0.70). BM25's lexical signal added
   ranking noise to already-strong embeddings — worst on `simple-lookup`
   (0.70 → 0.50) — while the real failures are *semantic* (casual wording vs.
   legalese), which lexical matching can't bridge. Lesson: measure per corpus,
   and beware conclusions drawn from a handful of questions.

3. **Multi-hop is the hardest category** (recall ≈ 0.47) — those questions need
   two+ notes (e.g. a regulation *and* a tariff) both in the top 5. This, plus
   the casual-vs-legalese semantic gap, is the target for `v3`'s cross-encoder
   reranking, which re-scores query–chunk *pairs* and can rescue semantic
   ranking that bi-encoder embeddings miss.

_Reproduce:_ `python -m eval.retrieval_metrics --retriever v1-naive` (or `v2-hybrid`).

## Technologies

| Area | Choice | Why |
|------|--------|-----|
| Orchestration | **LangChain** | Standard RAG building blocks; swappable retrievers |
| Vector store | **PostgreSQL + pgvector** | SQL-native, production-realistic, one `<=>` operator |
| Embeddings | **sentence-transformers** (`bge-small-en-v1.5`) | Local, deterministic, no API key |
| Generation | **Ollama** (`llama3.2:3b`), optional **Claude** | Runs offline by default |
| PDF parsing | **PyMuPDF** | Fast, layout-aware text extraction |
| Corpus format | **Obsidian vault** (markdown + YAML) | Human-inspectable, git-diffable, metadata-rich |
| Evaluation | **RAGAS** + custom recall@5 / MRR (via `pytest`) | LLM-judged + retrieval metrics |
| Infra | **Docker Compose** | One command, reproducible local stack |

## Project structure

```
flight-refund-rag/
├── vault/                 # Obsidian vault — policy notes w/ YAML frontmatter (committed)
├── data/raw/              # source PDFs — gitignored, reproduced via fetch_data.py
├── src/
│   ├── ingest/
│   │   ├── fetch_data.py  # download corpus PDFs from official URLs
│   │   ├── parse_pdfs.py  # PDF → markdown vault notes with frontmatter
│   │   └── index.py       # ObsidianLoader → chunk → embed → pgvector
│   └── retrievers/        # v1..v4 retriever implementations
├── eval/
│   └── golden_set.jsonl   # ~50 hand-authored Q&A pairs (incl. ~5 unanswerable)
├── notebooks/             # exploration
├── docker-compose.yml     # postgres + pgvector, ollama
├── requirements.txt
└── .env.example           # config template (no secrets)
```

## Quickstart

> [!WARNING]
> 🚧 Under active construction — commands below are being wired up as the
> ingest pipeline and `v1-naive` retriever land. This section will be verified
> end-to-end before the first release tag.

```bash
# 1. Start local infrastructure (Postgres + pgvector, Ollama)
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b   # one-time, ~2GB

# 2. Build the corpus and index it
python -m src.ingest.fetch_data     # download policy PDFs → data/raw/
python -m src.ingest.parse_pdfs     # PDF → vault/ markdown notes
python -m src.ingest.index          # embed + load into pgvector

# 3. Ask a question (v1-naive)
python -m src.retrievers.v1_naive "Can I get a refund if my flight was cancelled?"
```

## Disclaimer

- **Educational portfolio project.** Not affiliated with, endorsed by, or
  sponsored by any airline, government, or regulator.
- **Not legal advice.** Nothing here is legal advice. For your situation,
  consult the airline, the relevant regulator, or a qualified professional.
- **Policies change.** The corpus reflects source documents as retrieved on the
  dates recorded in each note's `retrieved_date` frontmatter. Rules may have
  changed since — always verify current rules with the airline or regulator.
- **AI-generated answers may be wrong.** Responses are produced by a language
  model over retrieved excerpts and may be inaccurate or incomplete. (The
  evaluation suite includes refusal tests precisely because of this.)
- **No redistribution of airline documents.** Airline contracts of carriage are
  **not** committed to this repo; `src/ingest/fetch_data.py` downloads them
  locally from official URLs for personal, educational use. Only
  government-sourced notes, code, and templates are committed.

## License

The **code** in this repository is licensed under the [MIT License](LICENSE).

Government regulatory text under `vault/regulations/` is reproduced under its own terms:
- **US — 14 CFR Parts 259 & 260** (eCFR): a U.S. Government work, in the **public domain**.
- **Canada — APPR (SOR/2019-150)** (Justice Laws): reproduced under the
  **Reproduction of Federal Law Order**. This is **not an official version** —
  the official version is on the
  [Justice Laws website](https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-150/).
  Text was reproduced as of the `retrieved_date` recorded in each note.

<div align="center">
<sub>Built as a learning project and portfolio piece.</sub>
</div>
