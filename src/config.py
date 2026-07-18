"""Central configuration, read from the environment with local-Docker defaults.

The whole project is driven by ``DATABASE_URL`` so it runs zero-config against
the local docker-compose stack, but can point at a hosted Postgres (e.g. Neon)
by setting that one variable in a gitignored ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load a local .env if present (gitignored); no-op otherwise

ROOT = Path(__file__).resolve().parents[1]
VAULT_DIR = ROOT / "vault"

# --- Vector store (pgvector) ---
# psycopg3 SQLAlchemy URL. Defaults match docker-compose.yml + .env.example.
_DEFAULT_DB = "postgresql+psycopg://rag:rag@localhost:5432/flight_refund"


def _normalize_db_url(url: str) -> str:
    """Force the psycopg3 driver so raw Neon/Postgres URLs work as-is.

    Hosted providers (e.g. Neon) hand out ``postgresql://...`` URLs, but
    langchain-postgres uses SQLAlchemy + psycopg3, which needs the explicit
    ``postgresql+psycopg://`` scheme (otherwise SQLAlchemy tries psycopg2).
    """
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):  # some providers use the short form
        return "postgresql+psycopg://" + url[len("postgres://"):]
    return url


DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", _DEFAULT_DB))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "flight_refund")


def get_engine():
    """Shared SQLAlchemy engine with stale-connection resilience.

    ``pool_pre_ping`` tests pooled connections before use and reconnects
    transparently — required for serverless Postgres (e.g. Neon), which
    suspends compute after inactivity and kills idle connections.
    """
    from sqlalchemy import create_engine

    return create_engine(DATABASE_URL, pool_pre_ping=True)

# --- Embeddings (local sentence-transformers) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chunking (v1-naive: fixed size) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Generation ---
# Ollama by default (no API key). If ANTHROPIC_API_KEY is set, callers upgrade.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
