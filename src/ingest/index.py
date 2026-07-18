"""Index the Obsidian vault into pgvector.

Pipeline (the "ingest" half of RAG):
    vault/*.md  ->  ObsidianLoader (frontmatter -> metadata)
                ->  RecursiveCharacterTextSplitter (fixed-size chunks)
                ->  HuggingFaceEmbeddings (bge-small, 384-dim, normalized)
                ->  PGVector (one row per chunk: text + embedding + metadata)

The vector store is driven by ``DATABASE_URL`` (see src/config.py), so this
loads into the local Docker Postgres by default or a hosted Neon database if
that variable points there.

Usage:
    python -m src.ingest.index            # rebuild the collection from scratch
    python -m src.ingest.index --append   # add without clearing existing rows
"""

from __future__ import annotations

import argparse

from langchain_community.document_loaders import ObsidianLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def load_notes():
    """Load every vault note; YAML frontmatter becomes document metadata."""
    loader = ObsidianLoader(str(config.VAULT_DIR), collect_metadata=True)
    docs = loader.load()
    if not docs:
        raise SystemExit(
            f"No notes found in {config.VAULT_DIR}. "
            "Run fetch_data.py and parse_pdfs.py first."
        )
    return docs


def chunk(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,  # records each chunk's offset for traceability
    )
    return splitter.split_documents(docs)


def get_embeddings() -> HuggingFaceEmbeddings:
    # Normalized embeddings so pgvector cosine distance is the right metric.
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(reset: bool = True) -> None:
    docs = load_notes()
    chunks = chunk(docs)
    print(f"Loaded {len(docs)} notes -> {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        connection=config.get_engine(),
        use_jsonb=True,
        pre_delete_collection=reset,  # rebuild cleanly on each run by default
    )
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into collection "
          f"'{config.COLLECTION_NAME}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--append",
        action="store_true",
        help="add to the existing collection instead of rebuilding it",
    )
    args = parser.parse_args()
    build_index(reset=not args.append)


if __name__ == "__main__":
    main()
