"""ChromaDB vector store setup with sentence-transformers embeddings."""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import config

_client: chromadb.ClientAPI | None = None


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is not None:
        return _client

    # Try persistent client first; fall back to in-memory if it fails
    try:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    except Exception:
        _client = chromadb.Client()
    return _client


def get_or_create_collection(
    client: chromadb.ClientAPI | None = None,
) -> chromadb.Collection:
    client = client or get_chroma_client()
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
