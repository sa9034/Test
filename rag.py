"""RAG retrieval — each agent queries ChromaDB filtered by its domain tags."""

from __future__ import annotations

import config
from models import RetrievedChunk

# Map agent names to their domain filter key
AGENT_DOMAIN_MAP: dict[str, str] = {
    config.AGENT_RISK: "risk",
    config.AGENT_SAFETY: "safety",
    config.AGENT_GOVERNANCE: "governance",
}


def retrieve_chunks(
    query: str,
    agent_name: str,
    top_k: int = config.TOP_K_CHUNKS,
) -> list[RetrievedChunk]:
    """Retrieve top-k relevant chunks for an agent's domain. Returns empty list on any failure."""
    try:
        from embeddings import get_or_create_collection
        collection = get_or_create_collection()
    except Exception:
        return []

    domain = AGENT_DOMAIN_MAP.get(agent_name)
    where_filter = {f"domain_{domain}": True} if domain else None

    # Try with domain filter first, then without, then give up gracefully
    for attempt_filter in [where_filter, None]:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=attempt_filter,
                include=["documents", "metadatas", "distances"],
            )
            if results and results["documents"] and results["documents"][0]:
                chunks: list[RetrievedChunk] = []
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    chunks.append(
                        RetrievedChunk(
                            text=doc,
                            source=meta.get("source", "unknown"),
                            page=meta.get("page"),
                            relevance_score=round(1 - dist, 4),
                        )
                    )
                return chunks
        except Exception:
            continue

    return []
