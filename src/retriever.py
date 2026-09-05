"""Retrieval orchestration and relevance filtering."""

from typing import Any


class Retriever:
    def __init__(
        self, vector_store, top_k: int = 4, relevance_threshold: float = 0.35
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return ranked context or an empty list when insufficient."""
        if not query.strip():
            return []
        return self.vector_store.search(
            query, self.top_k, self.relevance_threshold
        )
