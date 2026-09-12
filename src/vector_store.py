"""Persistent ChromaDB storage for document chunks."""

from pathlib import Path
import logging
from typing import Any

from .text_processor import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Thin, testable wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embedding_model,
    ) -> None:
        import chromadb
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model.model_name,
            },
        )

    def _refresh_collection(self) -> None:
        """Refresh the handle after another process rebuilds the collection."""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model.model_name,
            },
        )

    def count(self) -> int:
        """Return the current count, even after an external rebuild."""
        self._refresh_collection()
        return self.collection.count()

    def has_compatible_embedding_model(self) -> bool:
        """Return whether the collection was built with this model."""
        self._refresh_collection()
        return (
            self.collection.metadata.get("embedding_model")
            == self.embedding_model.model_name
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._refresh_collection()
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.to_dict() for chunk in chunks],
            embeddings=self.embedding_model.encode(
                [chunk.text for chunk in chunks]
            ),
        )

    def rebuild_collection(self) -> None:
        """Recreate the configured collection for clean ingestion."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model.model_name,
            },
        )

    def search(
        self, query: str, top_k: int = 4, relevance_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        document_count = self.count()
        logger.info(
            "Retrieval query=%r documents=%d top_k=%d threshold=%.3f",
            query,
            document_count,
            top_k,
            relevance_threshold,
        )
        if document_count == 0:
            logger.info("Retrieval returned 0 results")
            return []
        result = self.collection.query(
            query_embeddings=self.embedding_model.encode(query),
            n_results=min(top_k, document_count),
            include=["documents", "metadatas", "distances"],
        )
        matches: list[dict[str, Any]] = []
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            distance_value = float(distance)
            similarity = max(0.0, 1.0 - distance_value)
            logger.info(
                "Retrieval candidate distance=%.6f similarity=%.6f "
                "chunk_id=%s",
                distance_value,
                similarity,
                metadata.get("chunk_id"),
            )
            if similarity >= relevance_threshold:
                matches.append(
                    {
                        "text": document,
                        "metadata": metadata,
                        "distance": distance_value,
                        "similarity": similarity,
                    }
                )
        logger.info("Retrieval returned %d results", len(matches))
        return matches
