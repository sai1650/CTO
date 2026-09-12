"""Idempotent initialization of the document-specific Chroma index."""

from pathlib import Path
import logging
from typing import Any

from .config import PROJECT_ROOT, Settings
from .document_loader import load_document
from .text_processor import Chunk, chunk_document

logger = logging.getLogger(__name__)

REQUIRED_TERMS = (
    "गेहूं",
    "धान",
    "सिंचाई",
    "मिट्टी",
    "उर्वरक",
    "ड्रोन",
    "सतत कृषि",
)


def load_chunks(settings: Settings, source_path: Path) -> list[Chunk]:
    document = load_document(source_path)
    if not document:
        raise ValueError(f"Document is empty: {source_path}")

    missing_terms = [term for term in REQUIRED_TERMS if term not in document]
    if missing_terms:
        raise ValueError(
            "Source document is missing required agriculture terms: "
            + ", ".join(missing_terms)
        )

    chunks = chunk_document(
        document,
        source=source_path.name,
        target_characters=settings.chunk_size,
        overlap_characters=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"No chunks were created from {source_path}.")
    return chunks


def _has_valid_index(store: Any, chunks: list[Chunk]) -> bool:
    collection_metadata = getattr(store.collection, "metadata", None)
    if collection_metadata is not None and collection_metadata.get(
        "embedding_model"
    ) != store.embedding_model.model_name:
        return False

    expected_ids = {chunk.chunk_id for chunk in chunks}
    if store.count() != len(expected_ids):
        return False

    indexed = store.collection.get(
        ids=list(expected_ids),
        include=["metadatas"],
    )
    indexed_ids = set(indexed.get("ids", []))
    indexed_metadata = indexed.get("metadatas", [])
    return indexed_ids == expected_ids and all(
        metadata and metadata.get("source") == chunks[0].source
        for metadata in indexed_metadata
    )


def ensure_index(
    settings: Settings,
    store: Any,
    source_path: str | Path | None = None,
) -> int:
    """Ensure the expected source is indexed exactly once per store."""
    default_source = PROJECT_ROOT / "data" / "raw" / "hindi_document.txt"
    resolved_source = Path(source_path or default_source).resolve()
    logger.info("Checking ChromaDB index...")
    chunks = load_chunks(settings, resolved_source)

    if _has_valid_index(store, chunks):
        document_count = store.count()
        logger.info("Using existing index with %d chunks", document_count)
        return document_count

    logger.info("Building ChromaDB index...")
    store.rebuild_collection()
    store.upsert_chunks(chunks)
    document_count = store.count()
    if document_count != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} Chroma documents, found {document_count}."
        )
    logger.info("Index built successfully: %d chunks", document_count)
    return document_count
