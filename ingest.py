"""Build the local chunk artifact and ChromaDB collection."""

from pathlib import Path
import logging

from src.config import get_settings
from src.document_loader import load_document
from src.embeddings import EmbeddingModel
from src.text_processor import chunk_document, save_chunks
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    root = Path(__file__).resolve().parent
    source_path = root / "data" / "raw" / "hindi_document.txt"
    chunks_path = root / "data" / "processed" / "chunks.json"
    document = load_document(source_path)
    if not document:
        raise ValueError(f"Document is empty: {source_path}")
    required_terms = [
        "गेहूं",
        "धान",
        "सिंचाई",
        "मिट्टी",
        "उर्वरक",
        "ड्रोन",
        "सतत कृषि",
    ]
    missing_terms = [term for term in required_terms if term not in document]
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
    logger.info("Document characters: %d", len(document))
    logger.info("Chunks created: %d", len(chunks))
    logger.info("Chunk IDs: %s", [chunk.chunk_id for chunk in chunks])
    save_chunks(chunks, chunks_path)
    embeddings = EmbeddingModel(settings.embedding_model)
    store = ChromaVectorStore(
        settings.chroma_persist_directory,
        settings.chroma_collection_name,
        embeddings,
    )
    logger.info("Embedding model: %s", embeddings.model_name)
    logger.info("Chroma directory: %s", settings.chroma_persist_directory)
    logger.info("Chroma collection: %s", settings.chroma_collection_name)
    store.rebuild_collection()
    store.upsert_chunks(chunks)
    document_count = store.collection.count()
    logger.info("Chroma documents: %d", document_count)
    if document_count != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} Chroma documents, found {document_count}."
        )
    print(
        f"Ingestion completed: {len(document)} characters, "
        f"{len(chunks)} chunks, {document_count} Chroma documents."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
