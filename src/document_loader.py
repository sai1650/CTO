"""UTF-8 document loading and safe Unicode normalization."""

from pathlib import Path
import unicodedata


def load_document(path: str | Path) -> str:
    """Load a UTF-8 text document and normalize it without changing scripts."""
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Document not found: {document_path}")
    text = document_path.read_text(encoding="utf-8-sig")
    normalized = unicodedata.normalize("NFC", text)
    lines = [" ".join(line.split()) for line in normalized.splitlines()]
    return "\n".join(lines).strip()
