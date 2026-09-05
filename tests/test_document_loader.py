from src.document_loader import load_document
from pathlib import Path


def test_loader_normalizes_unicode_without_translation(tmp_path):
    path = tmp_path / "unicode.txt"
    path.write_text("नमस्ते   दुनिया", encoding="utf-8")
    assert load_document(path) == "नमस्ते दुनिया"


def test_project_source_is_agriculture_document():
    source = Path(__file__).parents[1] / "data" / "raw" / "hindi_document.txt"
    document = load_document(source)
    assert "गेहूं" in document
    assert "धान" in document
    assert "ड्रोन" in document
