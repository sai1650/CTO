from src.text_processor import Chunk


def test_chunk_serialization_preserves_metadata():
    chunk = Chunk("chunk-0001", "भारत", "doc.txt", "Section", 0, 4)
    payload = chunk.to_dict()
    assert payload["chunk_id"] == "chunk-0001"
    assert payload["text"] == "भारत"
