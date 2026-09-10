from pathlib import Path

from src.config import Settings
from src.indexing import ensure_index


class FakeCollection:
    def __init__(self):
        self.documents = {}

    def get(self, ids, include):
        indexed = [
            self.documents[item_id]
            for item_id in ids
            if item_id in self.documents
        ]
        return {
            "ids": [item_id for item_id in ids if item_id in self.documents],
            "metadatas": [item["metadata"] for item in indexed],
        }


class FakeStore:
    def __init__(self):
        self.collection = FakeCollection()
        self.rebuild_calls = 0
        self.upsert_calls = 0

    def count(self):
        return len(self.collection.documents)

    def rebuild_collection(self):
        self.rebuild_calls += 1
        self.collection.documents = {}

    def upsert_chunks(self, chunks):
        self.upsert_calls += 1
        self.collection.documents = {
            chunk.chunk_id: {"metadata": {"source": chunk.source}}
            for chunk in chunks
        }


def _source_path(tmp_path: Path) -> Path:
    source = tmp_path / "hindi_document.txt"
    source.write_text(
        "गेहूं धान सिंचाई मिट्टी उर्वरक ड्रोन सतत कृषि।",
        encoding="utf-8",
    )
    return source


def test_ensure_index_builds_empty_store_once(tmp_path):
    settings = Settings(CHUNK_SIZE=400, CHUNK_OVERLAP=60)
    store = FakeStore()
    source = _source_path(tmp_path)

    assert ensure_index(settings, store, source) == 1
    assert store.rebuild_calls == 1
    assert store.upsert_calls == 1

    assert ensure_index(settings, store, source) == 1
    assert store.rebuild_calls == 1
    assert store.upsert_calls == 1
