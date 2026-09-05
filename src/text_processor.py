"""Paragraph-aware chunking for Hindi and multilingual text."""

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any

_SENTENCE_PATTERN = re.compile(r"(?<=[।!?])\s+|(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable document segment and its source coordinates."""

    chunk_id: str
    text: str
    source: str
    section: str
    character_start: int
    character_end: int
    chunk_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sentences(paragraph: str) -> list[str]:
    return [
        part.strip()
        for part in _SENTENCE_PATTERN.split(paragraph)
        if part.strip()
    ]


def chunk_document(
    text: str,
    source: str = "hindi_document.txt",
    target_characters: int = 400,
    overlap_characters: int = 50,
) -> list[Chunk]:
    """Create character-sized chunks at paragraph and sentence boundaries."""
    if target_characters <= 0 or overlap_characters < 0:
        raise ValueError("Chunk sizes must be positive")
    if overlap_characters >= target_characters:
        raise ValueError(
            "overlap_characters must be smaller than target_characters"
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    units = [
        sentence
        for paragraph in paragraphs
        for sentence in _sentences(paragraph)
    ]
    chunks: list[Chunk] = []
    current: list[str] = []
    has_new_content = False
    character_cursor = 0

    def join_units(values: list[str]) -> str:
        return " ".join(values).strip()

    def overlap_units(values: list[str]) -> list[str]:
        overlap: list[str] = []
        overlap_length = 0
        for unit in reversed(values):
            separator_length = 1 if overlap else 0
            candidate_length = len(unit) + separator_length + overlap_length
            if overlap and candidate_length > overlap_characters:
                break
            overlap.insert(0, unit)
            overlap_length = candidate_length
            if overlap_length >= overlap_characters:
                break
        return overlap

    def emit(values: list[str]) -> list[str]:
        nonlocal character_cursor
        if not values:
            return []
        chunk_text = join_units(values)
        start = character_cursor
        end = start + len(chunk_text)
        section = next(
            (line[2:].strip() for line in values if line.startswith("#")),
            "Document",
        )
        chunks.append(
            Chunk(
                f"chunk-{len(chunks):04d}",
                chunk_text,
                source,
                section,
                start,
                end,
                len(chunks),
            )
        )
        overlap = overlap_units(values)
        character_cursor = max(start, end - len(join_units(overlap)))
        return overlap

    for unit in units:
        if not current:
            current = [unit]
            has_new_content = True
            continue

        candidate = join_units(current + [unit])
        if len(candidate) <= target_characters:
            current.append(unit)
            has_new_content = True
            continue

        if has_new_content:
            current = emit(current)
            has_new_content = False

        candidate = join_units(current + [unit])
        if current and len(candidate) <= target_characters:
            current.append(unit)
        else:
            current = [unit]
        has_new_content = True

    emit(current)
    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """Persist chunks as readable UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        [chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2
    )
    output_path.write_text(payload, encoding="utf-8")
