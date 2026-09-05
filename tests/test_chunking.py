from src.document_loader import load_document
from src.text_processor import chunk_document


def test_loader_preserves_hindi(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("नमस्ते\n\nभारत", encoding="utf-8")
    assert "नमस्ते" in load_document(path)


def test_chunking_has_metadata_and_boundaries():
    chunks = chunk_document(
        "## जल\n\nजल संरक्षण महत्वपूर्ण है। वर्षा जल संचयन उपयोगी है।",
        source="test.txt",
        target_characters=40,
        overlap_characters=10,
    )
    assert chunks
    assert chunks[0].chunk_id == "chunk-0000"
    assert chunks[0].source == "test.txt"
    assert chunks[0].character_end > chunks[0].character_start
    assert chunks[0].to_dict()["chunk_index"] == 0


def test_long_hindi_document_creates_multiple_chunks():
    text = "\n\n".join(
        [
            "## कृषि\nगेहूं की खेती के लिए दोमट मिट्टी उपयुक्त है। "
            "खेत में पर्याप्त जल निकास होना चाहिए।",
            "## सिंचाई\nफसल को समय पर पानी देना चाहिए। "
            "वर्षा जल का संरक्षण उपयोगी होता है।",
            "## खाद\nजैविक खाद मिट्टी की गुणवत्ता बढ़ाती है। "
            "किसानों को संतुलित पोषक तत्व देने चाहिए।",
        ]
    )
    chunks = chunk_document(
        text,
        source="agriculture.txt",
        target_characters=140,
        overlap_characters=30,
    )
    assert len(chunks) > 1
    assert any("गेहूं" in chunk.text for chunk in chunks)


def test_chunks_have_sentence_overlap():
    text = (
        "पहला वाक्य पूरा है। दूसरा वाक्य भी पूरा है। "
        "तीसरा वाक्य यहां है। चौथा वाक्य अंत में है।"
    )
    chunks = chunk_document(
        text,
        target_characters=45,
        overlap_characters=20,
    )
    assert len(chunks) > 1
    overlap = set()
    for previous, current in zip(chunks, chunks[1:]):
        for sentence in previous.text.split("।"):
            sentence = sentence.strip()
            if sentence:
                sentence = f"{sentence}।"
                if sentence in current.text:
                    overlap.add(sentence)
    assert overlap
    assert all(len(sentence) <= 45 for sentence in overlap)


def test_chunks_are_non_empty_and_preserve_complete_sentences():
    text = (
        "पहला वाक्य समाप्त होता है। दूसरा वाक्य भी समाप्त होता है। "
        "तीसरा वाक्य समाप्त होता है।"
    )
    chunks = chunk_document(text, target_characters=45, overlap_characters=15)
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.text.endswith("।") for chunk in chunks)


def test_short_document_stays_as_one_chunk():
    chunks = chunk_document("यह एक छोटा दस्तावेज़ है।")
    assert len(chunks) == 1
    assert chunks[0].text == "यह एक छोटा दस्तावेज़ है।"


def test_hindi_unicode_is_preserved_in_chunks():
    text = "गेहूं की खेती के लिए दोमट मिट्टी उपयुक्त है।"
    chunks = chunk_document(text)
    assert chunks[0].text == text
    assert "गेहूं" in chunks[0].text
    assert "मिट्टी" in chunks[0].text
