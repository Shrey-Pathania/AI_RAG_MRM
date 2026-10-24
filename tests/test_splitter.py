"""Fast unit tests for citation-safe text chunking."""

import pytest

from mrm_assistant.chunking.splitter import split_text


def test_split_text_keeps_overlap_between_chunks() -> None:
    text = "one two three four five six seven eight nine ten"

    chunks = split_text(text, max_words=4, overlap_words=1)

    assert chunks == ["one two three four", "four five six seven", "seven eight nine ten"]


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="less than max_words"):
        split_text("some text", max_words=3, overlap_words=3)
