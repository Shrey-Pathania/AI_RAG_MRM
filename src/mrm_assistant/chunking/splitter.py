"""Deterministic, overlap-aware text splitting for the initial RAG baseline."""


def split_text(text: str, max_words: int = 350, overlap_words: int = 60) -> list[str]:
    """Split text into overlapping word windows without dropping source content."""
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero.")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be at least zero and less than max_words.")

    words = text.split()
    if not words:
        return []

    step_size = max_words - overlap_words
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step_size
    return chunks
