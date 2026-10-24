"""Search Chroma with a question embedding and return citation-ready chunks."""

from pathlib import Path

from mrm_assistant.embeddings.openai_provider import build_client, embed_texts
from mrm_assistant.models import RetrievedChunk
from mrm_assistant.vector_store.chroma_store import ChromaStore


def search_chroma(
    question: str,
    chroma_dir: Path,
    api_key: str | None,
    base_url: str | None,
    embedding_model: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Embed a question, search Chroma, and return the best matching chunks."""
    if not question.strip():
        raise ValueError("Question must contain text.")

    client = build_client(api_key, base_url)
    question_embedding = embed_texts(client, [question], embedding_model)[0]
    return ChromaStore(chroma_dir).search(question_embedding, top_k=top_k)
