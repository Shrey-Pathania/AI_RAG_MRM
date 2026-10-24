"""OpenAI-compatible embedding provider implementation."""

from collections.abc import Iterable
from typing import Any


DEFAULT_BATCH_SIZE = 50


def build_client(api_key: str | None, base_url: str | None) -> Any:
    """Build the provider client only when an embedding operation is requested."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to create an embedding index.")

    from openai import OpenAI

    client_options: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    return OpenAI(**client_options)


def batched(items: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> Iterable[list[str]]:
    """Yield bounded batches to control request size and simplify retries later."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_texts(client: Any, texts: list[str], model: str) -> list[list[float]]:
    """Embed text in batches while preserving the original input order."""
    vectors: list[list[float]] = []
    for batch in batched(texts):
        response = client.embeddings.create(model=model, input=batch, encoding_format="float")
        ordered_embeddings = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered_embeddings)
    return vectors
