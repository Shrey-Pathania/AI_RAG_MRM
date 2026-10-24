"""Load the frozen local vector index into a persistent Chroma collection."""

import json
from pathlib import Path

from mrm_assistant.models import ChunkDocument, EmbeddedChunk
from mrm_assistant.vector_store.chroma_store import ChromaStore


VECTOR_INDEX_FILENAME = "vector_index.jsonl"


def read_embedded_chunks(path: Path) -> list[EmbeddedChunk]:
    """Read vector records created by the v0.1.0 embedding pipeline."""
    if not path.is_file():
        raise FileNotFoundError(f"Vector index was not found at {path}. Run `index` first.")

    records: list[EmbeddedChunk] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                EmbeddedChunk(
                    chunk=ChunkDocument(**item["chunk"]),
                    embedding_model=item["embedding_model"],
                    embedding=item["embedding"],
                )
            )
    return records


def run_chroma_indexing(data_dir: Path, chroma_dir: Path) -> int:
    """Persist existing embeddings and metadata in Chroma without re-embedding."""
    records = read_embedded_chunks(data_dir / VECTOR_INDEX_FILENAME)
    store = ChromaStore(chroma_dir)
    return store.upsert(records)
