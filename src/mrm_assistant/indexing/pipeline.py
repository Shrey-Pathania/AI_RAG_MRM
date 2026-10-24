"""Create a portable local vector index from retrieval chunks."""

from dataclasses import asdict
import json
from pathlib import Path

from mrm_assistant.chunking.pipeline import CHUNKS_FILENAME
from mrm_assistant.embeddings.openai_provider import build_client, embed_texts
from mrm_assistant.models import ChunkDocument, EmbeddedChunk


VECTOR_INDEX_FILENAME = "vector_index.jsonl"
INDEX_MANIFEST_FILENAME = "index_manifest.json"


def read_chunks(path: Path) -> list[ChunkDocument]:
    """Read the portable chunks created by the chunking pipeline."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunks were not found at {path}. Run the chunk command first.")
    with path.open(encoding="utf-8") as input_file:
        return [ChunkDocument(**json.loads(line)) for line in input_file if line.strip()]


def write_vector_index(records: list[EmbeddedChunk], output_path: Path) -> None:
    """Persist vectors alongside their citation metadata for local retrieval."""
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_manifest(records: list[EmbeddedChunk], output_path: Path) -> None:
    """Record index dimensions and provenance for reproducibility checks."""
    manifest = {
        "chunk_count": len(records),
        "embedding_model": records[0].embedding_model if records else None,
        "embedding_dimensions": len(records[0].embedding) if records else 0,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)
        output_file.write("\n")


def run_indexing(
    data_dir: Path,
    api_key: str | None,
    base_url: str | None,
    embedding_model: str,
) -> int:
    """Generate embeddings and save a local, reproducible vector index."""
    chunks = read_chunks(data_dir / CHUNKS_FILENAME)
    client = build_client(api_key, base_url)
    vectors = embed_texts(client, [chunk.text for chunk in chunks], embedding_model)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding provider returned a different number of vectors than chunks.")

    records = [
        EmbeddedChunk(chunk=chunk, embedding_model=embedding_model, embedding=vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    write_vector_index(records, data_dir / VECTOR_INDEX_FILENAME)
    write_manifest(records, data_dir / INDEX_MANIFEST_FILENAME)
    return len(records)
