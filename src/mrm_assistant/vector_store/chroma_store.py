"""Small ChromaDB adapter for storing and searching MRM chunk vectors.

This module hides Chroma-specific code behind a simple class. The rest of the
application only needs to provide chunk IDs, text, embeddings, and metadata.
"""

from pathlib import Path
from typing import Any

from mrm_assistant.models import EmbeddedChunk, RetrievedChunk


DEFAULT_COLLECTION_NAME = "mrm_chunks_v020"


def build_metadata(record: EmbeddedChunk) -> dict[str, str | int | float | bool]:
    """Flatten chunk provenance into Chroma's filterable metadata fields."""
    return {
        "document_name": record.chunk.document_name,
        "document_path": record.chunk.document_path,
        "document_sha256": record.chunk.document_sha256,
        "page_number": record.chunk.page_number,
        "chunk_index_on_page": record.chunk.chunk_index_on_page,
        "embedding_model": record.embedding_model,
        "index_version": "v0.2.0-chroma",
    }


class ChromaStore:
    """Persistent local Chroma collection for development and learning."""

    def __init__(self, database_path: Path, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "ChromaDB is not installed. Install project dependencies first."
            ) from error

        self._client = chromadb.PersistentClient(path=str(database_path))
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"purpose": "MRM RAG chunk retrieval", "distance_metric": "cosine"},
        )

    def upsert(self, records: list[EmbeddedChunk]) -> int:
        """Insert or replace vectors, text, and metadata using stable chunk IDs."""
        if not records:
            return 0

        self.collection.upsert(
            ids=[record.chunk.chunk_id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.chunk.text for record in records],
            metadatas=[build_metadata(record) for record in records],
        )
        return len(records)

    def count(self) -> int:
        """Return the number of vectors currently stored."""
        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return nearest chunks and their provenance metadata."""
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document,
                distance=float(distance),
                metadata=metadata,
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
