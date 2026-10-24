"""Unit tests for Chroma metadata mapping without starting a database."""

from mrm_assistant.models import ChunkDocument, EmbeddedChunk
from mrm_assistant.vector_store.chroma_store import build_metadata


def test_build_metadata_keeps_citation_fields_flat() -> None:
    record = EmbeddedChunk(
        chunk=ChunkDocument(
            chunk_id="chunk-1",
            document_name="guidance.pdf",
            document_path="docs/guidance.pdf",
            document_sha256="hash",
            page_number=7,
            chunk_index_on_page=2,
            text="Example text",
        ),
        embedding_model="text-embedding-3-small",
        embedding=[0.1, 0.2],
    )

    metadata = build_metadata(record)

    assert metadata["document_name"] == "guidance.pdf"
    assert metadata["page_number"] == 7
    assert metadata["index_version"] == "v0.2.0-chroma"
    assert all(not isinstance(value, (dict, list)) for value in metadata.values())
