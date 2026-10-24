"""Create citation-preserving chunks from normalized page records."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from mrm_assistant.chunking.splitter import split_text
from mrm_assistant.models import ChunkDocument, PageDocument


EXTRACTED_PAGES_FILENAME = "extracted_pages.jsonl"
CHUNKS_FILENAME = "chunks.jsonl"


def read_page_records(path: Path) -> list[PageDocument]:
    """Load the stable output produced by the ingestion pipeline."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Normalized pages were not found at {path}. Run the ingest command first."
        )
    with path.open(encoding="utf-8") as input_file:
        return [PageDocument(**json.loads(line)) for line in input_file if line.strip()]


def make_chunk_id(page: PageDocument, chunk_index: int) -> str:
    """Create a stable ID tied to a specific source version and page segment."""
    identity = f"{page.document_sha256}:{page.page_number}:{chunk_index}"
    return sha256(identity.encode("utf-8")).hexdigest()


def create_chunks(pages: list[PageDocument]) -> list[ChunkDocument]:
    """Turn page records into bounded retrieval units without losing provenance."""
    chunks: list[ChunkDocument] = []
    for page in pages:
        for chunk_index, text in enumerate(split_text(page.text), start=1):
            chunks.append(
                ChunkDocument(
                    chunk_id=make_chunk_id(page, chunk_index),
                    document_name=page.document_name,
                    document_path=page.document_path,
                    document_sha256=page.document_sha256,
                    page_number=page.page_number,
                    chunk_index_on_page=chunk_index,
                    text=text,
                )
            )
    return chunks


def write_chunks(chunks: list[ChunkDocument], output_path: Path) -> None:
    """Write portable chunk records, ready for any later vector-store choice."""
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def run_chunking(data_dir: Path) -> int:
    """Read extracted pages and persist their retrieval-ready chunks."""
    pages = read_page_records(data_dir / EXTRACTED_PAGES_FILENAME)
    chunks = create_chunks(pages)
    write_chunks(chunks, data_dir / CHUNKS_FILENAME)
    return len(chunks)
