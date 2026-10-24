"""Data contracts shared across ingestion, indexing, and answer generation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceDocument:
    """A PDF source before its pages are extracted and chunked."""

    path: Path
    filename: str
    size_bytes: int


@dataclass(frozen=True)
class PageDocument:
    """Normalized text and provenance for one source PDF page."""

    document_name: str
    document_path: str
    document_sha256: str
    page_number: int
    text: str
    extraction_status: str


@dataclass(frozen=True)
class ValidationIssue:
    """A quality finding attached to an extracted page."""

    document_name: str
    page_number: int
    severity: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Summary of extraction quality checks."""

    total_pages: int
    error_count: int
    warning_count: int
    issues: list[ValidationIssue]


@dataclass(frozen=True)
class ChunkDocument:
    """A citation-preserving, retrieval-ready unit of source text."""

    chunk_id: str
    document_name: str
    document_path: str
    document_sha256: str
    page_number: int
    chunk_index_on_page: int
    text: str


@dataclass(frozen=True)
class EmbeddedChunk:
    """A retrieval chunk paired with the numerical representation used for search."""

    chunk: ChunkDocument
    embedding_model: str
    embedding: list[float]
