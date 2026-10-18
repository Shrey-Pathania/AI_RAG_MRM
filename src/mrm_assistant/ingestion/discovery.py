"""Discover approved PDF source documents deterministically."""

from pathlib import Path

from mrm_assistant.models import SourceDocument


def discover_pdf_sources(source_dir: Path) -> list[SourceDocument]:
    """Return PDFs in a stable order; fail clearly when the corpus is absent."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    sources = [
        SourceDocument(path=path, filename=path.name, size_bytes=path.stat().st_size)
        for path in sorted(source_dir.glob("*.pdf"), key=lambda item: item.name.casefold())
    ]
    if not sources:
        raise FileNotFoundError(f"No PDF source documents found in: {source_dir}")
    return sources
