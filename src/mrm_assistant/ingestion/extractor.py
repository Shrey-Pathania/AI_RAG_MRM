"""Extract page-level text from PDF source documents."""

from hashlib import sha256
from pathlib import Path

from mrm_assistant.models import PageDocument, SourceDocument


def calculate_file_hash(path: Path) -> str:
    """Return a stable source version identifier without loading the whole file."""
    digest = sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pages(source: SourceDocument) -> list[PageDocument]:
    """Extract text from every page while retaining citation-ready provenance."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF extraction requires pypdf. Install the project dependencies first."
        ) from error

    document_hash = calculate_file_hash(source.path)
    reader = PdfReader(str(source.path))
    pages: list[PageDocument] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\r\n", "\n").strip()
        pages.append(
            PageDocument(
                document_name=source.filename,
                document_path=str(source.path),
                document_sha256=document_hash,
                page_number=page_number,
                text=text,
                extraction_status="success" if text else "empty_text",
            )
        )
    return pages
