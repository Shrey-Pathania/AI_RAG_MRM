"""Fast unit tests for extraction-quality rules."""

from mrm_assistant.ingestion.validator import validate_pages
from mrm_assistant.models import PageDocument


def make_page(text: str, page_number: int = 1) -> PageDocument:
    return PageDocument(
        document_name="sample.pdf",
        document_path="docs/sample.pdf",
        document_sha256="abc123",
        page_number=page_number,
        text=text,
        extraction_status="success" if text else "empty_text",
    )


def test_validate_pages_accepts_normal_text() -> None:
    report = validate_pages([make_page("This is enough extracted text to pass the quality check.")])

    assert report.error_count == 0
    assert report.warning_count == 0


def test_validate_pages_flags_empty_and_short_text() -> None:
    report = validate_pages([make_page(""), make_page("Cover", page_number=2)])

    assert report.error_count == 1
    assert report.warning_count == 1
