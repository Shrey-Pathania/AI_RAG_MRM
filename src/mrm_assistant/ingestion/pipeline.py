"""Coordinate discovery, extraction, validation, and durable ingestion output."""

from dataclasses import asdict
import json
from pathlib import Path

from mrm_assistant.ingestion.discovery import discover_pdf_sources
from mrm_assistant.ingestion.extractor import extract_pages
from mrm_assistant.ingestion.validator import validate_pages
from mrm_assistant.models import PageDocument, ValidationReport


EXTRACTED_PAGES_FILENAME = "extracted_pages.jsonl"
VALIDATION_REPORT_FILENAME = "validation_report.json"


def write_jsonl(records: list[PageDocument], output_path: Path) -> None:
    """Write one self-contained page record per line for downstream modules."""
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_validation_report(report: ValidationReport, output_path: Path) -> None:
    """Persist human- and machine-readable ingestion quality results."""
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(report), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def run_ingestion(source_dir: Path, output_dir: Path) -> ValidationReport:
    """Create normalized, validated page records from every approved PDF."""
    sources = discover_pdf_sources(source_dir)
    all_pages: list[PageDocument] = []
    for source in sources:
        all_pages.extend(extract_pages(source))

    report = validate_pages(all_pages)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_pages, output_dir / EXTRACTED_PAGES_FILENAME)
    write_validation_report(report, output_dir / VALIDATION_REPORT_FILENAME)
    return report
