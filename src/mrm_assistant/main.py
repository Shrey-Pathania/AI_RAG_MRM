"""Single application entry point that orchestrates independent modules."""

import argparse

from mrm_assistant.config import Settings
from mrm_assistant.ingestion.discovery import discover_pdf_sources
from mrm_assistant.ingestion.pipeline import run_ingestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MRM Knowledge Assistant")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("inventory", help="List PDF sources approved for ingestion.")
    subcommands.add_parser(
        "ingest", help="Extract and validate page-level records from approved PDFs."
    )
    return parser


def run_inventory(settings: Settings) -> None:
    sources = discover_pdf_sources(settings.source_dir)
    print(f"Found {len(sources)} PDF source document(s):")
    for source in sources:
        print(f"- {source.filename} ({source.size_bytes:,} bytes)")


def run_ingest(settings: Settings) -> None:
    report = run_ingestion(settings.source_dir, settings.data_dir)
    print(f"Extracted {report.total_pages} page(s) into {settings.data_dir}.")
    print(f"Validation findings: {report.error_count} error(s), {report.warning_count} warning(s).")


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_environment()

    if args.command == "inventory":
        run_inventory(settings)
    elif args.command == "ingest":
        run_ingest(settings)


if __name__ == "__main__":
    main()
