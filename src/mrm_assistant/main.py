"""Single application entry point that orchestrates independent modules."""

import argparse

from mrm_assistant.config import Settings
from mrm_assistant.chunking.pipeline import run_chunking
from mrm_assistant.ingestion.discovery import discover_pdf_sources
from mrm_assistant.ingestion.pipeline import run_ingestion
from mrm_assistant.indexing.chroma_pipeline import run_chroma_indexing
from mrm_assistant.indexing.pipeline import run_indexing
from mrm_assistant.retrieval.chroma_search import search_chroma


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MRM Knowledge Assistant")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("inventory", help="List PDF sources approved for ingestion.")
    subcommands.add_parser(
        "ingest", help="Extract and validate page-level records from approved PDFs."
    )
    subcommands.add_parser(
        "chunk", help="Create citation-preserving retrieval chunks from extracted pages."
    )
    subcommands.add_parser(
        "index", help="Create a local vector index from retrieval chunks."
    )
    subcommands.add_parser(
        "index-chroma", help="Store existing vectors and metadata in persistent ChromaDB."
    )
    search_parser = subcommands.add_parser(
        "search-chroma", help="Search the ChromaDB collection with a question."
    )
    search_parser.add_argument("question", help="Question to search for.")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
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


def run_chunk(settings: Settings) -> None:
    chunk_count = run_chunking(settings.data_dir)
    print(f"Created {chunk_count} retrieval chunk(s) in {settings.data_dir}.")


def run_index(settings: Settings) -> None:
    record_count = run_indexing(
        data_dir=settings.data_dir,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        embedding_model=settings.embedding_model,
    )
    print(f"Created vector index for {record_count} chunk(s) in {settings.data_dir}.")


def run_index_chroma(settings: Settings) -> None:
    record_count = run_chroma_indexing(settings.data_dir, settings.chroma_dir)
    print(f"Stored {record_count} chunk(s) in Chroma at {settings.chroma_dir}.")


def run_search_chroma(settings: Settings, question: str, top_k: int) -> None:
    results = search_chroma(
        question=question,
        chroma_dir=settings.chroma_dir,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        embedding_model=settings.embedding_model,
        top_k=top_k,
    )
    for rank, result in enumerate(results, start=1):
        print(
            f"[{rank}] distance={result.distance:.4f} "
            f"{result.metadata['document_name']} p.{result.metadata['page_number']}"
        )
        print(result.text[:500])
        print()


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_environment()

    if args.command == "inventory":
        run_inventory(settings)
    elif args.command == "ingest":
        run_ingest(settings)
    elif args.command == "chunk":
        run_chunk(settings)
    elif args.command == "index":
        run_index(settings)
    elif args.command == "index-chroma":
        run_index_chroma(settings)
    elif args.command == "search-chroma":
        run_search_chroma(settings, args.question, args.top_k)


if __name__ == "__main__":
    main()
