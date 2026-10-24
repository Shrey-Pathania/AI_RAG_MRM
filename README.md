# MRM Knowledge Assistant

A citation-first RAG assistant over a curated Model Risk Management document corpus.

## Structure

- `src/mrm_assistant/` — modular application code
- `src/mrm_assistant/main.py` — single application entry point and orchestration layer
- `src/mrm_assistant/ingestion/` — source discovery and PDF extraction modules
- `data/` — generated artifacts such as extracted pages and indexes (not source documents)
- `docs/adr/` — Architecture Decision Records (one per major design choice)
- `docs/runs/` — saved LLM outputs for evidence and reference

## First command

After installing project dependencies, verify that the intended source corpus is visible:

```bash
python -m mrm_assistant.main inventory
```

## Ingest source documents

Create page-level, citation-ready records and an extraction quality report:

```bash
python -m mrm_assistant.main ingest
```

Generated files are written to `data/` and deliberately excluded from Git. Run the
quality checks with `python -m pytest`.

## Create retrieval chunks

Split extracted pages into overlap-aware chunks while retaining the source document and
page number on every chunk:

```bash
python -m mrm_assistant.main chunk
```

## Create the local vector index

Generate embeddings for every chunk using the configured OpenAI-compatible provider.
This sends chunk text to that provider and writes a local index to `data/`.

```bash
python -m mrm_assistant.main index
```

## ChromaDB experiment

Store the existing embeddings, chunk text, and citation metadata in a persistent
local ChromaDB collection:

```bash
python -m mrm_assistant.main index-chroma
```

Search the collection with a question:

```bash
python -m mrm_assistant.main search-chroma "What is model validation?"
```
