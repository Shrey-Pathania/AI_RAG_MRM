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
