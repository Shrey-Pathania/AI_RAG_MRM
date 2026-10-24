"""Central, typed configuration for the application."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Paths and optional LLM settings shared by all modules."""

    project_root: Path
    source_dir: Path
    data_dir: Path
    chroma_dir: Path
    openai_api_key: str | None
    openai_base_url: str | None
    embedding_model: str

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> "Settings":
        load_dotenv()
        root = project_root or Path(__file__).resolve().parents[2]
        return cls(
            project_root=root,
            source_dir=root / "docs",
            data_dir=root / "data",
            chroma_dir=root / "data" / "chroma",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
