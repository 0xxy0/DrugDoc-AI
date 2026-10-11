"""Application configuration for DrugDoc AI.

Configuration precedence:
    1. Environment variables (DRUGDOC_*)
    2. configs/config.yaml
    3. Defaults defined in Settings

Secrets must be provided through environment variables, never committed to YAML.

Notes for the hackathon build:
    - Retrieval now supports multi-query expansion, recursive retrieval, and
      Maximal Marginal Relevance (MMR) diversity reranking.
    - Generation now supports an optional self-correction / verification pass.
    - Ingestion supports both fixed-size and semantic chunking.
    - A few UI-facing flags (retrieval trace, answer comparison) are exposed
      here so app.py does not hardcode them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


LOGGER = logging.getLogger(__name__)

# config.py -> drugdoc/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application settings shared across DrugDoc AI."""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "DrugDoc AI"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    sample_labels_dir: Path = PROJECT_ROOT / "data" / "sample_labels"
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"
    reports_dir: Path = PROJECT_ROOT / "reports"

    evaluation_questions_file: Path = (
        PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
    )
    expected_evidence_file: Path = (
        PROJECT_ROOT / "data" / "evaluation" / "expected_evidence.jsonl"
    )

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    max_upload_size_mb: int = 25
    allowed_extensions: tuple[str, ...] = (".pdf",)
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    min_text_characters_per_page: int = 40

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    # "fixed"    -> overlap-aware sliding window (original behavior)
    # "semantic" -> paragraph / section-aware chunking with soft size caps
    chunking_strategy: str = "fixed"
    chunk_size: int = 900
    chunk_overlap: int = 150
    semantic_min_chunk_chars: int = 250
    semantic_max_chunk_chars: int = 1400

    # Optional chunk summarization used as an extra embedded field.
    chunk_summarization_enabled: bool = False
    chunk_summary_max_tokens: int = 80

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    dense_retrieval_enabled: bool = True
    sparse_retrieval_enabled: bool = True

    retrieval_top_k: int = 5
    dense_candidate_k: int = 10
    sparse_candidate_k: int = 10
    rrf_k: int = 60
    min_evidence_score: float = 0.0

    # Multi-query expansion: ask the LLM for N reformulations of the query.
    retrieval_multi_query_enabled: bool = True
    retrieval_multi_query_count: int = 3

    # Recursive retrieval: after the first pass, re-query using the top
    # passage if the initial candidate set looks thin.
    retrieval_recursive_enabled: bool = True
    retrieval_recursive_depth: int = 2
    retrieval_recursive_min_chunks: int = 3

    # MMR diversity reranking.
    retrieval_mmr_enabled: bool = True
    retrieval_mmr_lambda: float = 0.7

    # ------------------------------------------------------------------
    # Vector storage
    # ------------------------------------------------------------------
    vector_store_type: str = "qdrant_local"
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "drugdoc_labels"

    # ------------------------------------------------------------------
    # Language model
    # ------------------------------------------------------------------
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    llm_temperature: float = 0.0
    llm_max_tokens: int = 900
    llm_timeout_seconds: int = 60

    # ------------------------------------------------------------------
    # Answer safety & verification
    # ------------------------------------------------------------------
    abstention_enabled: bool = True
    require_citations: bool = True
    max_context_chunks: int = 5

    # Self-correction: after drafting, verify the answer is fully supported
    # by the retrieved evidence. If not, the pipeline abstains.
    self_correction_enabled: bool = True
    display_verifier_reason: bool = False
    sanitize_answer_text: bool = True

    # ------------------------------------------------------------------
    # UI-facing behavior
    # ------------------------------------------------------------------
    show_retrieval_trace: bool = True
    enable_answer_comparison: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Reject invalid settings before application startup."""

        if self.environment not in {"development", "test", "production"}:
            raise ValueError(
                "environment must be development, test, or production."
            )

        if self.log_level.upper() not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise ValueError(f"Unsupported log level: {self.log_level!r}")

        if self.max_upload_size_mb < 1:
            raise ValueError("max_upload_size_mb must be at least 1.")

        if self.min_text_characters_per_page < 0:
            raise ValueError(
                "min_text_characters_per_page cannot be negative."
            )

        if not self.allowed_extensions:
            raise ValueError("At least one allowed file extension is required.")

        # Chunking
        if self.chunking_strategy not in {"fixed", "semantic"}:
            raise ValueError(
                "chunking_strategy must be 'fixed' or 'semantic'."
            )

        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        if self.semantic_min_chunk_chars < 1:
            raise ValueError("semantic_min_chunk_chars must be positive.")

        if self.semantic_max_chunk_chars < self.semantic_min_chunk_chars:
            raise ValueError(
                "semantic_max_chunk_chars must be >= semantic_min_chunk_chars."
            )

        if self.chunk_summary_max_tokens < 1:
            raise ValueError("chunk_summary_max_tokens must be positive.")

        # Embeddings
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive.")

        # Retrieval
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive.")

        if self.dense_candidate_k < 1 or self.sparse_candidate_k < 1:
            raise ValueError("Retrieval candidate counts must be positive.")

        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive.")

        if self.min_evidence_score < 0:
            raise ValueError("min_evidence_score cannot be negative.")

        if self.retrieval_multi_query_count < 1:
            raise ValueError(
                "retrieval_multi_query_count must be at least 1."
            )

        if not 1 <= self.retrieval_recursive_depth <= 5:
            raise ValueError(
                "retrieval_recursive_depth must be between 1 and 5."
            )

        if self.retrieval_recursive_min_chunks < 1:
            raise ValueError(
                "retrieval_recursive_min_chunks must be at least 1."
            )

        if not 0.0 <= self.retrieval_mmr_lambda <= 1.0:
            raise ValueError(
                "retrieval_mmr_lambda must be between 0.0 and 1.0."
            )

        # Vector store
        if self.vector_store_type not in {"qdrant_local", "qdrant_server"}:
            raise ValueError(
                "vector_store_type must be 'qdrant_local' or 'qdrant_server'."
            )

        if self.vector_store_type == "qdrant_server" and not self.qdrant_url:
            raise ValueError("qdrant_url is required for qdrant_server.")

        # LLM
        if self.llm_provider not in {"ollama", "openai"}:
            raise ValueError("llm_provider must be 'ollama' or 'openai'.")

        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("llm_temperature must be between 0 and 2.")

        if self.llm_max_tokens < 1 or self.llm_timeout_seconds < 1:
            raise ValueError("LLM token limit and timeout must be positive.")

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when llm_provider='openai'."
            )

        # Safety
        if self.max_context_chunks < 1:
            raise ValueError("max_context_chunks must be positive.")

        if self.environment == "production" and self.debug:
            raise ValueError("debug must be disabled in production.")

        # Normalize extension list and log level.
        normalized_extensions = tuple(
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in self.allowed_extensions
        )

        object.__setattr__(self, "allowed_extensions", normalized_extensions)
        object.__setattr__(self, "log_level", self.log_level.upper())

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_production(self) -> bool:
        """Return whether the application runs in production mode."""
        return self.environment == "production"

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def _parse_value(value: Any, expected_type: Any, field_name: str) -> Any:
    """Convert YAML/environment values into the expected Python type."""

    if value is None:
        return None

    if expected_type is Path:
        return Path(value).expanduser()

    if expected_type is bool:
        if isinstance(value, bool):
            return value

        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False

        raise ValueError(
            f"{field_name} must be a boolean, received {value!r}."
        )

    if expected_type is int:
        if isinstance(value, bool):
            raise ValueError(
                f"{field_name} must be an integer, not a boolean."
            )
        return int(value)

    if expected_type is float:
        return float(value)

    if expected_type is str:
        return str(value)

    # Handle Optional[str] as str | None
    if expected_type == (str | None):
        return str(value)

    # Handle tuple[str, ...]
    if expected_type == tuple[str, ...]:
        if isinstance(value, str):
            return tuple(
                item.strip()
                for item in value.split(",")
                if item.strip()
            )

        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)

        raise ValueError(
            f"{field_name} must be a list or comma-separated string."
        )

    return value


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file, returning an empty mapping if absent."""

    if not config_path.is_file():
        LOGGER.info(
            "Configuration file not found at %s; using application defaults.",
            config_path,
        )
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration in {config_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping: {config_path}"
        )

    return data


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_settings(
    config_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Load settings with environment variables overriding YAML values.

    Args:
        config_path: Optional path to the YAML configuration.
        environ: Optional environment mapping, useful for testing.
            Defaults to the current process environment.

    Returns:
        A validated, immutable Settings instance.
    """

    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    raw_values = _load_yaml(path)
    known_fields = {item.name: item for item in fields(Settings)}

    unknown_keys = set(raw_values) - set(known_fields)
    if unknown_keys:
        # Unknown keys are warnings, not errors. This lets older YAML files
        # continue to work while we add new settings.
        LOGGER.warning(
            "Ignoring unknown configuration keys in %s: %s",
            path,
            ", ".join(sorted(unknown_keys)),
        )
        for key in unknown_keys:
            raw_values.pop(key, None)

    type_hints = get_type_hints(Settings)
    values: dict[str, Any] = {}

    for name, field_info in known_fields.items():
        if name in raw_values:
            raw_value = raw_values[name]
        elif field_info.default is not MISSING:
            raw_value = field_info.default
        elif field_info.default_factory is not MISSING:
            raw_value = field_info.default_factory()
        else:
            continue

        values[name] = _parse_value(raw_value, type_hints[name], name)

    environment = os.environ if environ is None else environ

    for name, expected_type in type_hints.items():
        env_name = f"DRUGDOC_{name.upper()}"

        # The API key also supports the conventional OPENAI_API_KEY variable.
        if name == "openai_api_key" and "OPENAI_API_KEY" in environment:
            values[name] = environment["OPENAI_API_KEY"]
        elif env_name in environment:
            values[name] = _parse_value(
                environment[env_name],
                expected_type,
                env_name,
            )

    # Resolve relative paths against the project root, not the working directory.
    path_fields = {
        "data_dir",
        "upload_dir",
        "sample_labels_dir",
        "evaluation_dir",
        "reports_dir",
        "qdrant_path",
        "evaluation_questions_file",
        "expected_evidence_file",
    }

    for name in path_fields:
        value = values.get(name)
        if isinstance(value, Path) and not value.is_absolute():
            values[name] = PROJECT_ROOT / value

    return Settings(**values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    The cached instance avoids repeatedly parsing YAML throughout the app.
    Clear the cache in tests when different settings are required.
    """
    return load_settings()