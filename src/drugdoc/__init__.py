"""DrugDoc AI — evidence-grounded question answering for drug labels.

This package exposes a small, curated public surface. Importing the
package does NOT load heavy dependencies (sentence-transformers,
qdrant-client, PyMuPDF, ollama, openai, streamlit). Those are loaded
lazily by the modules that need them.

Typical usage:

    from drugdoc import (
        get_settings,
        DocumentStore,
        HybridRetriever,
        RAGPipeline,
    )

    settings = get_settings()
    store = DocumentStore(settings)
    retriever = HybridRetriever(settings)
    pipeline = RAGPipeline(retriever, settings)
"""

from __future__ import annotations

__version__ = "0.2.0"

# --- Configuration ---------------------------------------------------------
from drugdoc.config import (
    Settings,
    get_settings,
    load_settings,
)

# --- Ingestion -------------------------------------------------------------
from drugdoc.ingestion import (
    DocumentChunk,
    IngestionError,
    IngestionResult,
    PageMetadata,
    chunk_page_text,
    ingest_pdf,
    ingest_pdf_bytes,
    normalize_text,
)

# --- Retrieval -------------------------------------------------------------
from drugdoc.retrieval import (
    HybridRetriever,
    RetrievedChunk,
    RetrievalError,
    reciprocal_rank_fusion,
)

# --- RAG -------------------------------------------------------------------
from drugdoc.rag import (
    Citation,
    RAGError,
    RAGPipeline,
    RAGResponse,
)

# --- Storage ---------------------------------------------------------------
from drugdoc.storage import (
    DocumentRecord,
    DocumentStore,
    StorageError,
)

# --- Evaluation ------------------------------------------------------------
from drugdoc.evaluation import (
    EvaluationError,
    EvaluationQuestion,
    EvaluationReport,
    Evaluator,
    QuestionResult,
    load_questions,
)


__all__ = [
    "__version__",
    # Config
    "Settings",
    "get_settings",
    "load_settings",
    # Ingestion
    "DocumentChunk",
    "IngestionError",
    "IngestionResult",
    "PageMetadata",
    "chunk_page_text",
    "ingest_pdf",
    "ingest_pdf_bytes",
    "normalize_text",
    # Retrieval
    "HybridRetriever",
    "RetrievedChunk",
    "RetrievalError",
    "reciprocal_rank_fusion",
    # RAG
    "Citation",
    "RAGError",
    "RAGPipeline",
    "RAGResponse",
    # Storage
    "DocumentRecord",
    "DocumentStore",
    "StorageError",
    # Evaluation
    "EvaluationError",
    "EvaluationQuestion",
    "EvaluationReport",
    "Evaluator",
    "QuestionResult",
    "load_questions",
]