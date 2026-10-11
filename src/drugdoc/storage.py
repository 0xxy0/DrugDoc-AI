"""Persistent document and metadata storage for DrugDoc AI.

The store keeps:

    * Original PDFs on disk under ``settings.upload_dir``.
    * Document metadata in a SQLite database at
      ``settings.data_dir / "drugdoc.sqlite3"``.

Schema upgrades are applied automatically at startup, so an existing
database continues to work after new columns are introduced.

Vector indexing remains the responsibility of HybridRetriever. Documents
are only marked ``indexed`` after the retriever confirms the upsert.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drugdoc.config import Settings, get_settings
from drugdoc.ingestion import IngestionResult

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class StorageError(Exception):
    """Raised when document persistence or metadata operations fail."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Persistent metadata for an ingested drug-label PDF."""

    document_id: str
    filename: str
    file_path: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    created_at: str
    warnings: tuple[str, ...] = ()
    chunking_strategy: str = "fixed"
    embedding_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document record."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "created_at": self.created_at,
            "warnings": list(self.warnings),
            "chunking_strategy": self.chunking_strategy,
            "embedding_model": self.embedding_model,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALLOWED_STATUSES: frozenset[str] = frozenset(
    {"uploaded", "ingested", "indexed", "failed"}
)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class DocumentStore:
    """Store original PDFs and document metadata.

    The metadata database is SQLite. Vector indexing remains the
    responsibility of HybridRetriever; document records contain stable
    document IDs that can be used to filter retrieval results.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.RLock()

        self.upload_dir = self.settings.upload_dir.resolve()
        self.database_path = (
            self.settings.data_dir.resolve() / "drugdoc.sqlite3"
        )

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection with safe operational defaults."""
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize_database(self) -> None:
        """Create or upgrade the document metadata schema."""
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size_bytes INTEGER NOT NULL,
                        page_count INTEGER NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        warnings_json TEXT NOT NULL DEFAULT '[]'
                    )
                    """
                )

                # Additive migrations. Each column is added only if missing.
                existing_columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(documents)"
                    ).fetchall()
                }

                if "chunking_strategy" not in existing_columns:
                    connection.execute(
                        "ALTER TABLE documents "
                        "ADD COLUMN chunking_strategy TEXT NOT NULL "
                        "DEFAULT 'fixed'"
                    )

                if "embedding_model" not in existing_columns:
                    connection.execute(
                        "ALTER TABLE documents "
                        "ADD COLUMN embedding_model TEXT NOT NULL "
                        "DEFAULT ''"
                    )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_documents_created_at
                    ON documents(created_at DESC)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status)
                    """
                )

        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not initialize document metadata database: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_document(
        self,
        pdf_bytes: bytes,
        result: IngestionResult,
        *,
        status: str = "indexed",
    ) -> DocumentRecord:
        """Persist a successfully ingested PDF and its metadata.

        The caller should invoke this only after document ingestion and
        vector indexing have succeeded. A temporary file and atomic rename
        avoid exposing partially written PDFs.
        """

        if not pdf_bytes:
            raise StorageError("Cannot store an empty PDF.")

        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"Unsupported document status: {status!r}")

        if hashlib.sha256(pdf_bytes).hexdigest()[:24] != result.document_id:
            raise StorageError(
                "The PDF content does not match the ingestion result's "
                "document ID."
            )

        filename = Path(result.filename.replace("\\", "/")).name

        if not filename or filename in {".", ".."}:
            raise StorageError("Invalid document filename.")

        destination = self.upload_dir / f"{result.document_id}.pdf"

        if destination.resolve().parent != self.upload_dir:
            raise StorageError("Invalid upload destination.")

        timestamp = datetime.now(UTC).isoformat()
        temporary_path: Path | None = None

        chunking_strategy = getattr(
            self.settings, "chunking_strategy", "fixed"
        )
        embedding_model = getattr(
            self.settings, "embedding_model", ""
        )

        try:
            with self._lock:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.upload_dir,
                    prefix=".drugdoc-",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(pdf_bytes)
                    temporary_file.flush()
                    temporary_path = Path(temporary_file.name)

                temporary_path.replace(destination)

                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO documents (
                            document_id,
                            filename,
                            file_path,
                            file_size_bytes,
                            page_count,
                            chunk_count,
                            status,
                            created_at,
                            warnings_json,
                            chunking_strategy,
                            embedding_model
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id) DO UPDATE SET
                            filename = excluded.filename,
                            file_path = excluded.file_path,
                            file_size_bytes = excluded.file_size_bytes,
                            page_count = excluded.page_count,
                            chunk_count = excluded.chunk_count,
                            status = excluded.status,
                            warnings_json = excluded.warnings_json,
                            chunking_strategy = excluded.chunking_strategy,
                            embedding_model = excluded.embedding_model
                        """,
                        (
                            result.document_id,
                            filename,
                            str(destination),
                            len(pdf_bytes),
                            result.page_count,
                            result.chunk_count,
                            status,
                            timestamp,
                            json.dumps(result.warnings),
                            chunking_strategy,
                            embedding_model,
                        ),
                    )

        except (OSError, sqlite3.Error) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

            raise StorageError(
                f"Could not persist document {filename!r}: {exc}"
            ) from exc

        LOGGER.info(
            "Saved document %s (%d bytes, %d chunks, strategy=%s).",
            result.document_id,
            len(pdf_bytes),
            result.chunk_count,
            chunking_strategy,
        )

        record = self.get_document(result.document_id)

        if record is None:
            raise StorageError(
                "Document was saved but its metadata record could not be read."
            )

        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document record, or None when it does not exist."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT *
                    FROM documents
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchone()

        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not retrieve document metadata: {exc}"
            ) from exc

        return self._row_to_record(row) if row else None

    def get_document_by_hash(
        self,
        pdf_sha256_prefix: str,
    ) -> DocumentRecord | None:
        """Look up a document by its 24-character SHA-256 prefix.

        This is the same prefix used as the document ID, so it is useful
        for de-duplication on re-upload and for the "discover insights"
        flow that may need to resolve a hash to a stored record.
        """

        if not isinstance(pdf_sha256_prefix, str):
            return None

        prefix = pdf_sha256_prefix.strip().lower()

        if not prefix:
            return None

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT *
                    FROM documents
                    WHERE document_id = ?
                    """,
                    (prefix,),
                ).fetchone()

        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not retrieve document metadata by hash: {exc}"
            ) from exc

        return self._row_to_record(row) if row else None

    def list_documents(
        self,
        *,
        status: str | None = None,
    ) -> list[DocumentRecord]:
        """Return document records, newest first."""
        try:
            with self._connect() as connection:
                if status is None:
                    rows = connection.execute(
                        """
                        SELECT *
                        FROM documents
                        ORDER BY created_at DESC, document_id
                        """
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT *
                        FROM documents
                        WHERE status = ?
                        ORDER BY created_at DESC, document_id
                        """,
                        (status,),
                    ).fetchall()

        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not list document metadata: {exc}"
            ) from exc

        return [self._row_to_record(row) for row in rows]

    def list_document_ids(
        self,
        *,
        status: str | None = "indexed",
    ) -> list[str]:
        """Return only the document IDs, optionally filtered by status.

        Cheaper than ``list_documents`` when callers only need IDs, for
        example to pass a document filter to the retriever or the
        evaluation pipeline.
        """

        try:
            with self._connect() as connection:
                if status is None:
                    rows = connection.execute(
                        """
                        SELECT document_id
                        FROM documents
                        ORDER BY created_at DESC, document_id
                        """
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT document_id
                        FROM documents
                        WHERE status = ?
                        ORDER BY created_at DESC, document_id
                        """,
                        (status,),
                    ).fetchall()

        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not list document IDs: {exc}"
            ) from exc

        return [row["document_id"] for row in rows]

    def get_document_path(self, document_id: str) -> Path:
        """Resolve a stored PDF path without trusting user-supplied paths."""
        record = self.get_document(document_id)

        if record is None:
            raise StorageError(f"Unknown document ID: {document_id}")

        path = Path(record.file_path).resolve()

        if path.parent != self.upload_dir or not path.is_file():
            raise StorageError(
                f"The stored PDF is missing or its path is invalid: "
                f"{document_id}"
            )

        return path

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(
        self,
        document_id: str,
        *,
        delete_file: bool = True,
    ) -> bool:
        """Delete document metadata and optionally the stored PDF.

        Returns False if the document does not exist.

        Note: vector deletion must also be performed by the retrieval layer
        before calling this method if the document has been indexed.
        """

        with self._lock:
            record = self.get_document(document_id)

            if record is None:
                return False

            path = (
                self.get_document_path(document_id)
                if delete_file
                else None
            )

            try:
                with self._connect() as connection:
                    connection.execute(
                        "DELETE FROM documents WHERE document_id = ?",
                        (document_id,),
                    )

                if path is not None:
                    path.unlink(missing_ok=True)

            except (sqlite3.Error, OSError) as exc:
                raise StorageError(
                    f"Could not delete document {document_id}: {exc}"
                ) from exc

        LOGGER.info("Deleted document metadata: %s", document_id)
        return True

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        """Convert a database row into a typed document record.

        Tolerant of rows produced by older schema versions that did not
        yet include the additive columns.
        """

        row_keys = set(row.keys())

        try:
            warnings = json.loads(row["warnings_json"])
        except (TypeError, json.JSONDecodeError):
            warnings = []

        if not isinstance(warnings, list):
            warnings = []

        chunking_strategy = (
            row["chunking_strategy"]
            if "chunking_strategy" in row_keys
            else "fixed"
        )

        embedding_model = (
            row["embedding_model"]
            if "embedding_model" in row_keys
            else ""
        )

        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            file_path=row["file_path"],
            file_size_bytes=row["file_size_bytes"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            status=row["status"],
            created_at=row["created_at"],
            warnings=tuple(str(item) for item in warnings),
            chunking_strategy=str(chunking_strategy or "fixed"),
            embedding_model=str(embedding_model or ""),
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "StorageError",
    "DocumentRecord",
    "DocumentStore",
]