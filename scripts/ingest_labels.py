"""Ingest and index drug-label PDFs for DrugDoc AI.

Usage:
    python scripts/ingest_labels.py
    python scripts/ingest_labels.py --input-dir data/sample_labels
    python scripts/ingest_labels.py --input-dir data/sample_labels --recursive
    python scripts/ingest_labels.py --input-dir data/sample_labels --dry-run
    python scripts/ingest_labels.py --strategy semantic
    python scripts/ingest_labels.py --strategy fixed --chunk-size 1200 --chunk-overlap 200

The script uses the same ingestion and retrieval components as the application.
It reads the project settings and applies CLI overrides for the current run
without mutating the cached settings object.

Exit codes:
    0   All discovered PDFs were processed successfully.
    1   One or more files failed to ingest, but the run completed.
    2   Hard error (missing directory, no PDFs, or service initialization failed).
    130 Interrupted by Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# Support direct execution from the repository root without requiring an
# editable installation. Installed-package execution also remains supported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drugdoc.config import Settings, get_settings
from drugdoc.ingestion import IngestionError, ingest_pdf
from drugdoc.retrieval import HybridRetriever, RetrievalError
from drugdoc.storage import DocumentStore, StorageError

LOGGER = logging.getLogger("drugdoc.ingest")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(log_level: str) -> None:
    """Configure readable console logging for ingestion runs."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------

def discover_pdfs(input_dir: Path, recursive: bool) -> list[Path]:
    """Return sorted PDF files from the input directory."""

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()

    return sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: str(path).lower(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Ingest and index drug-label PDFs for DrugDoc AI."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.sample_labels_dir,
        help="Directory containing PDF labels.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for PDF files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Extract and validate PDFs without writing to storage or "
            "indexing."
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic"],
        default=None,
        help=(
            "Override the configured chunking strategy for this run. "
            "If omitted, settings.chunking_strategy is used."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk_size for this run (fixed strategy only).",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=(
            "Override chunk_overlap for this run (fixed strategy only). "
            "Must be smaller than --chunk-size."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=settings.log_level,
        help="Console logging verbosity.",
    )

    return parser


# ---------------------------------------------------------------------------
# Settings overrides
# ---------------------------------------------------------------------------

def apply_overrides(
    settings: Settings,
    *,
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Settings:
    """Return a new Settings instance with CLI overrides applied.

    Uses dataclasses.replace so the original cached settings object is
    never mutated. Validation in Settings.__post_init__ runs again on the
    new instance, so invalid combinations (e.g. overlap >= size) raise a
    clear error.
    """

    replacements: dict[str, object] = {}

    if strategy is not None:
        replacements["chunking_strategy"] = strategy

    if chunk_size is not None:
        replacements["chunk_size"] = chunk_size

    if chunk_overlap is not None:
        replacements["chunk_overlap"] = chunk_overlap

    if not replacements:
        return settings

    try:
        return replace(settings, **replacements)
    except ValueError as exc:
        raise SystemExit(f"Invalid settings override: {exc}") from exc


# ---------------------------------------------------------------------------
# Ingestion run
# ---------------------------------------------------------------------------

def run_ingestion(
    input_dir: Path,
    *,
    recursive: bool = False,
    dry_run: bool = False,
    settings: Settings | None = None,
) -> int:
    """Ingest the selected PDFs and return a process exit code."""

    base_settings = settings if settings is not None else get_settings()
    input_dir = input_dir.expanduser()

    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    try:
        pdf_paths = discover_pdfs(input_dir, recursive)
    except (FileNotFoundError, NotADirectoryError) as exc:
        LOGGER.error("%s", exc)
        return 2

    if not pdf_paths:
        LOGGER.error("No PDF files found in %s", input_dir)
        return 2

    LOGGER.info("Found %d PDF file(s) in %s", len(pdf_paths), input_dir)
    LOGGER.info(
        "Chunking strategy: %s", base_settings.chunking_strategy
    )

    if base_settings.chunking_strategy == "fixed":
        LOGGER.info(
            "Fixed chunk size=%d overlap=%d",
            base_settings.chunk_size,
            base_settings.chunk_overlap,
        )
    else:
        LOGGER.info(
            "Semantic chunk bounds min=%d max=%d",
            base_settings.semantic_min_chunk_chars,
            base_settings.semantic_max_chunk_chars,
        )

    # Initialize storage and retrieval only for real indexing runs.
    store: DocumentStore | None = None
    retriever: HybridRetriever | None = None

    if not dry_run:
        try:
            store = DocumentStore(base_settings)
            retriever = HybridRetriever(base_settings)
        except (StorageError, RetrievalError, OSError, ValueError) as exc:
            LOGGER.error(
                "Could not initialize ingestion services: %s", exc
            )
            return 2

    # Per-file summary rows: (name, pages, chunks, strategy, warnings)
    summary_rows: list[tuple[str, int, int, str, int]] = []

    succeeded = 0
    failed = 0
    total_chunks = 0
    total_pages = 0

    for pdf_path in pdf_paths:
        LOGGER.info("Processing %s", pdf_path.name)

        try:
            result = ingest_pdf(pdf_path, base_settings)

            LOGGER.info(
                "Extracted %d page(s) and %d chunk(s) from %s",
                result.page_count,
                result.chunk_count,
                pdf_path.name,
            )

            for warning in result.warnings:
                LOGGER.warning("%s: %s", pdf_path.name, warning)

            summary_rows.append(
                (
                    pdf_path.name,
                    result.page_count,
                    result.chunk_count,
                    base_settings.chunking_strategy,
                    len(result.warnings),
                )
            )

            total_pages += result.page_count

            if dry_run:
                succeeded += 1
                total_chunks += result.chunk_count
                continue

            assert store is not None
            assert retriever is not None

            # Index this document's chunks before marking its record indexed.
            # This ensures the metadata does not claim indexing succeeded
            # when the vector upsert failed.
            retriever.index_chunks(result.chunks)

            pdf_bytes = pdf_path.read_bytes()

            store.save_document(
                pdf_bytes,
                result,
                status="indexed",
            )

            succeeded += 1
            total_chunks += result.chunk_count

            LOGGER.info(
                "Indexed %s successfully (document_id=%s)",
                pdf_path.name,
                result.document_id,
            )

        except (
            IngestionError,
            RetrievalError,
            StorageError,
            OSError,
            ValueError,
        ) as exc:
            failed += 1
            LOGGER.error("Failed to process %s: %s", pdf_path.name, exc)

        except Exception:
            failed += 1
            LOGGER.exception(
                "Unexpected failure while processing %s",
                pdf_path.name,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(
        rows=summary_rows,
        succeeded=succeeded,
        failed=failed,
        total_pages=total_pages,
        total_chunks=total_chunks,
        dry_run=dry_run,
        strategy=base_settings.chunking_strategy,
    )

    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def _print_summary(
    *,
    rows: list[tuple[str, int, int, str, int]],
    succeeded: int,
    failed: int,
    total_pages: int,
    total_chunks: int,
    dry_run: bool,
    strategy: str,
) -> None:
    """Print the per-file table and totals."""

    LOGGER.info("=" * 72)
    LOGGER.info("INGESTION SUMMARY")
    LOGGER.info("=" * 72)

    if rows:
        header = (
            f"{'file':<32}"
            f"{'pages':>7}"
            f"{'chunks':>8}"
            f"{'strategy':>12}"
            f"{'warn':>6}"
        )
        LOGGER.info(header)
        LOGGER.info("-" * len(header))

        for name, pages, chunks, row_strategy, warnings_count in rows:
            display_name = name if len(name) <= 32 else name[:29] + "..."
            LOGGER.info(
                f"{display_name:<32}"
                f"{pages:>7}"
                f"{chunks:>8}"
                f"{row_strategy:>12}"
                f"{warnings_count:>6}"
            )

        LOGGER.info("-" * len(header))

    LOGGER.info("Files discovered : %d", succeeded + failed)
    LOGGER.info("Files succeeded  : %d", succeeded)
    LOGGER.info("Files failed     : %d", failed)
    LOGGER.info("Total pages      : %d", total_pages)
    LOGGER.info("Total chunks     : %d", total_chunks)
    LOGGER.info("Chunking strategy: %s", strategy)
    LOGGER.info("Mode             : %s", "DRY RUN" if dry_run else "INDEXING")
    LOGGER.info("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    base_settings = get_settings()

    try:
        settings = apply_overrides(
            base_settings,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except SystemExit as exc:
        # apply_overrides raises SystemExit on invalid combinations.
        LOGGER.error("%s", exc)
        return 2

    try:
        return run_ingestion(
            args.input_dir,
            recursive=args.recursive,
            dry_run=args.dry_run,
            settings=settings,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Ingestion interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())