"""Run the DrugDoc AI evaluation benchmark.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --questions data/evaluation/questions.jsonl
    python scripts/run_evaluation.py --top-k 5
    python scripts/run_evaluation.py --output reports/final-evaluation.json
    python scripts/run_evaluation.py --json-summary reports/latest-summary.json
    python scripts/run_evaluation.py --strict

The script rebuilds the retriever's in-memory chunk registry and BM25 index
from successfully stored PDFs before evaluating the benchmark. This ensures
the evaluation reflects the exact ingestion + retrieval configuration in
use right now.

Exit codes:
    0   Success (no failed questions, or --strict not set)
    1   Completed but at least one question failed, or --strict and any failure
    2   Hard error (missing dataset, ingestion failure, config problem)
    130 Interrupted by Ctrl+C
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drugdoc.config import get_settings
from drugdoc.evaluation import (
    EvaluationError,
    EvaluationReport,
    Evaluator,
    load_questions,
)
from drugdoc.ingestion import IngestionError, ingest_pdf
from drugdoc.rag import RAGError, RAGPipeline
from drugdoc.retrieval import HybridRetriever, RetrievalError
from drugdoc.storage import DocumentStore, StorageError

LOGGER = logging.getLogger("drugdoc.evaluate")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(log_level: str) -> None:
    """Configure console logging for evaluation runs."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""

    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Evaluate DrugDoc AI retrieval and answer quality."
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=settings.evaluation_questions_file,
        help="Path to the labelled JSONL benchmark.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retrieval_top_k,
        help="Number of passages to retrieve per question.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )

    parser.add_argument(
        "--json-summary",
        type=Path,
        default=None,
        help=(
            "Optional path for a compact summary JSON. Useful for "
            "dashboards and CI artifacts."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with code 1 if any individual question failed, even if "
            "the report was produced."
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
# Retriever rebuild
# ---------------------------------------------------------------------------

def rebuild_retriever(
    store: DocumentStore,
    retriever: HybridRetriever,
) -> tuple[int, int]:
    """Re-ingest stored PDFs to rebuild in-memory retrieval state.

    Only documents marked indexed are included. The document store is
    treated as the source of truth for which source files are eligible
    for evaluation.

    Returns:
        A tuple ``(document_count, total_chunks)``.
    """

    records = store.list_documents(status="indexed")

    if not records:
        raise EvaluationError(
            "No indexed documents were found in the document store. "
            "Run scripts/ingest_labels.py first."
        )

    total_chunks = 0
    failed_documents: list[str] = []

    for record in records:
        try:
            pdf_path = store.get_document_path(record.document_id)
            result = ingest_pdf(pdf_path)

            if result.document_id != record.document_id:
                raise EvaluationError(
                    f"Document ID mismatch for {record.filename}."
                )

            retriever.index_chunks(result.chunks)
            total_chunks += result.chunk_count

            LOGGER.info(
                "Loaded %s: %d chunks (strategy=%s)",
                record.filename,
                result.chunk_count,
                record.chunking_strategy,
            )

        except (IngestionError, RetrievalError, StorageError, OSError) as exc:
            failed_documents.append(f"{record.filename}: {exc}")
            LOGGER.error(
                "Could not prepare %s for evaluation: %s",
                record.filename,
                exc,
            )

    if failed_documents:
        raise EvaluationError(
            f"Failed to reconstruct {len(failed_documents)} indexed "
            "document(s). Evaluation was stopped to avoid silently "
            "benchmarking an incomplete corpus:\n- "
            + "\n- ".join(failed_documents)
        )

    if total_chunks == 0:
        raise EvaluationError(
            "No searchable chunks were reconstructed from indexed documents."
        )

    LOGGER.info(
        "Reconstructed %d chunks from %d indexed documents.",
        total_chunks,
        len(records),
    )

    return len(records), total_chunks


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 3) -> str:
    """Format a metric value for aligned console output."""

    if value is None:
        return "—"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def print_report_summary(report: EvaluationReport) -> None:
    """Print a human-readable summary, including per-difficulty metrics."""

    LOGGER.info("=" * 68)
    LOGGER.info("DRUGDOC AI — EVALUATION SUMMARY")
    LOGGER.info("=" * 68)
    LOGGER.info("Run ID               : %s", report.run_id)
    LOGGER.info("Created at           : %s", report.created_at)
    LOGGER.info("-" * 68)

    LOGGER.info(
        "Questions            : %d (answerable=%d, unanswerable=%d)",
        report.total_questions,
        report.answerable_questions,
        report.unanswerable_questions,
    )
    LOGGER.info(
        "Successful           : %d",
        report.successful_questions,
    )
    LOGGER.info(
        "Failed               : %d",
        report.failed_questions,
    )
    LOGGER.info(
        "Answered             : %d",
        report.answered_count,
    )
    LOGGER.info(
        "Abstained            : %d",
        report.abstained_count,
    )

    LOGGER.info("-" * 68)
    LOGGER.info("Retrieval & Citation Metrics")
    LOGGER.info("Recall@K             : %s", _fmt(report.recall_at_k))
    LOGGER.info("Precision@K          : %s", _fmt(report.precision_at_k))
    LOGGER.info(
        "Citation precision   : %s", _fmt(report.citation_precision)
    )
    LOGGER.info(
        "Citation recall      : %s", _fmt(report.citation_recall)
    )
    LOGGER.info(
        "Abstention accuracy  : %s", _fmt(report.abstention_accuracy)
    )
    LOGGER.info(
        "Answer token overlap : %s", _fmt(report.answer_token_overlap)
    )

    LOGGER.info("-" * 68)
    LOGGER.info("Latency & Safety")
    LOGGER.info(
        "Mean latency (ms)    : %s", _fmt(report.mean_latency_ms, 1)
    )
    LOGGER.info(
        "P50 latency (ms)     : %s", _fmt(report.p50_latency_ms, 1)
    )
    LOGGER.info(
        "P95 latency (ms)     : %s", _fmt(report.p95_latency_ms, 1)
    )
    LOGGER.info(
        "Self-correction hits : %d",
        report.self_correction_trigger_count,
    )

    if report.metrics_by_difficulty:
        LOGGER.info("-" * 68)
        LOGGER.info("Metrics by Difficulty")

        header = (
            f"{'difficulty':<14}"
            f"{'count':>7}"
            f"{'recall':>10}"
            f"{'precision':>11}"
            f"{'cit-P':>8}"
            f"{'cit-R':>8}"
            f"{'abstain':>9}"
            f"{'lat(ms)':>10}"
        )

        LOGGER.info(header)
        LOGGER.info("-" * len(header))

        for label, block in report.metrics_by_difficulty.items():
            LOGGER.info(
                f"{label:<14}"
                f"{_fmt(block.get('count')):>7}"
                f"{_fmt(block.get('recall_at_k')):>10}"
                f"{_fmt(block.get('precision_at_k')):>11}"
                f"{_fmt(block.get('citation_precision')):>8}"
                f"{_fmt(block.get('citation_recall')):>8}"
                f"{_fmt(block.get('abstention_accuracy')):>9}"
                f"{_fmt(block.get('mean_latency_ms'), 1):>10}"
            )

    LOGGER.info("=" * 68)


def build_compact_summary(report: EvaluationReport) -> dict[str, Any]:
    """Build a compact, human-readable summary for dashboards."""

    return {
        "run_id": report.run_id,
        "created_at": report.created_at,
        "totals": {
            "questions": report.total_questions,
            "successful": report.successful_questions,
            "failed": report.failed_questions,
            "answered": report.answered_count,
            "abstained": report.abstained_count,
        },
        "headline_metrics": {
            "recall_at_k": report.recall_at_k,
            "precision_at_k": report.precision_at_k,
            "citation_precision": report.citation_precision,
            "citation_recall": report.citation_recall,
            "abstention_accuracy": report.abstention_accuracy,
            "answer_token_overlap": report.answer_token_overlap,
            "self_correction_trigger_count": (
                report.self_correction_trigger_count
            ),
        },
        "latency_ms": {
            "mean": report.mean_latency_ms,
            "p50": report.p50_latency_ms,
            "p95": report.p95_latency_ms,
        },
        "metrics_by_difficulty": report.metrics_by_difficulty,
    }


def write_compact_summary(
    report: EvaluationReport,
    output_path: Path,
) -> Path:
    """Write a compact summary JSON to disk."""

    destination = output_path.expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = build_compact_summary(report)
    payload["written_at"] = datetime.now(UTC).isoformat()

    try:
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise EvaluationError(
            f"Could not write compact summary: {exc}"
        ) from exc

    return destination


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the evaluation and return an appropriate process exit code."""

    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    settings = get_settings()

    questions_path = args.questions.expanduser()
    if not questions_path.is_absolute():
        questions_path = PROJECT_ROOT / questions_path

    try:
        questions = load_questions(questions_path)

        LOGGER.info("Loaded %d benchmark question(s).", len(questions))

        store = DocumentStore(settings)
        retriever = HybridRetriever(settings)

        rebuild_retriever(store, retriever)

        pipeline = RAGPipeline(
            retriever=retriever,
            settings=settings,
        )

        evaluator = Evaluator(
            pipeline=pipeline,
            settings=settings,
        )

        report = evaluator.run(
            questions,
            top_k=args.top_k,
        )

        print_report_summary(report)

        output_path = args.output
        if output_path is not None:
            output_path = output_path.expanduser()
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / output_path

        saved_path = evaluator.save_report(
            report,
            output_path=output_path,
        )

        LOGGER.info("Full report written to: %s", saved_path)

        if args.json_summary is not None:
            summary_path = write_compact_summary(
                report, args.json_summary
            )
            LOGGER.info("Compact summary written to: %s", summary_path)

        # Exit code policy:
        #   * Any hard error above already raised and returned 2.
        #   * If --strict is set, any per-question failure exits 1.
        #   * Otherwise, we exit 1 only if every question failed.
        if args.strict and report.failed_questions:
            return 1

        if report.failed_questions == report.total_questions:
            return 1

        return 0

    except (
        EvaluationError,
        IngestionError,
        RetrievalError,
        RAGError,
        StorageError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("Evaluation could not complete: %s", exc)
        return 2

    except KeyboardInterrupt:
        LOGGER.warning("Evaluation interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())