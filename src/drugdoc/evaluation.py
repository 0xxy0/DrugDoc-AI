"""Reproducible evaluation for DrugDoc AI's retrieval and RAG pipeline.

Metrics covered:

    * Retrieval recall@k and precision@k
    * Citation precision and recall
    * Abstention accuracy
    * Latency statistics (mean / p50 / p95)
    * Per-difficulty breakdown (simple / complex / unanswerable / ...)
    * Optional answer token overlap against a reference answer

The report shape is stable and JSON-serializable. Additive fields may be
added over time; existing keys keep their name, type, and meaning.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from drugdoc.config import Settings, get_settings
from drugdoc.rag import RAGPipeline, RAGResponse

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EvaluationError(Exception):
    """Raised when an evaluation dataset or run cannot be processed."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALLOWED_DIFFICULTIES: frozenset[str] = frozenset(
    {"simple", "complex", "list", "synthesis", "unanswerable", "unspecified"}
)

_DIFFICULTY_ORDER: tuple[str, ...] = (
    "simple",
    "list",
    "complex",
    "synthesis",
    "unanswerable",
    "unspecified",
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """One benchmark question with its expected evidence."""

    question_id: str
    question: str
    expected_chunk_ids: tuple[str, ...] = ()
    expected_document_ids: tuple[str, ...] = ()
    answerable: bool = True
    reference_answer: str | None = None
    difficulty: str = "unspecified"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationQuestion":
        """Parse and validate one question record."""

        question_id = data.get("question_id")
        question = data.get("question")

        if not isinstance(question_id, str) or not question_id.strip():
            raise EvaluationError(
                "Every question needs a non-empty question_id."
            )

        if not isinstance(question, str) or not question.strip():
            raise EvaluationError(
                f"Question {question_id!r} needs a non-empty question."
            )

        expected_chunk_ids = data.get("expected_chunk_ids", [])
        expected_document_ids = data.get("expected_document_ids", [])
        answerable = data.get("answerable", True)
        reference_answer = data.get("reference_answer")
        difficulty_raw = data.get("difficulty", "unspecified")

        if not isinstance(expected_chunk_ids, list) or not all(
            isinstance(item, str) for item in expected_chunk_ids
        ):
            raise EvaluationError(
                f"Question {question_id!r}: expected_chunk_ids must be a "
                "list of strings."
            )

        if not isinstance(expected_document_ids, list) or not all(
            isinstance(item, str) for item in expected_document_ids
        ):
            raise EvaluationError(
                f"Question {question_id!r}: expected_document_ids must be "
                "a list of strings."
            )

        if not isinstance(answerable, bool):
            raise EvaluationError(
                f"Question {question_id!r}: answerable must be a boolean."
            )

        if reference_answer is not None and not isinstance(
            reference_answer, str
        ):
            raise EvaluationError(
                f"Question {question_id!r}: reference_answer must be text."
            )

        if answerable and not expected_chunk_ids and not expected_document_ids:
            raise EvaluationError(
                f"Answerable question {question_id!r} needs "
                "expected_chunk_ids or expected_document_ids."
            )

        difficulty = "unspecified"
        if isinstance(difficulty_raw, str):
            normalized = difficulty_raw.strip().lower()
            if normalized in _ALLOWED_DIFFICULTIES:
                difficulty = normalized
            elif normalized:
                # Unknown labels fall through to "unspecified" but are
                # preserved as a warning-worthy value.
                LOGGER.warning(
                    "Question %s uses unrecognized difficulty %r; "
                    "falling back to 'unspecified'.",
                    question_id,
                    difficulty_raw,
                )

        return cls(
            question_id=question_id.strip(),
            question=question.strip(),
            expected_chunk_ids=tuple(dict.fromkeys(expected_chunk_ids)),
            expected_document_ids=tuple(dict.fromkeys(expected_document_ids)),
            answerable=answerable,
            reference_answer=reference_answer,
            difficulty=difficulty,
        )


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Evaluation outcome for a single benchmark question."""

    question_id: str
    question: str
    expected_answerable: bool
    actual_status: str
    retrieved_chunk_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    recall_at_k: float | None
    precision_at_k: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_correct: bool
    latency_ms: int
    difficulty: str = "unspecified"
    answer_token_overlap: float | None = None
    self_correction_triggered: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate metrics and per-question results for a benchmark run."""

    run_id: str
    created_at: str
    total_questions: int
    successful_questions: int
    failed_questions: int
    answerable_questions: int
    unanswerable_questions: int
    answered_count: int
    abstained_count: int
    recall_at_k: float | None
    precision_at_k: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_accuracy: float | None
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    answer_token_overlap: float | None = None
    self_correction_trigger_count: int = 0
    metrics_by_difficulty: dict[str, dict[str, float | int | None]] = (
        field(default_factory=dict)
    )
    results: tuple[QuestionResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "total_questions": self.total_questions,
            "successful_questions": self.successful_questions,
            "failed_questions": self.failed_questions,
            "answerable_questions": self.answerable_questions,
            "unanswerable_questions": self.unanswerable_questions,
            "answered_count": self.answered_count,
            "abstained_count": self.abstained_count,
            "metrics": {
                "recall_at_k": self.recall_at_k,
                "precision_at_k": self.precision_at_k,
                "citation_precision": self.citation_precision,
                "citation_recall": self.citation_recall,
                "abstention_accuracy": self.abstention_accuracy,
                "mean_latency_ms": self.mean_latency_ms,
                "p50_latency_ms": self.p50_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "answer_token_overlap": self.answer_token_overlap,
                "self_correction_trigger_count": (
                    self.self_correction_trigger_count
                ),
            },
            "metrics_by_difficulty": self.metrics_by_difficulty,
            "results": [result.to_dict() for result in self.results],
        }


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: Iterable[str],
    expected_ids: Iterable[str],
    k: int,
) -> float | None:
    """Fraction of expected evidence items retrieved in the top k.

    Returns ``0.0`` when expected evidence is empty. Returns ``None`` only
    if the caller passes ``k < 1`` — callers should validate k upstream.
    """

    if k < 1:
        raise ValueError("k must be positive.")

    expected = set(expected_ids)

    if not expected:
        return 0.0

    retrieved = list(retrieved_ids)[:k]
    return len(set(retrieved) & expected) / len(expected)


def precision_at_k(
    retrieved_ids: Iterable[str],
    expected_ids: Iterable[str],
    k: int,
) -> float | None:
    """Fraction of the top-k retrieved items matching expected evidence."""

    if k < 1:
        raise ValueError("k must be positive.")

    retrieved = list(retrieved_ids)[:k]
    expected = set(expected_ids)

    if not retrieved or not expected:
        return 0.0

    return len(set(retrieved) & expected) / len(retrieved)


def citation_precision(
    cited_ids: Iterable[str],
    expected_ids: Iterable[str],
) -> float | None:
    """Fraction of cited chunks that match labelled supporting evidence."""

    cited = list(cited_ids)
    expected = set(expected_ids)

    if not cited or not expected:
        return 0.0

    return len(set(cited) & expected) / len(set(cited))


def citation_recall(
    cited_ids: Iterable[str],
    expected_ids: Iterable[str],
) -> float | None:
    """Fraction of labelled supporting chunks represented in citations."""

    cited = set(cited_ids)
    expected = set(expected_ids)

    if not expected:
        return 0.0

    return len(cited & expected) / len(expected)


def _percentile(values: list[int], percentile: float) -> float | None:
    """Calculate a linearly interpolated percentile without dependencies."""

    if not values:
        return None

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def _safe_mean(values: Iterable[float | None]) -> float | None:
    """Compute the mean over non-None values; return None if empty."""
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def _token_overlap(a: str | None, b: str | None) -> float | None:
    """Jaccard-ish overlap between two text answers.

    Returns the size of the intersection divided by the size of the union
    of lowercased alphanumeric tokens. Returns None if either side is empty.
    This is intentionally simple and dependency-free; it is used only as a
    rough signal in comparison-mode demos, not as a research metric.
    """

    if not isinstance(a, str) or not isinstance(b, str):
        return None

    tokens_a = {token.lower() for token in _TOKEN_RE.findall(a)}
    tokens_b = {token.lower() for token in _TOKEN_RE.findall(b)}

    if not tokens_a or not tokens_b:
        return None

    union = tokens_a | tokens_b
    intersection = tokens_a & tokens_b

    if not union:
        return None

    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    """Load the benchmark from JSON Lines format."""

    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise EvaluationError(
            f"Evaluation dataset does not exist: {dataset_path}"
        )

    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()

    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(
                        f"Invalid JSON at {dataset_path}:{line_number}: {exc}"
                    ) from exc

                if not isinstance(data, dict):
                    raise EvaluationError(
                        f"Each dataset line must be a JSON object "
                        f"(line {line_number})."
                    )

                question = EvaluationQuestion.from_dict(data)

                if question.question_id in seen_ids:
                    raise EvaluationError(
                        f"Duplicate question_id: {question.question_id!r}"
                    )

                seen_ids.add(question.question_id)
                questions.append(question)

    except OSError as exc:
        raise EvaluationError(
            f"Could not read evaluation dataset: {exc}"
        ) from exc

    if not questions:
        raise EvaluationError("The evaluation dataset contains no questions.")

    return questions


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Run a labelled benchmark against the actual DrugDoc RAG pipeline."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        settings: Settings | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        questions: list[EvaluationQuestion],
        *,
        top_k: int | None = None,
    ) -> EvaluationReport:
        """Run the benchmark and return aggregate and per-question metrics."""

        if not questions:
            # Empty benchmark is a no-op with a well-formed, empty report.
            return self._empty_report()

        results: list[QuestionResult] = []

        for item in questions:
            result = self._run_single(item, top_k=top_k)
            results.append(result)

        return self._build_report(questions, results)

    def save_report(
        self,
        report: EvaluationReport,
        output_path: str | Path | None = None,
    ) -> Path:
        """Persist an evaluation report as formatted JSON."""

        if output_path is None:
            output_dir = self.settings.reports_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / f"evaluation-{report.run_id}.json"
        else:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            destination.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise EvaluationError(
                f"Could not save evaluation report: {exc}"
            ) from exc

        return destination

    # ------------------------------------------------------------------
    # Per-question execution
    # ------------------------------------------------------------------

    def _run_single(
        self,
        item: EvaluationQuestion,
        *,
        top_k: int | None,
    ) -> QuestionResult:
        started_at = time.perf_counter()
        retrieved_ids: list[str] = []
        cited_ids: list[str] = []
        response: RAGResponse | None = None
        error: str | None = None

        document_filter: set[str] | None = (
            set(item.expected_document_ids)
            if item.expected_document_ids
            else None
        )

        try:
            retrieved = self.pipeline.retriever.retrieve(
                item.question,
                top_k=top_k,
                document_ids=document_filter,
            )
            retrieved_ids = [chunk.chunk_id for chunk in retrieved]

            response = self.pipeline.answer_question(
                item.question,
                top_k=top_k,
                document_ids=document_filter,
            )
            cited_ids = [
                citation.chunk_id for citation in response.citations
            ]

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception(
                "Evaluation failed for question %s", item.question_id
            )

        latency_ms = (
            response.latency_ms
            if response is not None
            else int((time.perf_counter() - started_at) * 1000)
        )

        actual_status = (
            response.status if response is not None else "error"
        )

        effective_k = top_k or self.settings.retrieval_top_k

        if item.answerable:
            expected_ids = set(item.expected_chunk_ids)

            if expected_ids:
                retrieval_recall = recall_at_k(
                    retrieved_ids, expected_ids, effective_k
                )
                retrieval_precision = precision_at_k(
                    retrieved_ids, expected_ids, effective_k
                )
                citation_p = citation_precision(cited_ids, expected_ids)
                citation_r = citation_recall(cited_ids, expected_ids)
            else:
                # Document-only labels cannot provide chunk-level metrics.
                retrieval_recall = None
                retrieval_precision = None
                citation_p = None
                citation_r = None
        else:
            retrieval_recall = None
            retrieval_precision = None
            citation_p = None
            citation_r = None

        expected_abstention = not item.answerable
        actual_abstention = actual_status == "insufficient_evidence"

        answer_overlap: float | None = None
        if (
            response is not None
            and item.reference_answer
            and response.status == "answered"
        ):
            answer_overlap = _token_overlap(
                response.answer, item.reference_answer
            )

        self_correction_triggered = False
        if response is not None and response.self_correction is not None:
            verdict = response.self_correction.get("verdict")
            self_correction_triggered = verdict == "unsupported"

        return QuestionResult(
            question_id=item.question_id,
            question=item.question,
            expected_answerable=item.answerable,
            actual_status=actual_status,
            retrieved_chunk_ids=tuple(retrieved_ids),
            cited_chunk_ids=tuple(cited_ids),
            recall_at_k=retrieval_recall,
            precision_at_k=retrieval_precision,
            citation_precision=citation_p,
            citation_recall=citation_r,
            abstention_correct=(
                actual_abstention == expected_abstention
                if actual_status != "error"
                else False
            ),
            latency_ms=latency_ms,
            difficulty=item.difficulty,
            answer_token_overlap=answer_overlap,
            self_correction_triggered=self_correction_triggered,
            error=error,
        )

    # ------------------------------------------------------------------
    # Report construction
    # ------------------------------------------------------------------

    def _build_report(
        self,
        questions: list[EvaluationQuestion],
        results: list[QuestionResult],
    ) -> EvaluationReport:
        successful_results = [
            result for result in results if result.error is None
        ]

        answerable_results = [
            result
            for result in successful_results
            if result.expected_answerable
        ]

        latency_values = [
            result.latency_ms for result in successful_results
        ]

        recall_values = [
            result.recall_at_k
            for result in answerable_results
            if result.recall_at_k is not None
        ]
        precision_values = [
            result.precision_at_k
            for result in answerable_results
            if result.precision_at_k is not None
        ]
        citation_precision_values = [
            result.citation_precision
            for result in answerable_results
            if result.citation_precision is not None
        ]
        citation_recall_values = [
            result.citation_recall
            for result in answerable_results
            if result.citation_recall is not None
        ]
        answer_overlap_values = [
            result.answer_token_overlap
            for result in successful_results
            if result.answer_token_overlap is not None
        ]

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        metrics_by_difficulty = self._aggregate_by_difficulty(
            successful_results
        )

        return EvaluationReport(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            total_questions=len(results),
            successful_questions=len(successful_results),
            failed_questions=len(results) - len(successful_results),
            answerable_questions=sum(
                item.answerable for item in questions
            ),
            unanswerable_questions=sum(
                not item.answerable for item in questions
            ),
            answered_count=sum(
                result.actual_status == "answered"
                for result in successful_results
            ),
            abstained_count=sum(
                result.actual_status == "insufficient_evidence"
                for result in successful_results
            ),
            recall_at_k=_safe_mean(recall_values),
            precision_at_k=_safe_mean(precision_values),
            citation_precision=_safe_mean(citation_precision_values),
            citation_recall=_safe_mean(citation_recall_values),
            abstention_accuracy=(
                mean(
                    float(result.abstention_correct)
                    for result in successful_results
                )
                if successful_results
                else None
            ),
            mean_latency_ms=_safe_mean(latency_values),
            p50_latency_ms=_percentile(latency_values, 50),
            p95_latency_ms=_percentile(latency_values, 95),
            answer_token_overlap=_safe_mean(answer_overlap_values),
            self_correction_trigger_count=sum(
                result.self_correction_triggered
                for result in successful_results
            ),
            metrics_by_difficulty=metrics_by_difficulty,
            results=tuple(results),
        )

    def _empty_report(self) -> EvaluationReport:
        """Return a well-formed report for an empty question set."""
        return EvaluationReport(
            run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            created_at=datetime.now(UTC).isoformat(),
            total_questions=0,
            successful_questions=0,
            failed_questions=0,
            answerable_questions=0,
            unanswerable_questions=0,
            answered_count=0,
            abstained_count=0,
            recall_at_k=None,
            precision_at_k=None,
            citation_precision=None,
            citation_recall=None,
            abstention_accuracy=None,
            mean_latency_ms=None,
            p50_latency_ms=None,
            p95_latency_ms=None,
            answer_token_overlap=None,
            self_correction_trigger_count=0,
            metrics_by_difficulty={},
            results=(),
        )

    def _aggregate_by_difficulty(
        self,
        results: list[QuestionResult],
    ) -> dict[str, dict[str, float | int | None]]:
        """Compute aggregate metrics grouped by question difficulty."""

        buckets: dict[str, list[QuestionResult]] = {
            label: [] for label in _DIFFICULTY_ORDER
        }

        for result in results:
            label = (
                result.difficulty
                if result.difficulty in buckets
                else "unspecified"
            )
            buckets[label].append(result)

        aggregate: dict[str, dict[str, float | int | None]] = {}

        for label in _DIFFICULTY_ORDER:
            items = buckets[label]

            if not items:
                continue

            answered = [r for r in items if r.actual_status == "answered"]
            abstained = [
                r for r in items if r.actual_status == "insufficient_evidence"
            ]

            aggregate[label] = {
                "count": len(items),
                "answered": len(answered),
                "abstained": len(abstained),
                "recall_at_k": _safe_mean(
                    r.recall_at_k for r in items
                ),
                "precision_at_k": _safe_mean(
                    r.precision_at_k for r in items
                ),
                "citation_precision": _safe_mean(
                    r.citation_precision for r in items
                ),
                "citation_recall": _safe_mean(
                    r.citation_recall for r in items
                ),
                "abstention_accuracy": (
                    mean(float(r.abstention_correct) for r in items)
                    if items
                    else None
                ),
                "mean_latency_ms": _safe_mean(
                    r.latency_ms for r in items
                ),
            }

        return aggregate


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "EvaluationError",
    "EvaluationQuestion",
    "QuestionResult",
    "EvaluationReport",
    "Evaluator",
    "load_questions",
    "recall_at_k",
    "precision_at_k",
    "citation_precision",
    "citation_recall",
    "_percentile",
]