
"""Unit tests for DrugDoc AI evaluation metrics and evaluator behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from drugdoc.evaluation import (
    EvaluationQuestion,
    Evaluator,
    _percentile,
    citation_precision,
    citation_recall,
    load_questions,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def test_recall_at_k_returns_fraction_of_relevant_chunks_found() -> None:
    retrieved = ["chunk-a", "chunk-b", "chunk-c"]
    expected = ["chunk-b", "chunk-c", "chunk-d"]

    assert recall_at_k(retrieved, expected, k=2) == pytest.approx(1 / 3)


def test_recall_at_k_returns_one_when_all_expected_chunks_are_found() -> None:
    assert recall_at_k(
        ["chunk-a", "chunk-b"],
        ["chunk-a", "chunk-b"],
        k=2,
    ) == pytest.approx(1.0)


def test_recall_at_k_returns_zero_when_no_relevant_chunks_are_found() -> None:
    assert recall_at_k(["chunk-a", "chunk-b"], ["chunk-x"], k=2) == 0.0


def test_recall_at_k_handles_empty_expected_chunks() -> None:
    # An unanswerable question has no positive chunk-recall target.
    assert recall_at_k(["chunk-a"], [], k=5) == 0.0


def test_recall_at_k_respects_k() -> None:
    assert recall_at_k(
        ["chunk-a", "chunk-b", "chunk-c"],
        ["chunk-c"],
        k=2,
    ) == 0.0


def test_precision_at_k_returns_fraction_of_retrieved_chunks_that_are_relevant() -> None:
    assert precision_at_k(
        ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
        ["chunk-b", "chunk-d"],
        k=4,
    ) == pytest.approx(0.5)


def test_precision_at_k_respects_k() -> None:
    assert precision_at_k(
        ["chunk-a", "chunk-b", "chunk-c"],
        ["chunk-c"],
        k=2,
    ) == 0.0


def test_precision_at_k_returns_zero_for_empty_retrieval() -> None:
    assert precision_at_k([], ["chunk-a"], k=5) == 0.0


def test_retrieval_metrics_handle_duplicate_ids_without_double_counting() -> None:
    # The expected set contains one relevant ID. A duplicate result should not
    # count as retrieving additional relevant evidence.
    assert recall_at_k(
        ["chunk-a", "chunk-a"],
        ["chunk-a"],
        k=2,
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------


def test_citation_precision_counts_supported_citations() -> None:
    assert citation_precision(
        ["chunk-a", "chunk-b", "chunk-x"],
        ["chunk-a", "chunk-b"],
    ) == pytest.approx(2 / 3)


def test_citation_recall_counts_expected_evidence_cited() -> None:
    assert citation_recall(
        ["chunk-a", "chunk-b"],
        ["chunk-a", "chunk-b", "chunk-c"],
    ) == pytest.approx(2 / 3)


def test_citation_precision_handles_no_citations() -> None:
    assert citation_precision([], ["chunk-a"]) == 0.0


def test_citation_recall_handles_empty_expected_evidence() -> None:
    assert citation_recall(["chunk-a"], []) == 0.0


# ---------------------------------------------------------------------------
# Evaluation question parsing
# ---------------------------------------------------------------------------


def test_evaluation_question_from_dict_parses_expected_fields() -> None:
    question = EvaluationQuestion.from_dict(
        {
            "question_id": "Q001",
            "question": "What contraindications are listed?",
            "expected_chunk_ids": ["chunk-a", "chunk-b"],
            "expected_document_ids": ["doc-a"],
            "answerable": True,
            "reference_answer": "See the contraindications section.",
        }
    )

    assert question.question_id == "Q001"
    assert question.question == "What contraindications are listed?"
    assert question.expected_chunk_ids == ("chunk-a", "chunk-b")
    assert question.expected_document_ids == ("doc-a",)
    assert question.answerable is True
    assert question.reference_answer == "See the contraindications section."


def test_evaluation_question_defaults_optional_fields() -> None:
    question = EvaluationQuestion.from_dict(
        {
            "question_id": "Q002",
            "question": "What is listed in this label?",
            "answerable": False,
        }
    )

    assert question.expected_chunk_ids == ()
    assert question.expected_document_ids == ()
    assert question.answerable is False
    assert question.reference_answer is None


def test_load_questions_reads_jsonl_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.jsonl"
    records = [
        {
            "question_id": "Q001",
            "question": "What is the dosage?",
            "expected_chunk_ids": ["chunk-a"],
            "answerable": True,
        },
        {
            "question_id": "Q002",
            "question": "What color is the patient's room?",
            "expected_chunk_ids": [],
            "answerable": False,
        },
    ]

    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    questions = load_questions(dataset_path)

    assert len(questions) == 2
    assert questions[0].question_id == "Q001"
    assert questions[1].answerable is False


def test_load_questions_rejects_malformed_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        '{"question_id": "Q001"}\nnot-valid-json\n',
        encoding="utf-8",
    )

    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_questions(dataset_path)


# ---------------------------------------------------------------------------
# Evaluator behavior
# ---------------------------------------------------------------------------


def test_evaluator_runs_questions_and_returns_report() -> None:
    """Verify evaluator orchestration without loading models or vector stores."""
    pipeline = Mock()

    pipeline.retriever.retrieve.return_value = [
        SimpleNamespace(
            chunk_id="chunk-a",
            document_id="doc-a",
            filename="label.pdf",
            page_number=1,
            text="The label lists the relevant information.",
            score=0.05,
            source="hybrid",
        )
    ]

    pipeline.answer_question.return_value = SimpleNamespace(
        question="What does the label say?",
        answer="The label lists the relevant information.",
        status="answered",
        citations=[
            SimpleNamespace(
                chunk_id="chunk-a",
                citation_id="C1",
                document_id="doc-a",
                filename="label.pdf",
                page_number=1,
                quote="The label lists the relevant information.",
            )
        ],
        evidence_count=1,
        latency_ms=10.0,
        abstention_reason=None,
        retrieval_source="hybrid",
        warnings=[],
    )

    evaluator = Evaluator(pipeline=pipeline, settings=Mock())

    question = EvaluationQuestion(
        question_id="Q001",
        question="What does the label say?",
        expected_chunk_ids=("chunk-a",),
        expected_document_ids=(),
        answerable=True,
        reference_answer=None,
    )

    report = evaluator.run([question])

    assert report is not None
    pipeline.answer_question.assert_called_once()


def test_evaluator_handles_empty_question_list() -> None:
    pipeline = Mock()
    evaluator = Evaluator(pipeline=pipeline, settings=Mock())

    report = evaluator.run([])

    assert report is not None
    pipeline.answer_question.assert_not_called()
