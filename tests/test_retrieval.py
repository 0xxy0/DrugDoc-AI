
"""Unit tests for DrugDoc AI retrieval and ranking."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

import pytest

import drugdoc.retrieval as retrieval_module
from drugdoc.ingestion import DocumentChunk
from drugdoc.retrieval import (
    HybridRetriever,
    RetrievedChunk,
    _tokenize,
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-a",
    filename: str = "label.pdf",
    page_number: int = 1,
    text: str = "The label contains safety and dosage information.",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def make_retrieved_chunk(
    chunk_id: str,
    *,
    score: float = 0.05,
    document_id: str = "doc-a",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="label.pdf",
        page_number=1,
        text=f"Evidence for {chunk_id}.",
        score=score,
        source="hybrid",
    )


def make_settings(**overrides):
    """Return the minimum settings needed for retrieval unit tests."""
    values = {
        "embedding_model": "BAAI/bge-m3",
        "embedding_device": "cpu",
        "embedding_batch_size": 2,
        "dense_retrieval_enabled": True,
        "sparse_retrieval_enabled": True,
        "retrieval_top_k": 5,
        "dense_candidate_k": 10,
        "sparse_candidate_k": 10,
        "rrf_k": 60,
        "vector_store_type": "qdrant_local",
        "qdrant_path": "unused-test-path",
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": None,
        "qdrant_collection": "test_labels",
    }
    values.update(overrides)
    return type("TestSettings", (), values)()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_tokenize_returns_lowercase_tokens() -> None:
    assert _tokenize("Drug Label SAFETY") == [
        "drug",
        "label",
        "safety",
    ]


def test_tokenize_handles_punctuation() -> None:
    tokens = _tokenize("Dose: 10 mg/day; warnings & precautions.")

    assert "dose" in tokens
    assert "warnings" in tokens
    assert "precautions" in tokens


def test_tokenize_handles_empty_text() -> None:
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


def test_rrf_combines_ranked_lists() -> None:
    fused = reciprocal_rank_fusion(
        [
            ["chunk-a", "chunk-b", "chunk-c"],
            ["chunk-b", "chunk-a", "chunk-d"],
        ],
        k=60,
    )

    assert set(fused) == {"chunk-a", "chunk-b", "chunk-c", "chunk-d"}
    assert fused["chunk-a"] > 0
    assert fused["chunk-b"] > 0


def test_rrf_rewards_documents_ranked_well_in_multiple_lists() -> None:
    fused = reciprocal_rank_fusion(
        [
            ["chunk-a", "chunk-b"],
            ["chunk-b", "chunk-a"],
        ],
        k=60,
    )

    assert fused["chunk-a"] == pytest.approx(fused["chunk-b"])


def test_rrf_returns_empty_result_for_empty_lists() -> None:
    assert reciprocal_rank_fusion([], k=60) == {}


def test_rrf_handles_empty_ranked_lists() -> None:
    fused = reciprocal_rank_fusion(
        [[], ["chunk-a", "chunk-b"], []],
        k=60,
    )

    assert list(fused) == ["chunk-a", "chunk-b"]


def test_rrf_uses_rank_positions_not_input_scores() -> None:
    fused = reciprocal_rank_fusion(
        [["chunk-a", "chunk-b"]],
        k=60,
    )

    assert fused["chunk-a"] > fused["chunk-b"]


def test_rrf_rejects_invalid_k() -> None:
    # k must be positive to avoid invalid reciprocal-rank calculations.
    with pytest.raises((ValueError, ZeroDivisionError)):
        reciprocal_rank_fusion([["chunk-a"]], k=0)


# ---------------------------------------------------------------------------
# Retrieved chunk representation
# ---------------------------------------------------------------------------


def test_retrieved_chunk_to_dict_preserves_metadata() -> None:
    chunk = make_retrieved_chunk("chunk-a")
    data = chunk.to_dict()

    assert data["chunk_id"] == "chunk-a"
    assert data["document_id"] == "doc-a"
    assert data["filename"] == "label.pdf"
    assert data["page_number"] == 1
    assert data["source"] == "hybrid"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_retrieved_chunk_has_expected_fields() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-a",
        document_id="doc-a",
        filename="label.pdf",
        page_number=3,
        text="Safety information.",
        score=0.02,
        dense_score=0.8,
        sparse_score=4.2,
        source="hybrid",
    )

    assert chunk.page_number == 3
    assert chunk.dense_score == pytest.approx(0.8)
    assert chunk.sparse_score == pytest.approx(4.2)


def test_rrf_empty_single_list() -> None:
    assert reciprocal_rank_fusion([[]], k=60) == {}
