"""Unit tests for DrugDoc AI retrieval-augmented generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import drugdoc.rag as rag_module
from drugdoc.rag import RAGPipeline
from drugdoc.retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    chunk_id: str = "chunk-a",
    *,
    document_id: str = "doc-a",
    filename: str = "label.pdf",
    page_number: int = 1,
    text: str = "The label lists important contraindications.",
    score: float = 0.05,
) -> RetrievedChunk:
    """Create a retrieved chunk without external dependencies."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        text=text,
        score=score,
        source="hybrid",
    )


def make_pipeline(
    chunks: list[RetrievedChunk] | None = None,
    *,
    settings_overrides: dict | None = None,
) -> tuple[RAGPipeline, Mock]:
    """Create a pipeline with mocked retrieval and lightweight settings."""
    settings: Any = SimpleNamespace(
        retrieval_top_k=5,
        max_context_chunks=5,
        require_citations=True,
        abstention_enabled=True,
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        openai_model="test-model",
        openai_api_key=None,
        llm_temperature=0.0,
        llm_max_tokens=100,
        llm_timeout_seconds=5,
        # New fields introduced in the hackathon build. Defaults keep
        # existing tests fast and deterministic.
        self_correction_enabled=False,
        enable_answer_comparison=False,
        show_retrieval_trace=True,
    )

    for name, value in (settings_overrides or {}).items():
        setattr(settings, name, value)

    retriever = Mock()
    retriever.retrieve.return_value = chunks or []

    pipeline = RAGPipeline(retriever=retriever, settings=settings)
    return pipeline, retriever


def fake_generation(
    *,
    answer: str = "The label lists contraindications.",
    supported_chunk_ids: list[str] | None = None,
    insufficient_evidence: bool = False,
) -> dict:
    """Build the structured response expected from the generation stage."""
    return {
        "answer": answer,
        "supported_chunk_ids": (
            ["chunk-a"]
            if supported_chunk_ids is None
            else supported_chunk_ids
        ),
        "insufficient_evidence": insufficient_evidence,
    }


def fake_verification(
    *,
    verdict: str = "supported",
    unsupported_claims: list[str] | None = None,
    reason: str = "Every claim is supported.",
) -> dict:
    """Build the structured response expected from the verification stage."""
    return {
        "verdict": verdict,
        "unsupported_claims": unsupported_claims or [],
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# No evidence and abstention
# ---------------------------------------------------------------------------


def test_pipeline_abstains_when_retrieval_returns_no_chunks() -> None:
    pipeline, retriever = make_pipeline([])

    response = pipeline.answer_question("What are the contraindications?")

    assert response.status != "answered"
    assert response.citations == []
    retriever.retrieve.assert_called_once()


def test_pipeline_passes_document_filter_to_retriever() -> None:
    pipeline, retriever = make_pipeline([])

    pipeline.answer_question(
        "What are the warnings?",
        document_ids={"doc-a"},
        top_k=3,
    )

    retriever.retrieve.assert_called_once_with(
        "What are the warnings?",
        top_k=3,
        document_ids={"doc-a"},
    )


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------


def test_pipeline_returns_citation_for_valid_supported_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = make_chunk()
    pipeline, _ = make_pipeline([chunk])

    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        lambda prompt, settings: fake_generation(
            supported_chunk_ids=["chunk-a"]
        ),
    )

    response = pipeline.answer_question("What does the label say?")

    assert response.status == "answered"
    assert response.citations
    assert response.citations[0].chunk_id == "chunk-a"
    assert response.citations[0].filename == "label.pdf"
    assert response.citations[0].page_number == 1


def test_pipeline_does_not_accept_unknown_citation_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = make_chunk()
    pipeline, _ = make_pipeline([chunk])

    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        lambda prompt, settings: fake_generation(
            supported_chunk_ids=["fabricated-chunk-id"]
        ),
    )

    response = pipeline.answer_question("What does the label say?")

    assert response.status != "answered"
    assert response.citations == []


def test_pipeline_abstains_when_model_reports_insufficient_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = make_chunk()
    pipeline, _ = make_pipeline([chunk])

    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        lambda prompt, settings: fake_generation(
            supported_chunk_ids=["chunk-a"],
            insufficient_evidence=True,
        ),
    )

    response = pipeline.answer_question("What does the label say?")

    assert response.status != "answered"


def test_pipeline_abstains_when_required_citations_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = make_chunk()
    pipeline, _ = make_pipeline([chunk])

    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        lambda prompt, settings: fake_generation(supported_chunk_ids=[]),
    )

    response = pipeline.answer_question("What does the label say?")

    assert response.status != "answered"
    assert response.citations == []


# ---------------------------------------------------------------------------
# Context and response metadata
# ---------------------------------------------------------------------------


def test_pipeline_respects_max_context_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        make_chunk(f"chunk-{index}", text=f"Evidence passage {index}.")
        for index in range(4)
    ]

    pipeline, _ = make_pipeline(
        chunks,
        settings_overrides={"max_context_chunks": 2},
    )

    captured: dict[str, str] = {}

    def fake_generate(prompt, settings):
        captured["prompt"] = prompt
        return fake_generation(supported_chunk_ids=["chunk-0"])

    monkeypatch.setattr(rag_module, "_generate_with_ollama", fake_generate)

    response = pipeline.answer_question("Summarize the label.")

    assert response.status == "answered"
    assert "Evidence passage 0." in captured["prompt"]
    assert "Evidence passage 1." in captured["prompt"]
    assert "Evidence passage 2." not in captured["prompt"]
    assert "Evidence passage 3." not in captured["prompt"]


def test_pipeline_records_retrieval_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _ = make_pipeline([make_chunk()])

    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        lambda prompt, settings: fake_generation(
            supported_chunk_ids=["chunk-a"]
        ),
    )

    response = pipeline.answer_question("What does the label say?")

    assert response.retrieval_source is not None


def test_response_to_dict_contains_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _ = make_pipeline([make_chunk()])

    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        lambda prompt, settings: fake_generation(
            supported_chunk_ids=["chunk-a"]
        ),
    )

    response = pipeline.answer_question("What does the label say?")
    data = response.to_dict()

    assert data["question"] == "What does the label say?"
    assert data["citations"]
    assert data["citations"][0]["chunk_id"] == "chunk-a"
    assert "status" in data
    assert "latency_ms" in data


# ---------------------------------------------------------------------------
# LLM failure handling
# ---------------------------------------------------------------------------


def test_pipeline_handles_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _ = make_pipeline([make_chunk()])

    def raise_generation_error(prompt, settings):
        raise RuntimeError("Simulated model failure")

    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        raise_generation_error,
    )

    with pytest.raises((rag_module.RAGError, RuntimeError)):
        pipeline.answer_question("What does the label say?")


def test_pipeline_uses_openai_provider_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _ = make_pipeline(
        [make_chunk()],
        settings_overrides={"llm_provider": "openai"},
    )

    called: dict[str, bool] = {"openai": False}

    def fake_openai(prompt, settings):
        called["openai"] = True
        return fake_generation(supported_chunk_ids=["chunk-a"])

    monkeypatch.setattr(rag_module, "_generate_with_openai", fake_openai)

    response = pipeline.answer_question("What does the label say?")

    assert response.status == "answered"
    assert called["openai"] is True


# ---------------------------------------------------------------------------
# Self-correction / verification (new in hackathon build)
# ---------------------------------------------------------------------------


def test_self_correction_pass_accepts_supported_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When verification returns 'supported', the answer is returned."""

    pipeline, _ = make_pipeline(
        [make_chunk()],
        settings_overrides={"self_correction_enabled": True},
    )

    call_log: list[str] = []

    def fake_generate(prompt, settings):
        call_log.append("generation")
        return fake_generation(supported_chunk_ids=["chunk-a"])

    def fake_verify(prompt, settings):
        call_log.append("verification")
        return fake_verification(verdict="supported")

    monkeypatch.setattr(rag_module, "_generate_with_ollama", fake_generate)
    monkeypatch.setattr(rag_module, "_generate_with_ollama", fake_generate)
    # Patch the verifier's provider too.
    monkeypatch.setattr(
        rag_module,
        "_generate_with_ollama",
        lambda prompt, settings: (
            fake_verify(prompt, settings)
            if "fact-checker" in prompt
            else fake_generate(prompt, settings)
        ),
    )

    response = pipeline.answer_question("What does the label say?")

    assert response.status == "answered"
    assert response.self_correction is not None
    assert response.self_correction["verdict"] == "supported"


def test_self_correction_pass_abstains_on_unsupported_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When verification returns 'unsupported', the pipeline abstains."""

    pipeline, _ = make_pipeline(
        [make_chunk()],
        settings_overrides={"self_correction_enabled": True},
    )

    def fake_generate(prompt, settings):
        return fake_generation(
            answer="The label recommends 500 mg daily.",
            supported_chunk_ids=["chunk-a"],
        )

    def fake_verifier(prompt, settings):
        return fake_verification(
            verdict="unsupported",
            unsupported_claims=["500 mg daily"],
            reason="The evidence does not mention a dosage.",
        )

    def dispatch(prompt, settings):
        if "fact-checker" in prompt:
            return fake_verifier(prompt, settings)
        return fake_generate(prompt, settings)

    monkeypatch.setattr(rag_module, "_generate_with_ollama", dispatch)

    response = pipeline.answer_question("What is the dosage?")

    assert response.status == "insufficient_evidence"
    assert response.abstention_reason is not None
    assert "self-correction" in response.abstention_reason.lower()
    assert response.self_correction is not None
    assert response.self_correction["verdict"] == "unsupported"


def test_self_correction_disabled_skips_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When disabled, no verification call is made."""

    pipeline, _ = make_pipeline(
        [make_chunk()],
        settings_overrides={"self_correction_enabled": False},
    )

    calls: list[str] = []

    def tracking_generate(prompt, settings):
        calls.append("generate")
        return fake_generation(supported_chunk_ids=["chunk-a"])

    monkeypatch.setattr(
        rag_module, "_generate_with_ollama", tracking_generate
    )

    response = pipeline.answer_question("What does the label say?")

    assert response.status == "answered"
    assert response.self_correction is None
    # Only the initial generation should have fired; no verification prompt.
    assert calls == ["generate"]