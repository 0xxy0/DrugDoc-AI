
"""Unit tests for DrugDoc AI document ingestion."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import fitz
import pytest

from drugdoc.config import Settings
from drugdoc.ingestion import (
    IngestionError,
    chunk_page_text,
    ingest_pdf,
    ingest_pdf_bytes,
    normalize_text,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_test_pdf(
    pages: list[str],
    *,
    filename: str = "test-label.pdf",
) -> tuple[bytes, str]:
    """Create a small in-memory PDF with one text page per input string."""
    buffer = io.BytesIO()

    with fitz.open() as document:
        for page_text in pages:
            page = document.new_page()
            page.insert_textbox(
                fitz.Rect(50, 50, 550, 750),
                page_text,
                fontsize=11,
            )

        pdf_bytes = document.tobytes()

    return pdf_bytes, filename


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Provide isolated settings for ingestion tests."""
    return Settings(
        app_name="DrugDoc AI Test",
        environment="test",
        debug=False,
        log_level="ERROR",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        sample_labels_dir=tmp_path / "data" / "sample_labels",
        evaluation_dir=tmp_path / "data" / "evaluation",
        reports_dir=tmp_path / "reports",
        max_upload_size_mb=1,
        allowed_extensions=(".pdf",),
        ocr_enabled=False,
        ocr_language="eng",
        min_text_characters_per_page=40,
        chunk_size=120,
        chunk_overlap=20,
        embedding_model="BAAI/bge-m3",
        embedding_device="cpu",
        embedding_batch_size=2,
        dense_retrieval_enabled=True,
        sparse_retrieval_enabled=True,
        retrieval_top_k=5,
        dense_candidate_k=10,
        sparse_candidate_k=10,
        rrf_k=60,
        min_evidence_score=0.0,
        vector_store_type="qdrant_local",
        qdrant_path=tmp_path / "data" / "qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_collection="test_labels",
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        openai_model="gpt-4o-mini",
        openai_api_key=None,
        llm_temperature=0.0,
        llm_max_tokens=100,
        llm_timeout_seconds=5,
        abstention_enabled=True,
        require_citations=True,
        max_context_chunks=3,
        evaluation_questions_file=tmp_path / "data" / "evaluation" / "questions.jsonl",
        expected_evidence_file=(
            tmp_path / "data" / "evaluation" / "expected_evidence.jsonl"
        ),
    )


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def test_normalize_text_repairs_hyphenated_line_breaks() -> None:
    text = "contraindi-\ncations are listed"

    assert normalize_text(text) == "contraindications are listed"


def test_normalize_text_collapses_whitespace() -> None:
    text = "Drug   label\n\ncontains\timportant   information."

    assert normalize_text(text) == (
        "Drug label contains important information."
    )


def test_normalize_text_removes_soft_hyphens() -> None:
    assert normalize_text("pharma\u00adceutical") == "pharmaceutical"


def test_normalize_text_handles_empty_input() -> None:
    assert normalize_text("") == ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_page_text_returns_no_chunks_for_blank_text() -> None:
    chunks = chunk_page_text(
        "",
        document_id="document-001",
        filename="label.pdf",
        page_number=1,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks == []


def test_chunk_page_text_preserves_document_and_page_metadata() -> None:
    text = (
        "Important safety information for this medication. "
        "Read the full prescribing information before use. "
    ) * 8

    chunks = chunk_page_text(
        text,
        document_id="document-001",
        filename="label.pdf",
        page_number=2,
        chunk_size=120,
        chunk_overlap=20,
    )

    assert chunks
    assert all(chunk.document_id == "document-001" for chunk in chunks)
    assert all(chunk.filename == "label.pdf" for chunk in chunks)
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_page_text_produces_unique_chunk_ids() -> None:
    text = (
        "Dosage and administration instructions. "
        "Review warnings and contraindications. "
    ) * 10

    chunks = chunk_page_text(
        text,
        document_id="document-001",
        filename="label.pdf",
        page_number=1,
        chunk_size=100,
        chunk_overlap=15,
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_page_text_uses_valid_offsets() -> None:
    text = (
        "The label contains indications, warnings, and dosage information. "
    ) * 8

    chunks = chunk_page_text(
        text,
        document_id="document-001",
        filename="label.pdf",
        page_number=1,
        chunk_size=100,
        chunk_overlap=20,
    )

    for chunk in chunks:
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start
        assert chunk.char_end - chunk.char_start == len(chunk.text)


def test_chunk_page_text_is_deterministic() -> None:
    text = (
        "Safety information should be reviewed carefully. "
        "Follow the official prescribing information. "
    ) * 8

    first = chunk_page_text(
        text,
        document_id="document-001",
        filename="label.pdf",
        page_number=1,
        chunk_size=100,
        chunk_overlap=20,
    )
    second = chunk_page_text(
        text,
        document_id="document-001",
        filename="label.pdf",
        page_number=1,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------


def test_ingest_pdf_bytes_extracts_text_and_creates_chunks(
    test_settings: Settings,
) -> None:
    pdf_bytes, filename = make_test_pdf(
        [
            (
                "INDICATIONS AND USAGE. This medication is indicated for "
                "the treatment of a specified condition. Review the official "
                "prescribing information for complete safety guidance."
            ),
            (
                "WARNINGS AND PRECAUTIONS. Monitor patients for adverse "
                "reactions. Review contraindications and dosage instructions "
                "before prescribing this medication."
            ),
        ]
    )

    result = ingest_pdf_bytes(pdf_bytes, filename, test_settings)

    assert result.filename == filename
    assert result.page_count == 2
    assert result.chunk_count > 0
    assert len(result.pages) == 2
    assert all(page.character_count > 0 for page in result.pages)
    assert all(chunk.text for chunk in result.chunks)


def test_ingest_pdf_bytes_generates_stable_document_id(
    test_settings: Settings,
) -> None:
    pdf_bytes, filename = make_test_pdf(
        [
            "Important prescribing information. " * 15,
        ]
    )

    first = ingest_pdf_bytes(pdf_bytes, filename, test_settings)
    second = ingest_pdf_bytes(pdf_bytes, filename, test_settings)

    expected_id = hashlib.sha256(pdf_bytes).hexdigest()[:24]

    assert first.document_id == expected_id
    assert second.document_id == expected_id


def test_ingest_pdf_bytes_rejects_non_pdf_content(
    test_settings: Settings,
) -> None:
    with pytest.raises(IngestionError):
        ingest_pdf_bytes(
            b"This is not a PDF document.",
            "label.pdf",
            test_settings,
        )


def test_ingest_pdf_bytes_rejects_empty_content(
    test_settings: Settings,
) -> None:
    with pytest.raises(IngestionError):
        ingest_pdf_bytes(b"", "label.pdf", test_settings)


def test_ingest_pdf_bytes_rejects_oversized_file(
    test_settings: Settings,
) -> None:
    oversized_pdf = b"%PDF-" + (b"x" * (1024 * 1024 + 1))

    with pytest.raises(IngestionError):
        ingest_pdf_bytes(
            oversized_pdf,
            "oversized.pdf",
            test_settings,
        )


def test_ingest_pdf_bytes_rejects_wrong_extension(
    test_settings: Settings,
) -> None:
    pdf_bytes, _ = make_test_pdf(["Label text. " * 20])

    with pytest.raises(IngestionError):
        ingest_pdf_bytes(
            pdf_bytes,
            "label.txt",
            test_settings,
        )


def test_ingest_pdf_bytes_rejects_pdf_with_no_extractable_text(
    test_settings: Settings,
) -> None:
    pdf_bytes, filename = make_test_pdf([""])

    with pytest.raises(IngestionError):
        ingest_pdf_bytes(pdf_bytes, filename, test_settings)


def test_ingest_pdf_reads_file_from_disk(
    tmp_path: Path,
    test_settings: Settings,
) -> None:
    pdf_bytes, filename = make_test_pdf(
        ["Important drug-label information. " * 20]
    )
    pdf_path = tmp_path / filename
    pdf_path.write_bytes(pdf_bytes)

    result = ingest_pdf(pdf_path, test_settings)

    assert result.filename == filename
    assert result.page_count == 1
    assert result.chunk_count > 0


def test_ingest_pdf_rejects_missing_file(
    tmp_path: Path,
    test_settings: Settings,
) -> None:
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises((IngestionError, FileNotFoundError, OSError)):
        ingest_pdf(missing_path, test_settings)


def test_ingestion_result_to_dict_includes_summary(
    test_settings: Settings,
) -> None:
    pdf_bytes, filename = make_test_pdf(
        ["Important prescribing information. " * 20]
    )

    result = ingest_pdf_bytes(pdf_bytes, filename, test_settings)
    data = result.to_dict()

    assert data["document_id"] == result.document_id
    assert data["filename"] == filename
    assert data["page_count"] == 1
    assert data["chunk_count"] == result.chunk_count
    assert len(data["chunks"]) == result.chunk_count
