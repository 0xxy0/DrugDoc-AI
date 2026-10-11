"""PDF ingestion, OCR, normalization, and page-aware chunking for DrugDoc AI.

Two chunking strategies are supported:

    * ``"fixed"``    — the original overlap-aware sliding window.
    * ``"semantic"`` — heading-aware chunking that prefers paragraph and
                       section boundaries while respecting size caps.

Public API (stable, used by scripts, the app, and tests):

    normalize_text
    chunk_page_text
    ingest_pdf_bytes
    ingest_pdf
    DocumentChunk
    PageMetadata
    IngestionResult
    IngestionError
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import fitz  # PyMuPDF

from drugdoc.config import Settings, get_settings

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Raised when a document cannot be safely ingested."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A searchable text chunk with traceable source metadata.

    The ``summary`` field is optional and only populated when
    ``chunk_summarization_enabled`` is true. It is embedded alongside the
    chunk text to improve high-level retrieval.
    """

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of the chunk."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class PageMetadata:
    """Extraction metadata for one source PDF page."""

    page_number: int
    character_count: int
    extraction_method: str
    ocr_used: bool


@dataclass(slots=True)
class IngestionResult:
    """Result of processing one PDF document."""

    document_id: str
    filename: str
    page_count: int
    chunks: list[DocumentChunk]
    pages: list[PageMetadata]
    warnings: list[str] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        """Number of chunks produced."""
        return len(self.chunks)

    @property
    def extracted_page_count(self) -> int:
        """Number of pages containing usable extracted text."""
        return sum(page.character_count > 0 for page in self.pages)

    def to_dict(self) -> dict[str, Any]:
        """Return document metadata without duplicating all chunk text."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "extracted_page_count": self.extracted_page_count,
            "chunk_count": self.chunk_count,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "warnings": list(self.warnings),
            "pages": [
                {
                    "page_number": page.page_number,
                    "character_count": page.character_count,
                    "extraction_method": page.extraction_method,
                    "ocr_used": page.ocr_used,
                }
                for page in self.pages
            ],
        }


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize extracted PDF text and collapse all whitespace."""

    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")  # Soft hyphen.
    text = text.replace("\u00a0", " ")  # Non-breaking space.

    # --- NEW: repair glued words caused by header/body concatenation -------
    # PyMuPDF sometimes extracts "HeaderBodyText" when a running header is
    # immediately followed by body text with no space. Split on the transition
    # from lowercase to uppercase when the uppercase run is a likely heading
    # boundary (2+ letters followed by a lowercase letter).
    text = re.sub(r"([a-z]{2})([A-Z][a-z]{3,})", r"\1 \2", text)

    # Example outputs:
    #   "USPMetformin Hydrochloride" -> "USP Metformin Hydrochloride"
    #   "Tablets, USPMetformin"      -> "Tablets, USP Metformin"
    # ----------------------------------------------------------------------

    # Repair words split across lines by PDF layout extraction:
    # "contraindi-\ncations" -> "contraindications"
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Collapse every whitespace run into a single space.
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------------------------------------------------------
# Identifiers and filenames
# ---------------------------------------------------------------------------

def _document_id(pdf_bytes: bytes) -> str:
    """Generate a deterministic ID from the PDF content."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:24]


def _safe_filename(filename: str) -> str:
    """Reject path-like names and unsupported file extensions."""

    name = Path(filename.replace("\\\\", "/")).name.strip()

    if not name or name in {".", ".."}:
        raise IngestionError("A valid filename is required.")

    if Path(name).suffix.lower() != ".pdf":
        raise IngestionError("Only PDF documents are supported.")

    if any(ord(char) < 32 for char in name):
        raise IngestionError("Filename contains invalid control characters.")

    return name


def _make_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    char_start: int,
    chunk_text: str,
) -> str:
    """Deterministic chunk identifier stable across re-ingestion."""
    identity = (
        f"{document_id}:{page_number}:{chunk_index}:"
        f"{char_start}:{chunk_text}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# OCR fallback
# ---------------------------------------------------------------------------

def _extract_with_ocr(page: fitz.Page, language: str) -> str:
    """Extract text from a rendered page using optional Tesseract OCR."""

    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise IngestionError(
            "OCR is required for a scanned page, but pytesseract/Pillow "
            "is not installed. Install both packages and the Tesseract "
            "system executable, or disable OCR in the configuration."
        ) from exc

    try:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return str(pytesseract.image_to_string(image, lang=language))

    except Exception as exc:
        page_number = (
            page.number + 1 if page.number is not None else "unknown"
        )
        raise IngestionError(
            f"OCR failed on PDF page {page_number}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Fixed-size chunking (original behavior)
# ---------------------------------------------------------------------------

def _find_chunk_end(
    text: str,
    start: int,
    target_end: int,
    chunk_size: int,
) -> int:
    """Prefer a natural text boundary without producing tiny chunks."""

    if target_end >= len(text):
        return len(text)

    minimum_end = start + max(1, int(chunk_size * 0.65))
    search_text = text[minimum_end:target_end]

    # Prefer paragraph, sentence, then clause separators.
    for separator in ("\n\n", "\n", ". ", "; ", ", ", " "):
        position = search_text.rfind(separator)
        if position >= 0:
            end = minimum_end + position + len(separator)
            if end > start:
                return end

    # Never cut inside a word: snap to the nearest space.
    probe = text.rfind(" ", start, target_end)
    if probe > start + int(chunk_size * 0.4):
        return probe + 1

    return target_end

def _chunk_fixed(
    text: str,
    *,
    document_id: str,
    filename: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Original overlap-aware sliding-window chunker."""

    if not text:
        return []

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = _find_chunk_end(text, start, target_end, chunk_size)

        raw_slice = text[start:end]
        chunk_text = raw_slice.strip()

        if chunk_text:
            leading_ws = len(raw_slice) - len(raw_slice.lstrip())
            actual_start = start + leading_ws
            actual_end = actual_start + len(chunk_text)

            chunk_id = _make_chunk_id(
                document_id=document_id,
                page_number=page_number,
                chunk_index=chunk_index,
                char_start=actual_start,
                chunk_text=chunk_text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=actual_start,
                    char_end=actual_end,
                )
            )

            chunk_index += 1

        if end >= len(text):
            break

        # Guarantee forward progress even if a boundary is unexpectedly short.
        next_start = end - chunk_overlap
        start = max(start + 1, next_start)

    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking (heading-aware)
# ---------------------------------------------------------------------------

# Common drug-label section headings. Matching is case-insensitive and
# tolerant of punctuation such as ":" or " -".
_LABEL_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:boxed\s+warning|black\s+box\s+warning)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:indications?(?:\s+and\s+usage)?|uses?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^dosage(?:\s+and\s+administration)?\b", re.IGNORECASE),
    re.compile(
        r"^contraindications?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^warnings?(?:\s+and\s+precautions?)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"^precautions?\b", re.IGNORECASE),
    re.compile(
        r"^adverse\s+reactions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^drug\s+interactions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^use\s+in\s+specific\s+populations?\b",
        re.IGNORECASE,
    ),
    re.compile(r"^overdosage\b", re.IGNORECASE),
    re.compile(r"^description\b", re.IGNORECASE),
    re.compile(
        r"^clinical\s+pharmacology\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^pharmacokinetics\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^storage(?:\s+and\s+handling)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^how\s+supplied(?:\s*/\s*storage(?:\s+and\s+handling)?)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^patient\s+counseling\s+information\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^mechanism\s+of\s+action\b",
        re.IGNORECASE,
    ),
)

# Generic numbered section headings such as "1 INDICATIONS" or "3.2 Dosage".
_NUMBERED_HEADING_RE = re.compile(
    r"^\d+(?:\.\d+)*[\.\s]+[A-Z][A-Za-z].{0,80}$",
)


def _looks_like_heading(line: str) -> bool:
    """Return True when a line looks like a drug-label section heading."""

    stripped = line.strip()

    if not stripped or len(stripped) > 120:
        return False

    # Drop common trailing punctuation for matching.
    probe = stripped.rstrip(":").strip()

    for pattern in _LABEL_HEADING_PATTERNS:
        if pattern.match(probe):
            return True

    # A short line with ALL-CAPS words (>=2) is very likely a heading.
    if len(probe) <= 80:
        uppercase_words = re.findall(r"\b[A-Z]{2,}\b", probe)
        if len(uppercase_words) >= 2:
            return True

    if _NUMBERED_HEADING_RE.match(probe):
        return True

    return False


def _split_paragraphs(text: str) -> list[str]:
    """Split normalized text into paragraph-like segments.

    Normalization collapses whitespace, so we cannot rely on blank lines.
    Instead we split on sentence boundaries and then greedily merge
    sentences up to the semantic maximum. Heading detection happens later
    and can force a boundary regardless of length.
    """

    # Sentence-aware splitting that preserves decimals like "1.5 mg".
    pieces = re.split(r"(?<=[\.\?\!;:])\s+(?=[A-Z0-9])", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _chunk_semantic(
    text: str,
    *,
    document_id: str,
    filename: str,
    page_number: int,
    min_chars: int,
    max_chars: int,
) -> list[DocumentChunk]:
    """Heading-aware chunker.

    The algorithm:

        1. Split the normalized page text into candidate segments.
        2. Greedily merge segments into a chunk until adding the next segment
           would exceed ``max_chars``.
        3. Force a chunk boundary when a heading is detected.
        4. Emit chunks that are at least ``min_chars`` when possible; the
           final chunk of a page may be shorter.
    """

    if not text:
        return []

    if min_chars < 1:
        raise ValueError("min_chars must be positive.")

    if max_chars < min_chars:
        raise ValueError("max_chars must be >= min_chars.")

    segments = _split_paragraphs(text)

    if not segments:
        return []

    # Reconstruct approximate character offsets as we merge, so every chunk
    # still carries meaningful char_start/char_end values.
    chunks: list[DocumentChunk] = []
    buffer: list[str] = []
    buffer_start: int = 0
    cursor: int = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_start, chunk_index

        if not buffer:
            return

        joined = " ".join(buffer).strip()
        if not joined:
            buffer = []
            return

        chunk_text = joined
        actual_start = buffer_start
        actual_end = actual_start + len(chunk_text)

        chunk_id = _make_chunk_id(
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            char_start=actual_start,
            chunk_text=chunk_text,
        )

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=filename,
                page_number=page_number,
                chunk_index=chunk_index,
                text=chunk_text,
                char_start=actual_start,
                char_end=actual_end,
            )
        )

        chunk_index += 1
        buffer = []

    for segment in segments:
        # Track a stable position estimate for the next buffered chunk.
        if not buffer:
            buffer_start = max(0, cursor)

        segment_is_heading = _looks_like_heading(segment)
        current_length = sum(len(part) for part in buffer) + max(
            0, len(buffer) - 1
        )

        # Force a boundary before a heading, unless the buffer is empty.
        if segment_is_heading and buffer:
            flush()
            buffer_start = max(0, cursor)

        # Force a boundary if adding this segment would overflow max_chars
        # and we already have enough content to emit a usable chunk.
        prospective_length = current_length + (
            1 if buffer else 0
        ) + len(segment)

        if prospective_length > max_chars and buffer:
            if current_length >= min_chars:
                flush()
                buffer_start = max(0, cursor)
            else:
                # Buffer is below min_chars; try to keep it intact and
                # instead cut at the nearest word boundary inside the
                # segment by splitting the segment itself.
                if len(segment) > max_chars:
                    head = segment[:max_chars]
                    tail = segment[max_chars:]
                    buffer.append(head.strip())
                    flush()
                    buffer_start = max(0, cursor + max_chars)
                    buffer.append(tail.strip())
                    cursor += len(segment) + 1
                    continue
                else:
                    flush()
                    buffer_start = max(0, cursor)

        buffer.append(segment)
        cursor += len(segment) + 1

    flush()

    return chunks


# ---------------------------------------------------------------------------
# Public chunker dispatcher (backward compatible)
# ---------------------------------------------------------------------------

def chunk_page_text(
    text: str,
    *,
    document_id: str,
    filename: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str = "fixed",
    semantic_min_chunk_chars: int = 250,
    semantic_max_chunk_chars: int = 1400,
) -> list[DocumentChunk]:
    """Split one page into traceable text chunks.

    By default this preserves the original fixed-size behavior so existing
    callers and tests are unaffected. Pass ``chunking_strategy="semantic"``
    (and optionally the semantic bounds) to use heading-aware chunking.
    """

    normalized = normalize_text(text)

    if not normalized:
        return []

    if chunking_strategy == "fixed":
        return _chunk_fixed(
            normalized,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if chunking_strategy == "semantic":
        return _chunk_semantic(
            normalized,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            min_chars=semantic_min_chunk_chars,
            max_chars=semantic_max_chunk_chars,
        )

    raise ValueError(
        f"Unknown chunking_strategy: {chunking_strategy!r}. "
        "Expected 'fixed' or 'semantic'."
    )


# ---------------------------------------------------------------------------
# Optional chunk summarization
# ---------------------------------------------------------------------------

def _maybe_generate_chunk_summaries(
    chunks: list[DocumentChunk],
    settings: Settings,
    warnings: list[str],
) -> list[DocumentChunk]:
    """Populate the ``summary`` field on each chunk when enabled.

    Summarization is best-effort: if the LLM is unreachable or the provider
    is misconfigured, we log a single warning and return the original
    chunks unmodified. Ingestion never fails because of summarization.
    """

    if not settings.chunk_summarization_enabled:
        return chunks

    try:
        from drugdoc.rag import _generate_with_ollama, _generate_with_openai
    except Exception as exc:  # pragma: no cover - defensive import guard
        warnings.append(
            f"Chunk summarization unavailable (import error): {exc}"
        )
        return chunks

    summarizer_prompt_template = (
        "Summarize the following drug-label passage in one short sentence. "
        "Do not add any information not present in the passage.\n\n"
        "PASSAGE:\n{passage}\n\n"
        "Return only the summary text."
    )

    summarized: list[DocumentChunk] = []
    failures = 0

    for chunk in chunks:
        prompt = summarizer_prompt_template.format(passage=chunk.text)

        try:
            if settings.llm_provider == "ollama":
                raw = _generate_with_ollama(prompt, settings)
            elif settings.llm_provider == "openai":
                raw = _generate_with_openai(prompt, settings)
            else:
                raise RuntimeError(
                    f"Unsupported LLM provider: {settings.llm_provider!r}"
                )

            summary = (raw or "").strip().splitlines()[0][:400]
        except Exception:
            failures += 1
            summary = ""

        summarized.append(
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                summary=summary,
            )
        )

    if failures:
        warnings.append(
            f"Chunk summarization failed for {failures} of "
            f"{len(chunks)} chunk(s); those chunks keep empty summaries."
        )

    return summarized


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def ingest_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    settings: Settings | None = None,
) -> IngestionResult:
    """Ingest an uploaded PDF held in memory.

    This is the main entry point for Streamlit uploads and future API routes.
    It does not persist the original file or write to the vector database.
    """

    settings = settings or get_settings()
    filename = _safe_filename(filename)

    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise IngestionError("The uploaded PDF is empty or invalid.")

    if len(pdf_bytes) > settings.max_upload_size_bytes:
        raise IngestionError(
            f"File exceeds the {settings.max_upload_size_mb} MB upload limit."
        )

    # Check the PDF signature before attempting document parsing.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise IngestionError(
            "The uploaded file does not have a valid PDF signature."
        )

    document_id = _document_id(pdf_bytes)
    chunks: list[DocumentChunk] = []
    pages_metadata: list[PageMetadata] = []
    warnings: list[str] = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise IngestionError(
                    "Password-protected PDFs are not supported."
                )

            page_count = len(document)

            if page_count == 0:
                raise IngestionError("The PDF contains no pages.")

            for page_number in range(1, page_count + 1):
                page = document.load_page(page_number - 1)

                extracted_text = normalize_text(
                    cast(str, page.get_text("text"))
                )
                extraction_method = "pdf_text"
                ocr_used = False

                # OCR is a fallback for pages with insufficient extractable text.
                if (
                    settings.ocr_enabled
                    and len(extracted_text)
                    < settings.min_text_characters_per_page
                ):
                    try:
                        ocr_text = normalize_text(
                            _extract_with_ocr(page, settings.ocr_language)
                        )

                        if len(ocr_text) > len(extracted_text):
                            extracted_text = ocr_text
                            extraction_method = "ocr"
                            ocr_used = True

                    except IngestionError as exc:
                        # Keep usable native text, but record OCR failure.
                        if not extracted_text:
                            warnings.append(
                                f"Page {page_number}: {exc}"
                            )
                        else:
                            warnings.append(
                                f"Page {page_number}: OCR fallback failed; "
                                "native PDF text was retained."
                            )

                pages_metadata.append(
                    PageMetadata(
                        page_number=page_number,
                        character_count=len(extracted_text),
                        extraction_method=extraction_method,
                        ocr_used=ocr_used,
                    )
                )

                if not extracted_text:
                    warnings.append(
                        f"Page {page_number} produced no extractable text."
                    )
                    continue
                
                if extracted_text:
                    digit_count = sum(
                        c.isdigit() for c in extracted_text
                    )
                    if (
                        len(extracted_text) > 40
                        and digit_count / len(extracted_text) > 0.6
                    ):
                        warnings.append(
                            f"Page {page_number} skipped: looks like a "
                            "barcode or printing artifact."
                        )
                        continue

                page_chunks = chunk_page_text(
                    extracted_text,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    chunking_strategy=settings.chunking_strategy,
                    semantic_min_chunk_chars=settings.semantic_min_chunk_chars,
                    semantic_max_chunk_chars=settings.semantic_max_chunk_chars,
                )

                chunks.extend(page_chunks)

    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Could not read PDF {filename!r}: {exc}"
        ) from exc

    if not chunks:
        raise IngestionError(
            "No usable text could be extracted from this PDF. "
            "Try a text-based PDF or verify that OCR is installed and "
            "configured."
        )

    # Optional: add short summaries used by the retrieval layer.
    if settings.chunk_summarization_enabled:
        chunks = _maybe_generate_chunk_summaries(chunks, settings, warnings)

    result = IngestionResult(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        chunks=chunks,
        pages=pages_metadata,
        warnings=warnings,
    )

    LOGGER.info(
        "Ingested %s: document_id=%s pages=%d chunks=%d strategy=%s "
        "warnings=%d",
        filename,
        document_id,
        page_count,
        result.chunk_count,
        settings.chunking_strategy,
        len(warnings),
    )

    return result


def ingest_pdf(
    file_path: str | Path,
    settings: Settings | None = None,
) -> IngestionResult:
    """Ingest a PDF from disk using the same pipeline as uploaded files."""

    path = Path(file_path)

    if not path.is_file():
        raise IngestionError(f"PDF file does not exist: {path}")

    try:
        with path.open("rb") as file:
            pdf_bytes = file.read()
    except OSError as exc:
        raise IngestionError(f"Could not read PDF file: {exc}") from exc

    return ingest_pdf_bytes(
        pdf_bytes=pdf_bytes,
        filename=path.name,
        settings=settings,
    )