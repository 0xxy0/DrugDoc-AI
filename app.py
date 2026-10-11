"""Streamlit application for DrugDoc AI.

Two-page layout:

    * DrugDoc AI        — upload, ask, library, evaluation
    * Pipeline Observatory — live, step-by-step view of the RAG pipeline

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

# Ensure the src-layout package is importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drugdoc.config import get_settings
from drugdoc.evaluation import Evaluator, load_questions
from drugdoc.ingestion import (
    IngestionError,
    ingest_pdf_bytes,
    ingest_pdf,
)
from drugdoc.rag import RAGPipeline, RAGError
from drugdoc.retrieval import HybridRetriever, RetrievalError
from drugdoc.storage import DocumentStore
from drugdoc.tracing import TraceEvent

LOGGER = logging.getLogger("drugdoc.app")


# ---------------------------------------------------------------------------
# Page config + global theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DrugDoc AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

/* Force a consistent light-on-light palette for our custom blocks,
   regardless of the ambient Streamlit theme. */
:root {
    --dd-ink:        #0f172a;   /* primary text */
    --dd-ink-soft:   #334155;   /* secondary text */
    --dd-ink-mute:   #64748b;   /* tertiary text */
    --dd-surface:    #ffffff;   /* card background */
    --dd-surface-2:  #f8fafc;   /* subtle background */
    --dd-border:     rgba(15, 23, 42, 0.10);
    --dd-accent:     #4f46e5;
    --dd-accent-2:   #7c3aed;
}

/* ------------------------------------------------------------------ */
/* Hero banner                                                         */
/* ------------------------------------------------------------------ */
.drugdoc-hero {
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #eef4ff 0%, #f7f0ff 100%);
    border: 1px solid rgba(80, 100, 200, 0.18);
    margin-bottom: 1.1rem;
    color: var(--dd-ink);
}
.drugdoc-hero h1 {
    margin: 0 0 .35rem 0;
    font-weight: 700;
    letter-spacing: -0.02em;
    font-size: 1.85rem;
    color: var(--dd-ink) !important;
    line-height: 1.2;
}
.drugdoc-hero p {
    margin: 0;
    color: var(--dd-ink-soft) !important;
    font-size: 0.98rem;
    line-height: 1.55;
}
.drugdoc-hero .eyebrow {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--dd-accent) !important;
    text-transform: uppercase;
    margin-bottom: .45rem;
}

/* ------------------------------------------------------------------ */
/* Cards                                                               */
/* ------------------------------------------------------------------ */
.drugdoc-card {
    padding: 1rem 1.15rem;
    border-radius: 12px;
    background: var(--dd-surface);
    border: 1px solid var(--dd-border);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    margin-bottom: .75rem;
    color: var(--dd-ink) !important;
    line-height: 1.6;
    font-size: 0.98rem;
}
.drugdoc-card * {
    color: var(--dd-ink) !important;
}
.drugdoc-card a,
.drugdoc-card a * {
    color: var(--dd-accent) !important;
    text-decoration: underline;
}

/* ------------------------------------------------------------------ */
/* Pills                                                               */
/* ------------------------------------------------------------------ */
.drugdoc-pill {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    line-height: 1.4;
    border: 1px solid transparent;
}
.drugdoc-pill.ok   { background: #dcfce7; color: #14532d !important; border-color: #86efac; }
.drugdoc-pill.warn { background: #fef3c7; color: #78350f !important; border-color: #fcd34d; }
.drugdoc-pill.err  { background: #fee2e2; color: #7f1d1d !important; border-color: #fca5a5; }
.drugdoc-pill.info { background: #e0e7ff; color: #312e81 !important; border-color: #a5b4fc; }

/* ------------------------------------------------------------------ */
/* Trace steps                                                         */
/* ------------------------------------------------------------------ */
.drugdoc-trace-step {
    border-left: 3px solid var(--dd-accent);
    padding: .5rem .85rem;
    margin: .4rem 0;
    background: var(--dd-surface-2);
    border-radius: 0 8px 8px 0;
    color: var(--dd-ink) !important;
}
.drugdoc-trace-step .kind {
    font-size: 0.70rem;
    font-weight: 700;
    color: var(--dd-accent) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.drugdoc-trace-step .title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--dd-ink) !important;
    margin: 2px 0;
    line-height: 1.4;
}
.drugdoc-trace-step .elapsed {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    color: var(--dd-ink-mute) !important;
}
.drugdoc-trace-step.ok   { border-left-color: #22c55e; }
.drugdoc-trace-step.ok   .kind { color: #15803d !important; }
.drugdoc-trace-step.warn { border-left-color: #f59e0b; }
.drugdoc-trace-step.warn .kind { color: #b45309 !important; }
.drugdoc-trace-step.err  { border-left-color: #ef4444; }
.drugdoc-trace-step.err  .kind { color: #b91c1c !important; }

/* ------------------------------------------------------------------ */
/* Confidence meter                                                    */
/* ------------------------------------------------------------------ */
.drugdoc-meter-wrap {
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
    height: 10px;
    margin: .35rem 0 .35rem 0;
}
.drugdoc-meter-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    transition: width .3s ease;
}
.drugdoc-meter-label {
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--dd-ink) !important;
}

/* ------------------------------------------------------------------ */
/* Sidebar status dots + kv                                            */
/* ------------------------------------------------------------------ */
.drugdoc-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.drugdoc-dot.ok   { background: #22c55e; box-shadow: 0 0 0 3px #dcfce7; }
.drugdoc-dot.warn { background: #f59e0b; box-shadow: 0 0 0 3px #fef3c7; }
.drugdoc-dot.err  { background: #ef4444; box-shadow: 0 0 0 3px #fee2e2; }

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
.drugdoc-muted {
    color: var(--dd-ink-mute) !important;
    font-size: 0.86rem;
    line-height: 1.5;
}
.drugdoc-kv {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
    color: var(--dd-ink-soft) !important;
}

/* ------------------------------------------------------------------ */
/* Streamlit caption fix (avoid faint/invisible text)                  */
/* ------------------------------------------------------------------ */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: var(--dd-ink-mute) !important;
    opacity: 1 !important;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared services
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading DrugDoc AI services...")
def get_services() -> tuple[DocumentStore, HybridRetriever, RAGPipeline]:
    """Create one shared service stack and restore the saved document corpus."""

    settings = get_settings()
    store = DocumentStore(settings)
    retriever = HybridRetriever(settings)

    # Rebuild in-memory chunk metadata and sparse retrieval from saved PDFs.
    records = store.list_documents(status="indexed")

    for record in records:
        pdf_path = store.get_document_path(record.document_id)

        if not pdf_path or not Path(pdf_path).is_file():
            LOGGER.warning(
                "Saved document %s has no accessible PDF; skipping restoration.",
                record.document_id,
            )
            continue

        try:
            result = ingest_pdf(pdf_path, settings)
            if result.document_id != record.document_id:
                LOGGER.error(
                    "Document hash mismatch while restoring %s.",
                    record.document_id,
                )
                continue

            retriever.index_chunks(result.chunks)

        except (IngestionError, RetrievalError, OSError, ValueError):
            LOGGER.exception(
                "Could not restore document %s.",
                record.document_id,
            )

    pipeline = RAGPipeline(retriever, settings)
    return store, retriever, pipeline


def clear_service_cache() -> None:
    """Clear cached service objects after a corpus change."""
    get_services.clear()


def display_error(message: str, error: Exception) -> None:
    """Show a user-friendly error while retaining diagnostic logs."""
    LOGGER.exception("%s", message, exc_info=error)
    st.error(f"{message} Details: {error}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _best_sentence_quote(
    quote: str, question: str, max_chars: int = 420
) -> str:
    """Return the most relevant slice of a citation for the UI."""

    if not quote:
        return ""

    sentences = re.split(r"(?<=[\.\?\!])\s+", quote)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return (
            quote[:max_chars].rstrip()
            + ("…" if len(quote) > max_chars else "")
        )

    query_terms = {
        t.lower()
        for t in re.findall(r"[a-zA-Z]{3,}", question)
        if t.lower() not in {"the", "and", "for", "with", "are", "was"}
    }

    def score(sentence: str) -> int:
        words = {t.lower() for t in re.findall(r"[a-zA-Z]{3,}", sentence)}
        return len(words & query_terms)

    ranked = sorted(sentences, key=score, reverse=True)
    best = ranked[0]

    if len(best) < 160 and len(ranked) > 1:
        best = f"{best} {ranked[1]}".strip()

    if len(best) > max_chars:
        best = best[:max_chars].rstrip() + "…"

    return best


def _confidence_score(response: Any) -> int:
    """Heuristic 0–100 confidence score for the UI.

    Combines citations, evidence count, and self-correction verdict.
    Not a calibrated probability — only a visual signal.
    """

    score = 0.0
    score += min(40.0, len(response.citations) * 20.0)
    score += min(30.0, response.evidence_count * 6.0)

    if response.self_correction is not None:
        verdict = response.self_correction.get("verdict")
        if verdict == "supported":
            score += 30.0
        elif verdict == "unsupported":
            score += 0.0
        else:
            score += 15.0
    else:
        score += 15.0

    if response.status != "answered":
        score = min(score, 30.0)

    return int(max(0.0, min(100.0, score)))


def _render_confidence_meter(score: int) -> None:
    """Render the confidence meter as a horizontal bar."""

    st.markdown(
        f"""
        <div class="drugdoc-meter-label">Answer confidence · {score} / 100</div>
        <div class="drugdoc-meter-wrap">
            <div class="drugdoc-meter-bar" style="width: {score}%;"></div>
        </div>
        <div class="drugdoc-muted">
            Heuristic combining citations, evidence count, and the
            self-correction verdict. Not a calibrated probability.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_pill(status: str) -> None:
    """Render a colored status pill."""
    mapping = {
        "answered": ("ok", "answered"),
        "insufficient_evidence": ("warn", "insufficient evidence"),
        "error": ("err", "error"),
    }
    css_class, label = mapping.get(status, ("info", status))
    st.markdown(
        f'<span class="drugdoc-pill {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


def _render_trace_event(event: TraceEvent) -> None:
    """Render one pipeline step in the observatory trace stream."""

    ok_kinds = {
        "llm.raw",
        "parsed",
        "citations",
        "verification",
        "final",
        "prompt",
    }
    warn_kinds = {"retrieval.dense", "retrieval.sparse", "recursion"}
    info_kinds = {"query", "expansion", "fusion", "mmr"}

    if event.kind in ok_kinds:
        css_class = "ok"
    elif event.kind in warn_kinds:
        css_class = "warn"
    elif event.kind in info_kinds:
        css_class = ""
    else:
        css_class = ""

    st.markdown(
        f"""
        <div class="drugdoc-trace-step {css_class}">
            <div class="kind">{event.kind}</div>
            <div class="title">{event.title}</div>
            <div class="elapsed">{event.elapsed_ms} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if event.notes:
        st.caption(event.notes)

    if event.payload:
        with st.expander("Show details", expanded=False):
            st.json(event.payload)


def _render_sidebar(settings: Any, store: DocumentStore) -> None:
    """Render the sidebar with system status and feature flags."""

    with st.sidebar:
        st.markdown("### 💊 DrugDoc AI")
        st.caption("Evidence-grounded label QA")

        st.divider()
        st.markdown("#### System status")

        indexed = store.list_documents(status="indexed")
        total_docs = len(store.list_documents())

        st.markdown(
            f'<span class="drugdoc-dot ok"></span>'
            f'<span class="drugdoc-kv">'
            f'{settings.llm_provider} · {settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model}'
            f'</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="drugdoc-dot ok"></span>'
            f'<span class="drugdoc-kv">'
            f'{settings.vector_store_type} · {settings.qdrant_collection}'
            f'</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="drugdoc-dot ok"></span>'
            f'<span class="drugdoc-kv">'
            f'{settings.embedding_model}'
            f'</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="drugdoc-dot {"ok" if indexed else "warn"}"></span>'
            f'<span class="drugdoc-kv">'
            f'{len(indexed)} indexed · {total_docs} stored'
            f'</span>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("#### Retrieval features")

        feature_rows = [
            ("Multi-query", settings.retrieval_multi_query_enabled),
            ("Recursive", settings.retrieval_recursive_enabled),
            ("MMR diversity", settings.retrieval_mmr_enabled),
            ("Self-correction", settings.self_correction_enabled),
            ("Answer comparison", settings.enable_answer_comparison),
        ]

        for label, enabled in feature_rows:
            icon = "✅" if enabled else "❌"
            st.markdown(
                f'<span class="drugdoc-kv">{icon} {label}</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(
            '<span class="drugdoc-muted">'
            "First-time model loading may take a while. Local LLM answers "
            "require the configured Ollama model to be installed and running."
            "</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page: DrugDoc AI (main tabbed app)
# ---------------------------------------------------------------------------

def _page_main(
    settings: Any,
    store: DocumentStore,
    retriever: HybridRetriever,
    pipeline: RAGPipeline,
) -> None:
    """Main tabbed application."""

    # --- Hero --------------------------------------------------------------
    st.markdown(
        """
        <div class="drugdoc-hero">
            <div class="eyebrow">Evidence-grounded RAG · drug labels</div>
            <h1>💊 DrugDoc AI</h1>
            <p>Ask questions about pharmaceutical drug labels and get
            answers backed by citable source passages — not model
            hallucinations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "DrugDoc AI retrieves passages from uploaded labels and uses them "
        "to answer questions with source citations. It is a document-research "
        "aid, not a substitute for a healthcare professional or the official "
        "label."
    )

    upload_tab, ask_tab, documents_tab, evaluation_tab = st.tabs(
        [
            "📄 Upload labels",
            "💬 Ask questions",
            "📚 Document library",
            "📊 Evaluation",
        ]
    )

    # ------------------------------------------------------------------
    # Upload tab
    # ------------------------------------------------------------------
    with upload_tab:
        st.subheader("Add drug labels")
        st.write(
            "Upload PDF drug labels. Text is extracted page by page, divided "
            "into retrievable chunks, and indexed for question answering."
        )

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="The configured upload-size limit applies to each PDF.",
            key="drugdoc_pdf_upload",
        )

        if uploaded_files:
            max_bytes = settings.max_upload_size_mb * 1024 * 1024
            st.write(f"Selected files: **{len(uploaded_files)}**")

            for uploaded_file in uploaded_files:
                size_mb = uploaded_file.size / (1024 * 1024)
                st.write(f"• {uploaded_file.name} — {size_mb:.2f} MB")

            if st.button(
                "Process and index PDFs",
                type="primary",
                disabled=not uploaded_files,
            ):
                successful = 0
                failures = 0
                progress = st.progress(
                    0, text="Preparing document ingestion..."
                )

                for index, uploaded_file in enumerate(uploaded_files):
                    progress.progress(
                        index / len(uploaded_files),
                        text=f"Processing {uploaded_file.name}...",
                    )

                    try:
                        pdf_bytes = uploaded_file.getvalue()

                        if len(pdf_bytes) > max_bytes:
                            raise ValueError(
                                f"File exceeds the "
                                f"{settings.max_upload_size_mb} MB upload "
                                "limit."
                            )

                        result = ingest_pdf_bytes(
                            pdf_bytes,
                            uploaded_file.name,
                            settings,
                        )

                        retriever.index_chunks(result.chunks)

                        record = store.save_document(
                            pdf_bytes=pdf_bytes,
                            result=result,
                            status="indexed",
                        )

                        successful += 1

                        st.success(
                            f"Indexed **{record.filename}** — "
                            f"{result.page_count} pages, "
                            f"{result.chunk_count} chunks."
                        )

                        for warning in result.warnings:
                            st.warning(f"{record.filename}: {warning}")

                    except (
                        IngestionError,
                        RetrievalError,
                        OSError,
                        ValueError,
                    ) as exc:
                        failures += 1
                        LOGGER.exception(
                            "Failed to process uploaded file %s.",
                            uploaded_file.name,
                        )
                        st.error(
                            f"Could not process {uploaded_file.name}: {exc}"
                        )

                    except Exception as exc:
                        failures += 1
                        LOGGER.exception(
                            "Unexpected ingestion error for %s.",
                            uploaded_file.name,
                        )
                        st.error(
                            f"Unexpected error while processing "
                            f"{uploaded_file.name}: {exc}"
                        )

                progress.progress(1.0, text="Ingestion finished.")

                st.write(
                    f"**Completed:** {successful} · **Failed:** {failures}"
                )

                if successful:
                    st.toast(f"Indexed {successful} document(s).", icon="✅")
                    st.cache_data.clear()
                    st.rerun()

    # ------------------------------------------------------------------
    # Ask tab
    # ------------------------------------------------------------------
    with ask_tab:
        st.subheader("Ask about your drug labels")

        available_documents = store.list_documents(status="indexed")

        if not available_documents:
            st.info(
                "Upload and index at least one PDF to start asking "
                "questions. You can also open the **Pipeline Observatory** "
                "from the sidebar to see the full RAG pipeline work end to "
                "end once a document is available."
            )
        else:
            document_options = {
                f"{record.filename} · {record.document_id}": record.document_id
                for record in available_documents
            }

            scope = st.radio(
                "Search scope",
                ["All indexed documents", "Selected documents"],
                horizontal=True,
            )

            selected_ids: list[str] | None = None

            if scope == "Selected documents":
                selected_labels = st.multiselect(
                    "Choose documents",
                    options=list(document_options.keys()),
                    default=list(document_options.keys()),
                )
                selected_ids = [
                    document_options[label] for label in selected_labels
                ]

            with st.form("drugdoc_question_form"):
                question = st.text_area(
                    "Your question",
                    placeholder=(
                        "Example: What contraindications are listed on this "
                        "label?"
                    ),
                    height=100,
                    max_chars=2000,
                )
                submitted = st.form_submit_button(
                    "Find evidence and answer",
                    type="primary",
                )

            if submitted:
                if not question.strip():
                    st.warning("Enter a question first.")
                elif not selected_ids:
                    st.warning("Select at least one document.")
                else:
                    try:
                        with st.spinner(
                            "Retrieving evidence and generating an answer..."
                        ):
                            response = pipeline.answer_question(
                                question=question.strip(),
                                document_ids=set(selected_ids),
                            )

                        # --- Answer header --------------------------------
                        _render_status_pill(response.status)
                        st.markdown("#### Answer")

                        if response.status == "answered":
                            st.markdown(
                                f'<div class="drugdoc-card">'
                                f'{response.answer}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.warning(response.answer)

                        # --- Confidence meter -----------------------------
                        _render_confidence_meter(
                            _confidence_score(response)
                        )

                        st.caption(
                            f"Evidence chunks: {response.evidence_count} · "
                            f"Latency: {response.latency_ms} ms · "
                            f"Retrieval source: "
                            f"{response.retrieval_source or '—'}"
                        )

                        if response.abstention_reason:
                            st.caption(
                                f"Abstention reason: "
                                f"{response.abstention_reason}"
                            )

                        if response.warnings:
                            for warning in response.warnings:
                                st.warning(warning)

                        # --- Retrieval trace ------------------------------
                        if (
                            settings.show_retrieval_trace
                            and response.retrieval_trace
                        ):
                            with st.expander(
                                "🔎 Retrieval trace",
                                expanded=False,
                            ):
                                for step in response.retrieval_trace:
                                    st.markdown(f"- {step}")

                        # --- Self-correction ------------------------------
                        if response.self_correction is not None:
                            verdict = response.self_correction.get(
                                "verdict", "skipped"
                            )
                            reason = (
                                response.self_correction.get("reason") or ""
                            ).strip()
                            unsupported = response.self_correction.get(
                                "unsupported_claims"
                            ) or []

                            informative_reason = (
                                reason
                                if reason
                                and not reason.lower().startswith(
                                    "this is a valid json"
                                )
                                else ""
                            )

                            if verdict == "supported":
                                label = (
                                    "✅ Self-correction: every claim is "
                                    "supported by the evidence."
                                )
                                if informative_reason:
                                    label += f" ({informative_reason})"
                                st.success(label)

                            elif verdict == "unsupported":
                                label = (
                                    "❌ Self-correction: the drafted answer "
                                    "is not fully supported by the evidence."
                                )
                                if informative_reason:
                                    label += f" ({informative_reason})"
                                st.error(label)

                                if unsupported:
                                    with st.expander(
                                        "Unsupported claims flagged by the "
                                        "verifier"
                                    ):
                                        for claim in unsupported:
                                            st.markdown(f"- {claim}")

                            else:
                                st.caption(
                                    "Self-correction skipped: verification "
                                    "was unavailable for this answer."
                                )

                        # --- Base comparison ------------------------------
                        if (
                            settings.enable_answer_comparison
                            and response.base_answer
                        ):
                            with st.expander(
                                "🆚 Compare with base LLM (no retrieval)",
                            ):
                                col_rag, col_base = st.columns(2)
                                with col_rag:
                                    st.caption("**DrugDoc AI (grounded)**")
                                    st.write(response.answer)
                                with col_base:
                                    st.caption("**Base LLM (ungrounded)**")
                                    st.write(response.base_answer)

                        # --- Citations ------------------------------------
                        st.markdown("#### Source evidence")

                        if response.citations:
                            for citation in response.citations:
                                title = (
                                    f"{citation.filename} — "
                                    f"page {citation.page_number}"
                                )
                                with st.expander(title):
                                    st.caption(
                                        f"Citation ID: {citation.citation_id}"
                                    )
                                    st.caption(
                                        f"Chunk ID: {citation.chunk_id}"
                                    )
                                    st.write(
                                        _best_sentence_quote(
                                            citation.quote,
                                            question,
                                        )
                                    )
                                    with st.popover("View full source passage"):
                                        st.caption(
                                            f"Full chunk text "
                                            f"({len(citation.quote)} chars)"
                                        )
                                        st.text(citation.quote)
                        else:
                            st.info(
                                "No valid source citations were returned."
                            )

                        # --- Metadata -------------------------------------
                        with st.expander("Inspect response metadata"):
                            st.json(response.to_dict())

                        # --- Download -------------------------------------
                        st.download_button(
                            "⬇ Download answer as JSON",
                            data=json.dumps(
                                response.to_dict(),
                                indent=2,
                                default=str,
                            ),
                            file_name=(
                                f"drugdoc-answer-{int(time.time())}.json"
                            ),
                            mime="application/json",
                        )

                    except (
                        RAGError,
                        RetrievalError,
                        OSError,
                        ValueError,
                    ) as exc:
                        display_error("Question answering failed.", exc)
                    except Exception as exc:
                        display_error(
                            "An unexpected error occurred while answering.",
                            exc,
                        )

    # ------------------------------------------------------------------
    # Document library tab
    # ------------------------------------------------------------------
    with documents_tab:
        st.subheader("Indexed document library")

        records = store.list_documents()

        if not records:
            st.info("No documents have been saved yet.")
        else:
            total_pages = sum(record.page_count for record in records)
            total_chunks = sum(record.chunk_count for record in records)
            indexed_count = sum(
                1 for record in records if record.status == "indexed"
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Documents", len(records))
            col2.metric("Indexed", indexed_count)
            col3.metric("Pages", total_pages)
            col4.metric("Stored chunks", total_chunks)

            table_rows: list[dict[str, Any]] = [
                {
                    "filename": record.filename,
                    "status": record.status,
                    "pages": record.page_count,
                    "chunks": record.chunk_count,
                    "strategy": record.chunking_strategy,
                    "size_kb": round(record.file_size_bytes / 1024, 1),
                    "created_at": record.created_at,
                }
                for record in records
            ]

            st.dataframe(
                table_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.caption("Per-document details")

            for record in records:
                with st.expander(
                    f"{record.filename} · {record.document_id}"
                ):
                    st.write(f"**Status:** {record.status}")
                    st.write(f"**Pages:** {record.page_count}")
                    st.write(f"**Chunks:** {record.chunk_count}")
                    st.write(
                        f"**Chunking strategy:** {record.chunking_strategy}"
                    )
                    st.write(
                        f"**Embedding model:** "
                        f"{record.embedding_model or '(unknown)'}"
                    )
                    st.write(
                        f"**File size:** {record.file_size_bytes:,} bytes"
                    )
                    st.write(f"**Created:** {record.created_at}")

                    for warning in record.warnings:
                        st.warning(warning)

                    st.caption(
                        "Document deletion is not exposed here because the "
                        "current storage and retrieval implementations must "
                        "remove vector records together to avoid stale "
                        "index data."
                    )

    # ------------------------------------------------------------------
    # Evaluation tab
    # ------------------------------------------------------------------
    with evaluation_tab:
        st.subheader("Retrieval and answer evaluation")

        st.write(
            "Run the curated JSONL question set against the indexed "
            "document corpus. Evaluation quality depends on verified "
            "expected evidence IDs and a stable ingestion/chunking "
            "configuration."
        )

        questions_path = Path(settings.evaluation_questions_file)
        if not questions_path.is_absolute():
            questions_path = PROJECT_ROOT / questions_path

        st.write(f"**Question dataset:** `{questions_path}`")

        if not questions_path.is_file():
            st.warning(
                "The evaluation dataset was not found. Create "
                "data/evaluation/questions.jsonl before running evaluation."
            )
        else:
            if st.button("Run evaluation", type="primary"):
                try:
                    questions = load_questions(questions_path)

                    if not questions:
                        st.warning(
                            "The evaluation dataset contains no questions."
                        )
                    else:
                        with st.spinner(
                            f"Evaluating {len(questions)} questions..."
                        ):
                            evaluator = Evaluator(pipeline, settings)
                            report = evaluator.run(questions)

                        st.subheader("Evaluation report")

                        if is_dataclass(report):
                            report_data = asdict(report)
                        else:
                            report_data = report

                        if isinstance(report_data, dict):
                            metrics = report_data.get("metrics", {})

                            if isinstance(metrics, dict) and metrics:
                                headline_keys = [
                                    "recall_at_k",
                                    "precision_at_k",
                                    "citation_precision",
                                    "citation_recall",
                                    "abstention_accuracy",
                                ]

                                metric_columns = st.columns(
                                    len(headline_keys)
                                )

                                for i, name in enumerate(headline_keys):
                                    value = metrics.get(name)
                                    with metric_columns[i]:
                                        if isinstance(value, (int, float)):
                                            st.metric(
                                                name.replace(
                                                    "_", " "
                                                ).title(),
                                                f"{value:.3f}",
                                            )
                                        else:
                                            st.metric(
                                                name.replace(
                                                    "_", " "
                                                ).title(),
                                                "—",
                                            )

                                with st.expander("Full metrics"):
                                    st.json(metrics)

                        by_difficulty = (
                            report_data.get("metrics_by_difficulty")
                            if isinstance(report_data, dict)
                            else None
                        )

                        if isinstance(by_difficulty, dict) and by_difficulty:
                            st.subheader("Metrics by difficulty")

                            rows = []
                            for label, block in by_difficulty.items():
                                rows.append(
                                    {
                                        "difficulty": label,
                                        "count": block.get("count"),
                                        "recall@k": block.get(
                                            "recall_at_k"
                                        ),
                                        "precision@k": block.get(
                                            "precision_at_k"
                                        ),
                                        "citation_precision": block.get(
                                            "citation_precision"
                                        ),
                                        "citation_recall": block.get(
                                            "citation_recall"
                                        ),
                                        "abstention_accuracy": block.get(
                                            "abstention_accuracy"
                                        ),
                                        "mean_latency_ms": block.get(
                                            "mean_latency_ms"
                                        ),
                                    }
                                )

                            st.dataframe(
                                rows,
                                use_container_width=True,
                                hide_index=True,
                            )

                        output_path = (
                            PROJECT_ROOT
                            / settings.reports_dir
                            / f"evaluation-"
                            f"{time.strftime('%Y%m%d-%H%M%S')}.json"
                        )

                        saved_path = evaluator.save_report(
                            report,
                            output_path,
                        )
                        st.success(
                            f"Evaluation report saved to: {saved_path}"
                        )

                        with st.expander("View complete report"):
                            st.json(report_data)

                except Exception as exc:
                    display_error("Evaluation failed.", exc)


# ---------------------------------------------------------------------------
# Page: Pipeline Observatory
# ---------------------------------------------------------------------------

def _page_observatory(
    settings: Any,
    store: DocumentStore,
    pipeline: RAGPipeline,
) -> None:
    """Live, step-by-step view of the RAG pipeline for one question."""

    st.markdown(
        """
        <div class="drugdoc-hero">
            <div class="eyebrow">Live internals · not a wrapper</div>
            <h1>🔬 Pipeline Observatory</h1>
            <p>Watch every internal step of the RAG pipeline in real time —
            query expansion, hybrid retrieval, reciprocal rank fusion, MMR
            reranking, prompt construction, generation, parsing, citation
            resolution, and self-correction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available = store.list_documents(status="indexed")
    if not available:
        st.warning(
            "Index at least one document (from the DrugDoc AI page) before "
            "running the observatory."
        )
        return

    doc_options = {
        f"{r.filename} · {r.document_id}": r.document_id for r in available
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "Question to trace",
            placeholder=(
                "e.g. What contraindications are listed on this label?"
            ),
            height=90,
            max_chars=2000,
            key="obs_question",
        )
    with col2:
        scope = st.radio(
            "Scope",
            ["All docs", "Selected"],
            key="obs_scope",
        )

    selected_ids: list[str] | None = None
    if scope == "Selected":
        chosen = st.multiselect(
            "Documents",
            options=list(doc_options.keys()),
            default=list(doc_options.keys()),
            key="obs_docs",
        )
        selected_ids = [doc_options[label] for label in chosen]

    run = st.button("▶ Run with full trace", type="primary")

    if not run:
        st.caption(
            "Every run rebuilds the full trace from scratch. Nothing is "
            "cached across runs. Use **Download trace** after a run to "
            "export the complete step-by-step record."
        )
        return

    if not question.strip():
        st.warning("Enter a question first.")
        return

    st.markdown("### Live pipeline trace")

    streamed_events: list[TraceEvent] = []

    def render_event(event: TraceEvent) -> None:
        streamed_events.append(event)
        _render_trace_event(event)

    progress = st.progress(0, text="Running pipeline…")

    try:
        response = pipeline.answer_question(
            question=question.strip(),
            document_ids=set(selected_ids) if selected_ids else None,
            trace_sink=render_event,
        )
    except TypeError as exc:
        # Retrieval layer not yet instrumented with trace_sink.
        progress.empty()
        st.error(
            "The retrieval layer does not yet accept `trace_sink`. "
            "Apply the retrieval patch, then retry. Details: "
            f"{exc}"
        )
        return
    except Exception as exc:
        progress.empty()
        display_error("Pipeline execution failed.", exc)
        return

    progress.progress(1.0, text="Pipeline complete.")

    # --- Final response ----------------------------------------------------
    st.divider()
    st.markdown("### Final response")

    _render_status_pill(response.status)
    st.markdown(
        f'<div class="drugdoc-card">{response.answer}</div>',
        unsafe_allow_html=True,
    )

    _render_confidence_meter(_confidence_score(response))

    st.caption(
        f"Evidence chunks: {response.evidence_count} · "
        f"Latency: {response.latency_ms} ms · "
        f"Retrieval source: {response.retrieval_source or '—'}"
    )

    # --- Citations ---------------------------------------------------------
    if response.citations:
        st.markdown("#### Citations")
        for c in response.citations:
            with st.expander(
                f"{c.filename} · page {c.page_number} · {c.citation_id}"
            ):
                st.caption(f"Chunk ID: `{c.chunk_id}`")
                st.write(_best_sentence_quote(c.quote, question))
                with st.popover("View full source passage"):
                    st.text(c.quote)

    # --- Download ----------------------------------------------------------
    st.download_button(
        "⬇ Download trace (JSON)",
        data=json.dumps(
            {
                "question": question,
                "response": response.to_dict(),
                "events": [e.to_dict() for e in streamed_events],
            },
            indent=2,
            default=str,
        ),
        file_name=f"drugdoc-trace-{int(time.time())}.json",
        mime="application/json",
    )

    # --- Event summary table ----------------------------------------------
    if streamed_events:
        with st.expander("Event summary (all steps)"):
            rows = [
                {
                    "step": i + 1,
                    "kind": e.kind,
                    "title": e.title,
                    "elapsed_ms": e.elapsed_ms,
                }
                for i, e in enumerate(streamed_events)
            ]
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def main() -> None:
    """Streamlit entry point."""

    try:
        settings = get_settings()
        store, retriever, pipeline = get_services()
    except Exception as exc:
        LOGGER.exception("Application initialization failed.")
        st.error(
            "DrugDoc AI could not initialize. Check configs/config.yaml, "
            "installed dependencies, and the application logs."
        )
        st.exception(exc)
        st.stop()
        return

    _render_sidebar(settings, store)

    # Named page functions ensure that Streamlit generates unique URLs.
    def drugdoc_page() -> None:
        _page_main(settings, store, retriever, pipeline)

    def observatory_page() -> None:
        _page_observatory(settings, store, pipeline)

    pages = [
        st.Page(
            drugdoc_page,
            title="DrugDoc AI",
            url_path="drugdoc",
            icon="💊",
            default=True,
        ),
        st.Page(
            observatory_page,
            title="Pipeline Observatory",
            url_path="pipeline-observatory",
            icon="🔬",
        ),
    ]

    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()