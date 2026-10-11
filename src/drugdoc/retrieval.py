"""Hybrid evidence retrieval for DrugDoc AI.

Supports:
    * Dense vector search via Qdrant.
    * Sparse BM25 keyword search via rank_bm25.
    * Reciprocal Rank Fusion (RRF) to combine ranked lists.
    * Multi-query expansion (LLM rewrites the question N times).
    * Recursive retrieval (re-query using the top passage).
    * Maximal Marginal Relevance (MMR) diversity reranking.

All retrieved results retain the original chunk IDs and source metadata
for citation resolution in the RAG layer.

Tracing:
    ``retrieve(..., trace_sink=callable)`` emits a TraceEvent at every
    observable retrieval step. When ``trace_sink`` is None (the default),
    behavior is identical to a build without tracing.
"""

from __future__ import annotations

import logging
import re
import uuid
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from drugdoc.config import Settings, get_settings
from drugdoc.ingestion import DocumentChunk
from drugdoc.tracing import (
    KIND_EXPANSION,
    KIND_FUSION,
    KIND_MMR,
    KIND_RECURSION,
    KIND_RETRIEVAL_DENSE,
    KIND_RETRIEVAL_SPARSE,
    TraceEvent,
    TraceSink,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RetrievalError(Exception):
    """Raised when evidence retrieval cannot be completed."""


# ---------------------------------------------------------------------------
# Retrieved chunk
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved evidence passage with source and ranking metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    source: str = "hybrid"
    retrieval_trace: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "source": self.source,
            "retrieval_trace": list(self.retrieval_trace),
        }


# ---------------------------------------------------------------------------
# Tracing helper
# ---------------------------------------------------------------------------

def _emit(
    sink: TraceSink | None,
    kind: str,
    title: str,
    payload: dict[str, Any] | None = None,
    *,
    elapsed_ms: int = 0,
    notes: str = "",
) -> None:
    """Emit a TraceEvent when a sink is provided, otherwise no-op."""
    if sink is None:
        return

    try:
        sink(
            TraceEvent(
                kind=kind,
                title=title,
                elapsed_ms=elapsed_ms,
                payload=payload or {},
                notes=notes,
            )
        )
    except Exception:
        # Tracing must never break retrieval.
        LOGGER.exception("trace_sink raised in retrieval; continuing.")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenize text while retaining drug names, numbers, and decimals."""
    return re.findall(
        r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|\d+(?:\.\d+)?%?|[a-zA-Z0-9]+",
        text.lower(),
    )


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked chunk-ID lists using Reciprocal Rank Fusion."""

    if k < 1:
        raise ValueError("RRF k must be positive.")

    fused_scores: dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()

        for rank, chunk_id in enumerate(ranked_list, start=1):
            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)
            fused_scores[chunk_id] += 1.0 / (k + rank)

    return dict(
        sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, safe for zero-length inputs."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / ((norm_a**0.5) * (norm_b**0.5))


def _mmr_rerank(
    query_vector: Sequence[float],
    candidates: list[tuple[str, float]],
    candidate_vectors: Mapping[str, Sequence[float]],
    *,
    lambda_param: float,
    top_k: int,
) -> list[tuple[str, float]]:
    """Rerank candidates using Maximal Marginal Relevance."""

    if top_k < 1 or not candidates:
        return []

    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be between 0.0 and 1.0.")

    scores = [score for _, score in candidates]
    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score if max_score > min_score else 1.0

    normalized: dict[str, float] = {
        chunk_id: (score - min_score) / span
        for chunk_id, score in candidates
    }

    selected: list[tuple[str, float]] = []
    remaining = [chunk_id for chunk_id, _ in candidates]
    selected_ids: set[str] = set()

    while remaining and len(selected) < top_k:
        best_id: str | None = None
        best_score = float("-inf")

        for chunk_id in remaining:
            vector = candidate_vectors.get(chunk_id)
            relevance = normalized[chunk_id]

            if vector is None:
                mmr_score = relevance
            else:
                if selected_ids:
                    redundancy = max(
                        _cosine_similarity(
                            vector,
                            candidate_vectors.get(sel_id, ()),
                        )
                        for sel_id in selected_ids
                    )
                else:
                    redundancy = 0.0

                mmr_score = (
                    lambda_param * relevance
                    - (1.0 - lambda_param) * redundancy
                )

            if mmr_score > best_score:
                best_score = mmr_score
                best_id = chunk_id

        if best_id is None:
            break

        selected.append((best_id, dict(candidates)[best_id]))
        selected_ids.add(best_id)
        remaining.remove(best_id)

    return selected


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Build and query dense and sparse indexes over document chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else get_settings()

        if not (
            self.settings.dense_retrieval_enabled
            or self.settings.sparse_retrieval_enabled
        ):
            raise ValueError("At least one retrieval mode must be enabled.")

        self._chunks: dict[str, DocumentChunk] = {}
        self._ordered_ids: list[str] = []

        self._bm25: Any | None = None
        self._tokenized_corpus: list[list[str]] = []

        self._dense_ready = False
        self._sparse_ready = False

        self._embedding_model: Any | None = None
        self._qdrant_client: Any | None = None
        self._qdrant_models: Any | None = None
        self._collection_ready = False

        self._chunk_vectors: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Model and Qdrant lifecycle
    # ------------------------------------------------------------------

    def _load_embedding_model(self) -> None:
        """Load the configured sentence-transformer model on demand."""
        if self._embedding_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Could not load embedding model "
                f"{self.settings.embedding_model!r}: {exc}"
            ) from exc

    def _connect_qdrant(self) -> None:
        """Initialize the Qdrant client and models exactly once."""
        if self._qdrant_client is not None and self._qdrant_models is not None:
            return

        try:
            from qdrant_client import QdrantClient, models

            if self.settings.vector_store_type == "qdrant_local":
                qdrant_path = Path(self.settings.qdrant_path)
                qdrant_path.mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(qdrant_path))
            else:
                client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key or None,
                    timeout=10,
                )

            self._qdrant_client = client
            self._qdrant_models = models

        except Exception as exc:
            self._qdrant_client = None
            self._qdrant_models = None
            raise RetrievalError(f"Could not initialize Qdrant: {exc}") from exc

    def _require_qdrant(self) -> tuple[Any, Any]:
        """Return initialized Qdrant objects or raise a clear error."""
        self._connect_qdrant()

        client = self._qdrant_client
        models = self._qdrant_models

        if client is None or models is None:
            raise RetrievalError(
                "Qdrant initialization completed without a usable client "
                "and model module."
            )

        return client, models

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the collection if absent; validate its vector dimensions."""
        if vector_size < 1:
            raise RetrievalError("Embedding dimension must be positive.")

        client, models = self._require_qdrant()
        collection_name = self.settings.qdrant_collection

        try:
            collections_response = client.get_collections()
            collections = collections_response.collections

            exists = any(
                collection.name == collection_name
                for collection in collections
            )

            if not exists:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            else:
                info = client.get_collection(collection_name)
                vector_config = info.config.params.vectors

                if isinstance(vector_config, dict):
                    raise RetrievalError(
                        "The configured Qdrant collection uses named vectors, "
                        "but this retriever expects one unnamed dense vector."
                    )

                existing_size = vector_config.size
                if existing_size != vector_size:
                    raise RetrievalError(
                        f"Qdrant collection {collection_name!r} uses "
                        f"{existing_size}-dimensional vectors, but the current "
                        f"embedding model produces {vector_size}. Use a new "
                        "collection name or rebuild the old collection."
                    )

            self._collection_ready = True

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Could not prepare Qdrant collection "
                f"{collection_name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate normalized embeddings for a sequence of texts."""
        if not texts:
            return []

        self._load_embedding_model()
        model = self._embedding_model

        if model is None:
            raise RetrievalError("Embedding model is not initialized.")

        try:
            vectors = model.encode(
                list(texts),
                batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

        except Exception as exc:
            raise RetrievalError(
                f"Embedding generation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        """Index chunks and return the number of unique chunks processed."""

        supplied_chunks = list(chunks)

        if not supplied_chunks:
            raise RetrievalError("Cannot index an empty chunk collection.")

        unique_chunks: dict[str, DocumentChunk] = {}

        for chunk in supplied_chunks:
            if not chunk.text or not chunk.text.strip():
                raise RetrievalError(
                    f"Chunk {chunk.chunk_id!r} contains no usable text."
                )

            if not chunk.chunk_id or not chunk.document_id:
                raise RetrievalError(
                    "Every chunk must have a chunk_id and document_id."
                )

            unique_chunks[chunk.chunk_id] = chunk

        self._chunks = dict(unique_chunks)
        self._ordered_ids = list(unique_chunks.keys())

        corpus_chunks = [
            self._chunks[chunk_id] for chunk_id in self._ordered_ids
        ]

        # ----- Sparse (BM25) index -----
        if self.settings.sparse_retrieval_enabled:
            try:
                from rank_bm25 import BM25Okapi

                self._tokenized_corpus = [
                    _tokenize(chunk.text) for chunk in corpus_chunks
                ]

                if not any(self._tokenized_corpus):
                    raise RetrievalError(
                        "No searchable tokens were found in the supplied "
                        "chunks."
                    )

                self._bm25 = BM25Okapi(self._tokenized_corpus)
                self._sparse_ready = True

            except RetrievalError:
                raise
            except Exception as exc:
                self._bm25 = None
                self._sparse_ready = False
                raise RetrievalError(
                    f"Could not build the BM25 index: {exc}"
                ) from exc
        else:
            self._bm25 = None
            self._tokenized_corpus = []
            self._sparse_ready = False

        # ----- Dense (Qdrant) index -----
        if self.settings.dense_retrieval_enabled:
            vectors = self._encode([chunk.text for chunk in corpus_chunks])

            if not vectors:
                raise RetrievalError("The embedding model returned no vectors.")

            vector_size = len(vectors[0])
            if any(len(vector) != vector_size for vector in vectors):
                raise RetrievalError(
                    "The embedding model returned inconsistent vector "
                    "dimensions."
                )

            self._ensure_collection(vector_size)
            client, models = self._require_qdrant()

            self._chunk_vectors = {
                chunk.chunk_id: vector
                for chunk, vector in zip(corpus_chunks, vectors)
            }

            try:
                points = []

                for chunk, vector in zip(corpus_chunks, vectors):
                    point_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"drugdoc:{chunk.chunk_id}",
                        )
                    )

                    payload = chunk.to_dict()
                    payload["chunk_id"] = chunk.chunk_id

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    )

                batch_size = max(
                    1,
                    int(self.settings.embedding_batch_size),
                )

                for offset in range(0, len(points), batch_size):
                    client.upsert(
                        collection_name=self.settings.qdrant_collection,
                        points=points[offset : offset + batch_size],
                        wait=True,
                    )

                self._dense_ready = True

            except Exception as exc:
                self._dense_ready = False
                raise RetrievalError(
                    f"Could not index vectors in Qdrant: {exc}"
                ) from exc
        else:
            self._dense_ready = False
            self._chunk_vectors = {}

        LOGGER.info(
            "Indexed %d unique chunks (dense=%s, sparse=%s).",
            len(unique_chunks),
            self.settings.dense_retrieval_enabled,
            self.settings.sparse_retrieval_enabled,
        )

        return len(unique_chunks)

    # ------------------------------------------------------------------
    # Low-level search primitives
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query: str,
        limit: int,
        document_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return original chunk IDs and similarity scores from Qdrant."""
        if not self._dense_ready or limit < 1:
            return []

        query_vector = self._encode([query])[0]
        client, models = self._require_qdrant()

        query_filter = None
        if document_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=list(document_ids)),
                    )
                ]
            )

        try:
            response = client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            results: list[tuple[str, float]] = []

            for point in response.points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")

                if not isinstance(chunk_id, str) or not chunk_id:
                    LOGGER.warning(
                        "Ignoring Qdrant point %s without a valid chunk_id.",
                        point.id,
                    )
                    continue

                results.append((chunk_id, float(point.score)))

            return results

        except Exception as exc:
            raise RetrievalError(f"Dense search failed: {exc}") from exc

    def _sparse_search(
        self,
        query: str,
        limit: int,
        document_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return BM25-ranked chunk IDs and raw lexical scores."""
        if not self._sparse_ready or self._bm25 is None or limit < 1:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        try:
            scores = self._bm25.get_scores(query_tokens)
        except Exception as exc:
            raise RetrievalError(f"BM25 search failed: {exc}") from exc

        ranked: list[tuple[str, float]] = []

        for index, score in enumerate(scores):
            if index >= len(self._ordered_ids):
                break

            chunk_id = self._ordered_ids[index]
            chunk = self._chunks.get(chunk_id)

            if chunk is None:
                continue

            if document_ids and chunk.document_id not in document_ids:
                continue

            ranked.append((chunk_id, float(score)))

        ranked.sort(key=lambda item: (-item[1], item[0]))

        return [
            (chunk_id, score)
            for chunk_id, score in ranked
            if score > 0
        ][:limit]

    # ------------------------------------------------------------------
    # Multi-query expansion
    # ------------------------------------------------------------------

    def _expand_queries(self, query: str) -> list[str]:
        """Use the LLM to generate alternate phrasings of the query."""

        if not self.settings.retrieval_multi_query_enabled:
            return [query]

        count = max(1, self.settings.retrieval_multi_query_count)

        try:
            from drugdoc.rag import (
                _generate_with_ollama,
                _generate_with_openai,
            )

            prompt = (
                "You rewrite search queries for a pharmaceutical drug-label "
                "retrieval system. Produce {count} short, distinct "
                "reformulations of the user's question. Preserve medical "
                "terminology, drug names, dosages, and units. Do not answer "
                "the question.\n\n"
                "QUESTION:\n{query}\n\n"
                "Return one reformulation per line. No numbering, no bullets, "
                "no extra text."
            ).format(count=count, query=query.strip())

            if self.settings.llm_provider == "ollama":
                raw = _generate_with_ollama(prompt, self.settings)
            elif self.settings.llm_provider == "openai":
                raw = _generate_with_openai(prompt, self.settings)
            else:
                return [query]

        except Exception as exc:
            LOGGER.warning(
                "Multi-query expansion skipped (%s).", exc
            )
            return [query]

        variants: list[str] = []

        for line in (raw or "").splitlines():
            cleaned = line.strip().lstrip("-•*0123456789. )").strip()
            if cleaned and cleaned.lower() != query.strip().lower():
                variants.append(cleaned)

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            key = variant.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(variant)
            if len(deduped) >= count:
                break

        return [query, *deduped]

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def _retrieve_single_query(
        self,
        query: str,
        *,
        candidate_limit: int,
        document_ids: set[str] | None,
    ) -> dict[str, float]:
        """Retrieve and fuse for one query string. Returns chunk_id -> score."""

        dense_results = (
            self._dense_search(query, candidate_limit, document_ids)
            if self.settings.dense_retrieval_enabled
            else []
        )

        sparse_results = (
            self._sparse_search(query, candidate_limit, document_ids)
            if self.settings.sparse_retrieval_enabled
            else []
        )

        dense_ranked_ids = [chunk_id for chunk_id, _ in dense_results]
        sparse_ranked_ids = [chunk_id for chunk_id, _ in sparse_results]

        ranked_lists = [
            ranked_ids
            for ranked_ids in (dense_ranked_ids, sparse_ranked_ids)
            if ranked_ids
        ]

        if not ranked_lists:
            return {}

        return reciprocal_rank_fusion(
            ranked_lists,
            k=self.settings.rrf_k,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: set[str] | None = None,
        trace_sink: TraceSink | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and rank evidence passages relevant to a question.

        Pipeline:

            1. Optionally expand the query into multiple variants.
            2. Retrieve with each variant; fuse all ranked lists with RRF.
            3. Optionally recurse using the top passage as extra context.
            4. Optionally rerank the final set with MMR for diversity.
            5. Return the top ``top_k`` results with a retrieval trace.
        """

        query = query.strip()

        if not query:
            raise ValueError("Retrieval query cannot be empty.")

        if not self._chunks:
            raise RetrievalError(
                "No chunks have been indexed in this retriever instance. "
                "Ingest and index the documents first."
            )

        result_limit = (
            top_k if top_k is not None else self.settings.retrieval_top_k
        )

        if result_limit < 1:
            raise ValueError("top_k must be positive.")

        candidate_limit = max(
            result_limit,
            self.settings.dense_candidate_k,
            self.settings.sparse_candidate_k,
        )

        trace: list[str] = []

        # ------------------------------------------------------------------
        # Step 1: Multi-query expansion
        # ------------------------------------------------------------------
        expansion_started = time.perf_counter()
        queries = self._expand_queries(query)
        expansion_elapsed = int(
            (time.perf_counter() - expansion_started) * 1000
        )

        if len(queries) > 1:
            trace.append(f"Expanded query into {len(queries)} variants.")
        else:
            trace.append("Using original query.")

        _emit(
            trace_sink,
            KIND_EXPANSION,
            f"Query expanded into {len(queries)} variant(s)",
            payload={
                "original": query,
                "variants": queries,
            },
            elapsed_ms=expansion_elapsed,
            notes=(
                "Multi-query disabled."
                if not self.settings.retrieval_multi_query_enabled
                else ""
            ),
        )

        # ------------------------------------------------------------------
        # Step 2: Retrieve + fuse for each variant
        # ------------------------------------------------------------------
        per_query_rankings: list[list[str]] = []
        all_dense: list[tuple[str, float]] = []
        all_sparse: list[tuple[str, float]] = []

        for variant in queries:
            dense_results = (
                self._dense_search(variant, candidate_limit, document_ids)
                if self.settings.dense_retrieval_enabled
                else []
            )
            sparse_results = (
                self._sparse_search(variant, candidate_limit, document_ids)
                if self.settings.sparse_retrieval_enabled
                else []
            )

            all_dense.extend(dense_results)
            all_sparse.extend(sparse_results)

            fused = self._retrieve_single_query(
                variant,
                candidate_limit=candidate_limit,
                document_ids=document_ids,
            )
            if fused:
                per_query_rankings.append(list(fused.keys()))

        # --- Dense events ---
        if self.settings.dense_retrieval_enabled:
            _emit(
                trace_sink,
                KIND_RETRIEVAL_DENSE,
                f"Dense candidates from Qdrant ({len(all_dense)})",
                payload={
                    "candidate_limit": candidate_limit,
                    "results": [
                        {"chunk_id": cid, "score": round(score, 4)}
                        for cid, score in all_dense[:20]
                    ],
                },
                notes=(
                    "Dense retrieval disabled."
                    if not self.settings.dense_retrieval_enabled
                    else ""
                ),
            )

        # --- Sparse events ---
        if self.settings.sparse_retrieval_enabled:
            _emit(
                trace_sink,
                KIND_RETRIEVAL_SPARSE,
                f"BM25 candidates ({len(all_sparse)})",
                payload={
                    "candidate_limit": candidate_limit,
                    "results": [
                        {"chunk_id": cid, "score": round(score, 4)}
                        for cid, score in all_sparse[:20]
                    ],
                },
                notes=(
                    "Sparse retrieval disabled."
                    if not self.settings.sparse_retrieval_enabled
                    else ""
                ),
            )

        if not per_query_rankings:
            trace.append("No candidates found in first retrieval pass.")
            _emit(
                trace_sink,
                KIND_FUSION,
                "No candidates to fuse",
                payload={},
            )
            return []

        combined_scores = reciprocal_rank_fusion(
            per_query_rankings,
            k=self.settings.rrf_k,
        )

        trace.append(
            f"Fused candidates from {len(per_query_rankings)} ranked "
            f"list(s): {len(combined_scores)} unique chunk(s)."
        )

        _emit(
            trace_sink,
            KIND_FUSION,
            f"Reciprocal Rank Fusion (k={self.settings.rrf_k})",
            payload={
                "ranked_lists": len(per_query_rankings),
                "unique_chunks": len(combined_scores),
                "top_fused": [
                    {"chunk_id": cid, "score": round(score, 6)}
                    for cid, score in list(combined_scores.items())[:15]
                ],
            },
        )

        # ------------------------------------------------------------------
        # Step 3: Recursive retrieval
        # ------------------------------------------------------------------
        if (
            self.settings.retrieval_recursive_enabled
            and len(combined_scores)
            < self.settings.retrieval_recursive_min_chunks
            and len(combined_scores) > 0
        ):
            top_chunk_id = next(iter(combined_scores))
            top_chunk = self._chunks.get(top_chunk_id)

            if top_chunk is not None:
                expanded_query = (
                    f"{query} {top_chunk.text[:300]}"
                ).strip()

                trace.append(
                    "Recursive pass: re-querying with top passage as "
                    "additional context."
                )

                recursive_started = time.perf_counter()
                recursive_scores = self._retrieve_single_query(
                    expanded_query,
                    candidate_limit=candidate_limit,
                    document_ids=document_ids,
                )
                recursive_elapsed = int(
                    (time.perf_counter() - recursive_started) * 1000
                )

                if recursive_scores:
                    combined_scores = reciprocal_rank_fusion(
                        [
                            list(combined_scores.keys()),
                            list(recursive_scores.keys()),
                        ],
                        k=self.settings.rrf_k,
                    )
                    trace.append(
                        f"Recursive pass added candidates: "
                        f"{len(combined_scores)} total."
                    )

                _emit(
                    trace_sink,
                    KIND_RECURSION,
                    "Recursive retrieval pass",
                    payload={
                        "expanded_query": expanded_query[:400],
                        "recursive_candidates": list(
                            recursive_scores.keys()
                        )[:15],
                        "combined_total": len(combined_scores),
                    },
                    elapsed_ms=recursive_elapsed,
                )

        # ------------------------------------------------------------------
        # Step 4: Build final ranked list
        # ------------------------------------------------------------------
        ranked_pairs: list[tuple[str, float]] = [
            (chunk_id, score)
            for chunk_id, score in combined_scores.items()
            if chunk_id in self._chunks
        ]

        # ------------------------------------------------------------------
        # Step 5: MMR diversity reranking
        # ------------------------------------------------------------------
        if (
            self.settings.retrieval_mmr_enabled
            and len(ranked_pairs) > result_limit
            and self._chunk_vectors
        ):
            try:
                mmr_started = time.perf_counter()
                query_vector = self._encode([query])[0]
                before_ids = [cid for cid, _ in ranked_pairs]
                reranked = _mmr_rerank(
                    query_vector=query_vector,
                    candidates=ranked_pairs,
                    candidate_vectors=self._chunk_vectors,
                    lambda_param=self.settings.retrieval_mmr_lambda,
                    top_k=result_limit,
                )
                mmr_elapsed = int(
                    (time.perf_counter() - mmr_started) * 1000
                )
                after_ids = [cid for cid, _ in reranked]

                trace.append(
                    f"MMR reranked {len(ranked_pairs)} candidates "
                    f"down to {len(reranked)} diverse chunks."
                )

                _emit(
                    trace_sink,
                    KIND_MMR,
                    f"MMR reranked {len(ranked_pairs)} → {len(reranked)}",
                    payload={
                        "lambda": self.settings.retrieval_mmr_lambda,
                        "before": before_ids[:20],
                        "after": after_ids,
                        "dropped": [
                            cid for cid in before_ids if cid not in set(after_ids)
                        ][:20],
                    },
                    elapsed_ms=mmr_elapsed,
                    notes=(
                        f"λ={self.settings.retrieval_mmr_lambda}"
                    ),
                )

                ranked_pairs = reranked
            except Exception as exc:
                LOGGER.warning("MMR reranking skipped: %s", exc)
                trace.append("MMR reranking skipped due to error.")
                _emit(
                    trace_sink,
                    KIND_MMR,
                    "MMR reranking skipped",
                    payload={"error": str(exc)},
                )

        ranked_pairs = ranked_pairs[:result_limit]

        # ------------------------------------------------------------------
        # Step 6: Build RetrievedChunk objects
        # ------------------------------------------------------------------
        results: list[RetrievedChunk] = []
        trace_tuple = tuple(trace)

        if (
            self.settings.dense_retrieval_enabled
            and self.settings.sparse_retrieval_enabled
        ):
            source_label = "hybrid"
        elif self.settings.dense_retrieval_enabled:
            source_label = "dense"
        else:
            source_label = "sparse"

        for chunk_id, fused_score in ranked_pairs:
            chunk = self._chunks.get(chunk_id)

            if chunk is None:
                continue

            if document_ids and chunk.document_id not in document_ids:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=fused_score,
                    dense_score=None,
                    sparse_score=None,
                    source=source_label,
                    retrieval_trace=trace_tuple,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "RetrievalError",
    "RetrievedChunk",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "_tokenize",
]