"""Lightweight, dependency-free pipeline tracing.

`TraceEvent` is a plain dataclass. `trace_sink` is an optional callable
passed into the pipeline. When provided, every internal step emits an
event before continuing. When None, there is zero overhead beyond a
single `if` check per step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Event kinds used by the UI to group and color steps.
KIND_QUERY = "query"
KIND_EXPANSION = "expansion"
KIND_RETRIEVAL_DENSE = "retrieval.dense"
KIND_RETRIEVAL_SPARSE = "retrieval.sparse"
KIND_FUSION = "fusion"
KIND_RECURSION = "recursion"
KIND_MMR = "mmr"
KIND_PROMPT = "prompt"
KIND_LLM_RAW = "llm.raw"
KIND_PARSED = "parsed"
KIND_CITATIONS = "citations"
KIND_VERIFICATION = "verification"
KIND_FINAL = "final"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One observable step in the pipeline."""

    kind: str
    title: str
    elapsed_ms: int
    payload: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "elapsed_ms": self.elapsed_ms,
            "payload": self.payload,
            "notes": self.notes,
        }


# A trace sink is any callable that accepts a TraceEvent.
TraceSink = Callable[[TraceEvent], None]


class TraceCollector:
    """A convenience sink that stores events in a list.

    Useful in tests and when the caller wants the events *after* the
    call returns rather than streaming them.
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._started_at = time.perf_counter()

    def __call__(self, event: TraceEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()
        self._started_at = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._started_at) * 1000)