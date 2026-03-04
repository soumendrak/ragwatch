"""Retrieval-stage helpers.

Sets standard retrieval span attributes such as ``top_k`` and
``chunks.returned``.
"""

from __future__ import annotations

from opentelemetry import trace as otel_trace

from ragwatch.instrumentation.semconv import (
    RETRIEVAL_CHUNKS_RETURNED,
    RETRIEVAL_TOP_K,
)


def set_retrieval_attributes(
    span: otel_trace.Span,
    *,
    top_k: int | None = None,
    chunks_returned: int | None = None,
) -> None:
    """Set standard retrieval span attributes.

    Args:
        span: Active span.
        top_k: Number of chunks requested.
        chunks_returned: Number of chunks actually returned.
    """
    if top_k is not None:
        span.set_attribute(RETRIEVAL_TOP_K, top_k)
    if chunks_returned is not None:
        span.set_attribute(RETRIEVAL_CHUNKS_RETURNED, chunks_returned)
