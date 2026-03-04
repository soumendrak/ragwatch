"""Embedding-stage helpers.

When a function decorated with ``SpanKind.EMBEDDING`` returns a list of
floats, the result is automatically stored in the OTel context so that
the retrieval stage can compute ``chunk_relevance_score``.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace as otel_trace

from ragwatch.core.context import set_query_embedding
from ragwatch.instrumentation.semconv import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_DURATION_MS,
    EMBEDDING_MODEL_NAME,
)


def set_embedding_attributes(
    span: otel_trace.Span,
    *,
    model_name: str | None = None,
    dimensions: int | None = None,
    duration_ms: float | None = None,
) -> None:
    """Set standard embedding span attributes.

    Args:
        span: Active span.
        model_name: Name of the embedding model.
        dimensions: Dimensionality of the embedding.
        duration_ms: Time taken to generate the embedding in milliseconds.
    """
    if model_name is not None:
        span.set_attribute(EMBEDDING_MODEL_NAME, model_name)
    if dimensions is not None:
        span.set_attribute(EMBEDDING_DIMENSIONS, dimensions)
    if duration_ms is not None:
        span.set_attribute(EMBEDDING_DURATION_MS, duration_ms)


def store_embedding_in_context(result: Any) -> None:
    """If *result* looks like an embedding vector, store it in thread-local context.

    Args:
        result: Return value from the decorated function.
    """
    if isinstance(result, list) and result and isinstance(result[0], (int, float)):
        set_query_embedding(result)
