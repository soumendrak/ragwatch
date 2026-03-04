"""Quality-score evaluators for RAGWatch.

Provides:
- ``chunk_relevance_score`` — cosine similarity between query embedding and
  chunk embeddings.
- ``record_feedback`` — attaches ``user.feedback_score`` to a trace.
"""

from __future__ import annotations

import math
from typing import Optional

from opentelemetry import trace as otel_trace

from ragwatch.core.context import get_query_embedding
from ragwatch.core.tracer import get_tracer
from ragwatch.instrumentation.semconv import (
    CHUNK_RELEVANCE_SCORE,
    CHUNK_RELEVANCE_SCORES,
    USER_FEEDBACK_SCORE,
    USER_FEEDBACK_TRACE_ID,
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors of equal length."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_relevance_score(
    chunk_embeddings: list[list[float]],
    query_embedding: Optional[list[float]] = None,
) -> list[float]:
    """Compute relevance scores for retrieved chunks.

    If *query_embedding* is ``None``, the embedding stored in the current
    OTel context (set during the embedding stage) is used.

    Args:
        chunk_embeddings: List of chunk embedding vectors.
        query_embedding: Optional explicit query embedding.  Falls back to
            the context-stored embedding.

    Returns:
        A list of cosine-similarity scores, one per chunk.

    Raises:
        ValueError: If no query embedding is available.
    """
    if query_embedding is None:
        query_embedding = get_query_embedding()
    if query_embedding is None:
        raise ValueError(
            "No query embedding available. Either pass query_embedding "
            "explicitly or use @trace with SpanKind.EMBEDDING to store it "
            "in context."
        )

    scores = [_cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]

    span = otel_trace.get_current_span()
    if span.is_recording():
        if scores:
            avg = sum(scores) / len(scores)
            span.set_attribute(CHUNK_RELEVANCE_SCORE, avg)
            span.set_attribute(CHUNK_RELEVANCE_SCORES, scores)

    return scores


def record_feedback(trace_id: str, score: float) -> None:
    """Record user feedback as a separate span linked to the given trace.

    Creates a ``ragwatch.feedback`` span with ``user.feedback_score`` and
    ``user.feedback_trace_id`` attributes.

    Args:
        trace_id: The trace ID of the request being rated.
        score: Feedback score (typically 0.0 – 1.0).
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("ragwatch.feedback") as span:
        span.set_attribute(USER_FEEDBACK_SCORE, score)
        span.set_attribute(USER_FEEDBACK_TRACE_ID, trace_id)
