"""Quality-score evaluators for RAGWatch.

Provides:
- ``chunk_relevance_score`` — cosine similarity between query embedding and
  chunk embeddings.
- ``record_feedback`` — attaches ``user.feedback_score`` to a trace.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Link, SpanContext, TraceFlags, TraceState

from ragwatch.core.context import get_query_embedding
from ragwatch.core.tracer import get_tracer
from ragwatch.instrumentation.attributes import safe_set_attribute
from ragwatch.instrumentation.semconv import (
    CHUNK_RELEVANCE_SCORE,
    CHUNK_RELEVANCE_SCORES,
    USER_FEEDBACK_SCORE,
    USER_FEEDBACK_SPAN_ID,
    USER_FEEDBACK_TRACE_ID,
)

_logger = logging.getLogger(__name__)


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
            safe_set_attribute(span, CHUNK_RELEVANCE_SCORE, avg)
            safe_set_attribute(span, CHUNK_RELEVANCE_SCORES, scores)

    return scores


def _parse_otel_id(value: str, *, expected_length: int, field_name: str) -> int | None:
    """Parse a hex OTel trace/span identifier if it is linkable."""
    normalized = value.strip().lower()
    if len(normalized) != expected_length:
        return None
    try:
        parsed = int(normalized, 16)
    except ValueError:
        _logger.warning("Invalid %s for feedback link: %r", field_name, value)
        return None
    return parsed or None


def _feedback_link(trace_id: str, span_id: str | None) -> list[Link]:
    """Build an OTel link for feedback when trace and span IDs are valid."""
    if span_id is None:
        return []
    parsed_trace_id = _parse_otel_id(
        trace_id,
        expected_length=32,
        field_name="trace_id",
    )
    parsed_span_id = _parse_otel_id(
        span_id,
        expected_length=16,
        field_name="span_id",
    )
    if parsed_trace_id is None or parsed_span_id is None:
        return []
    context = SpanContext(
        trace_id=parsed_trace_id,
        span_id=parsed_span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState(),
    )
    return [Link(context)]


def record_feedback(trace_id: str, score: float, *, span_id: str | None = None) -> None:
    """Record user feedback as a separate span linked to the given trace.

    Creates a ``ragwatch.feedback`` span with ``user.feedback_score`` and
    ``user.feedback_trace_id`` attributes. When *span_id* is provided and both
    IDs are valid OTel hex IDs, the feedback span is linked to the rated span.

    Args:
        trace_id: The trace ID of the request being rated.
        score: Feedback score (typically 0.0 – 1.0).
        span_id: Optional span ID of the response span being rated. Enables an
            OpenTelemetry span link when paired with a valid trace ID.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "ragwatch.feedback",
        links=_feedback_link(trace_id, span_id),
    ) as span:
        safe_set_attribute(span, USER_FEEDBACK_SCORE, score)
        safe_set_attribute(span, USER_FEEDBACK_TRACE_ID, trace_id)
        if span_id is not None:
            safe_set_attribute(span, USER_FEEDBACK_SPAN_ID, span_id)
