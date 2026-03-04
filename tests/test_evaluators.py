"""Tests for ragwatch.instrumentation.evaluators — quality scores."""

from __future__ import annotations

import math

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.context import clear_query_embedding, set_query_embedding
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.evaluators import (
    _cosine_similarity,
    chunk_relevance_score,
    record_feedback,
)
from ragwatch.instrumentation.semconv import (
    CHUNK_RELEVANCE_SCORE,
    USER_FEEDBACK_SCORE,
    USER_FEEDBACK_TRACE_ID,
)


@pytest.fixture(autouse=True)
def _reset():
    clear_query_embedding()
    yield
    clear_query_embedding()
    reset_tracer_provider()


def test_chunk_relevance_score_computed():
    query = [1.0, 0.0, 0.0]
    chunks = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ]
    scores = chunk_relevance_score(chunks, query_embedding=query)
    assert len(scores) == 3
    assert math.isclose(scores[0], 1.0, abs_tol=1e-9)
    assert math.isclose(scores[1], 0.0, abs_tol=1e-9)


def test_chunk_relevance_score_identical_vectors():
    v = [0.3, 0.4, 0.5]
    scores = chunk_relevance_score([v], query_embedding=v)
    assert math.isclose(scores[0], 1.0, abs_tol=1e-9)


def test_chunk_relevance_score_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    scores = chunk_relevance_score([b], query_embedding=a)
    assert math.isclose(scores[0], 0.0, abs_tol=1e-9)


def test_chunk_relevance_score_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    scores = chunk_relevance_score([b], query_embedding=a)
    assert math.isclose(scores[0], -1.0, abs_tol=1e-9)


def test_chunk_relevance_score_from_context():
    query = [1.0, 0.0, 0.0]
    set_query_embedding(query)
    try:
        chunks = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        scores = chunk_relevance_score(chunks)
        assert math.isclose(scores[0], 1.0, abs_tol=1e-9)
        assert math.isclose(scores[1], 0.0, abs_tol=1e-9)
    finally:
        clear_query_embedding()


def test_chunk_relevance_score_raises_without_embedding():
    with pytest.raises(ValueError, match="No query embedding available"):
        chunk_relevance_score([[1.0, 0.0]])


def test_chunk_relevance_score_sets_span_attribute():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    from ragwatch import trace, SpanKind

    @trace("retrieval", span_kind=SpanKind.RETRIEVER)
    def retrieve():
        query = [1.0, 0.0]
        chunks = [[1.0, 0.0], [0.0, 1.0]]
        return chunk_relevance_score(chunks, query_embedding=query)

    scores = retrieve()
    assert len(scores) == 2

    span = next(s for s in exporter.get_finished_spans() if s.name == "retrieval")
    assert CHUNK_RELEVANCE_SCORE in span.attributes


def test_record_feedback_sets_attribute():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    record_feedback(trace_id="abc123", score=0.85)

    span = next(
        s for s in exporter.get_finished_spans() if s.name == "ragwatch.feedback"
    )
    assert span.attributes[USER_FEEDBACK_SCORE] == 0.85
    assert span.attributes[USER_FEEDBACK_TRACE_ID] == "abc123"


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
