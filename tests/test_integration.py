"""Integration tests — full pipeline smoke tests."""

from __future__ import annotations

import math

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch import SpanKind, trace
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.context import clear_query_embedding
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.evaluators import chunk_relevance_score, record_feedback
from ragwatch.instrumentation.semconv import (
    CHUNK_RELEVANCE_SCORE,
    OPENINFERENCE_SPAN_KIND,
    USER_FEEDBACK_SCORE,
)


@pytest.fixture(autouse=True)
def _setup():
    clear_query_embedding()
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="integration", exporter=exporter), _force_flush=True
    )
    yield exporter
    clear_query_embedding()
    reset_tracer_provider()


def test_full_linear_rag_pipeline(_setup):
    """Embedding → Retrieval → Response pipeline produces correct spans."""
    exporter = _setup

    @trace("ragwatch.embedding.generate", span_kind=SpanKind.EMBEDDING)
    def embed_query(text: str) -> list[float]:
        return [0.5, 0.5, 0.0]

    @trace("ragwatch.retrieval.search", span_kind=SpanKind.RETRIEVER)
    def retrieve(query: str) -> list[dict]:
        chunk_embs = [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        scores = chunk_relevance_score(chunk_embs)
        return [{"text": f"chunk-{i}", "score": s} for i, s in enumerate(scores)]

    @trace("ragwatch.response.emit", span_kind=SpanKind.CHAIN)
    def generate_response(chunks: list[dict]) -> str:
        return f"Response based on {len(chunks)} chunks"

    # Run the pipeline
    _ = embed_query("What is RAG?")
    chunks = retrieve("What is RAG?")
    response = generate_response(chunks)

    assert len(chunks) == 3
    assert "Response based on 3 chunks" in response

    finished = exporter.get_finished_spans()
    span_names = [s.name for s in finished]
    assert "ragwatch.embedding.generate" in span_names
    assert "ragwatch.retrieval.search" in span_names
    assert "ragwatch.response.emit" in span_names


def test_full_pipeline_with_feedback(_setup):
    """record_feedback() creates a feedback span."""
    exporter = _setup

    record_feedback(trace_id="test-trace-123", score=0.9)

    finished = exporter.get_finished_spans()
    feedback_span = next(s for s in finished if s.name == "ragwatch.feedback")
    assert feedback_span.attributes[USER_FEEDBACK_SCORE] == 0.9


def test_full_pipeline_with_quality_scores(_setup):
    """Relevance scores appear on retrieval spans."""
    exporter = _setup

    @trace("ragwatch.embedding.generate", span_kind=SpanKind.EMBEDDING)
    def embed(text):
        return [1.0, 0.0, 0.0]

    @trace("ragwatch.retrieval.search", span_kind=SpanKind.RETRIEVER)
    def retrieve():
        chunk_embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        scores = chunk_relevance_score(chunk_embs)
        return scores

    embed("test query")
    scores = retrieve()

    assert math.isclose(scores[0], 1.0, abs_tol=1e-9)
    assert math.isclose(scores[1], 0.0, abs_tol=1e-9)

    retrieval_span = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "ragwatch.retrieval.search"
    )
    assert CHUNK_RELEVANCE_SCORE in retrieval_span.attributes


def test_langgraph_agent_end_to_end(_setup):
    """LangGraph adapter decorators produce correct span hierarchy."""
    exporter = _setup

    from ragwatch.adapters.langgraph import node, workflow

    @node("retrieve-node")
    def retrieve_node(state):
        return {**state, "docs": ["doc1", "doc2"]}

    @node("generate-node")
    def generate_node(state):
        return {**state, "response": "answer"}

    @workflow("rag-workflow")
    def run_pipeline(input_data):
        state = retrieve_node(input_data)
        state = generate_node(state)
        return state

    result = run_pipeline({"query": "test"})
    assert result["response"] == "answer"

    finished = exporter.get_finished_spans()
    names = [s.name for s in finished]
    assert "rag-workflow" in names
    assert "retrieve-node" in names
    assert "generate-node" in names

    workflow_span = next(s for s in finished if s.name == "rag-workflow")
    assert workflow_span.attributes[OPENINFERENCE_SPAN_KIND] == "CHAIN"

    node_span = next(s for s in finished if s.name == "retrieve-node")
    assert node_span.attributes[OPENINFERENCE_SPAN_KIND] == "AGENT"
    assert node_span.parent.span_id == workflow_span.context.span_id


def test_crewai_agent_end_to_end(_setup):
    """CrewAI adapter decorators produce correct spans."""
    exporter = _setup

    from ragwatch.adapters.crewai import endpoint, node

    @node("researcher")
    def researcher(task):
        return {"findings": "important data"}

    @node("writer")
    def writer(findings):
        return {"article": "written"}

    @endpoint("research-crew")
    def run_crew(input_data):
        findings = researcher(input_data)
        article = writer(findings)
        return article

    result = run_crew("research topic")
    assert result["article"] == "written"

    finished = exporter.get_finished_spans()
    names = [s.name for s in finished]
    assert "research-crew" in names
    assert "researcher" in names
    assert "writer" in names
