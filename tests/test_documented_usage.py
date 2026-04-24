"""Compatibility tests for documented public usage patterns."""

from __future__ import annotations

from tests.conftest import InMemorySpanExporter

import ragwatch
from ragwatch import RAGWatchConfig, SpanKind, trace
from ragwatch.adapters.langgraph import node
from ragwatch.core.tracer import get_tracer_provider, reset_tracer_provider
from ragwatch.instrumentation.evaluators import chunk_relevance_score


def _flush_spans() -> None:
    provider = get_tracer_provider()
    if provider is not None:
        provider.force_flush()


def test_readme_minimal_rag_pipeline_usage():
    exporter = InMemorySpanExporter()
    ragwatch.configure(RAGWatchConfig(service_name="docs-minimal", exporter=exporter))

    @trace("embedding.generate", span_kind=SpanKind.EMBEDDING)
    def embed_query(text: str) -> list[float]:
        return [0.5, 0.3, 0.2]

    @trace("retrieval.search", span_kind=SpanKind.RETRIEVER)
    def retrieve_chunks(query: str) -> list[dict]:
        chunk_embeddings = [[0.5, 0.3, 0.2], [0.1, 0.9, 0.0]]
        scores = chunk_relevance_score(chunk_embeddings)
        return [{"text": "chunk", "score": score} for score in scores]

    @trace("response.generate", span_kind=SpanKind.CHAIN)
    def generate_response(chunks: list[dict]) -> str:
        return f"Generated response from {len(chunks)} chunks"

    embed_query("What is RAG?")
    chunks = retrieve_chunks("What is RAG?")
    response = generate_response(chunks)
    _flush_spans()

    assert response == "Generated response from 2 chunks"
    span_names = {span.name for span in exporter.get_finished_spans()}
    assert {
        "embedding.generate",
        "retrieval.search",
        "response.generate",
    } <= span_names

    reset_tracer_provider()


def test_langgraph_decorator_usage_without_manual_adapter_registration():
    exporter = InMemorySpanExporter()
    ragwatch.configure(RAGWatchConfig(service_name="docs-langgraph", exporter=exporter))

    @node("retrieve-node", telemetry=["agent_completion"])
    def retrieve_node(state: dict) -> dict:
        return {**state, "final_answer": "answer"}

    result = retrieve_node({"question": "What?"})
    _flush_spans()

    assert result["final_answer"] == "answer"
    span = next(
        span for span in exporter.get_finished_spans() if span.name == "retrieve-node"
    )
    assert span.attributes["agent.completion_status"] == "success"

    reset_tracer_provider()


def test_runtime_trace_usage_from_api_reference():
    exporter = InMemorySpanExporter()
    runtime = ragwatch.configure(
        RAGWatchConfig(service_name="docs-runtime", exporter=exporter)
    )

    @runtime.trace("runtime-span")
    def process() -> str:
        return "ok"

    assert process() == "ok"
    _flush_spans()
    assert any(span.name == "runtime-span" for span in exporter.get_finished_spans())

    reset_tracer_provider()
