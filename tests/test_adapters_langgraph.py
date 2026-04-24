"""Tests for ragwatch.adapters.langgraph — @node and @workflow decorators."""

from __future__ import annotations

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch.adapters.langgraph import node, workflow
from ragwatch.adapters.base import clear_adapters, get_adapter
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.semconv import (
    INPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_VALUE,
)


@pytest.fixture(autouse=True)
def _setup():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    yield exporter
    reset_tracer_provider()


def test_node_decorator_creates_span(_setup):
    exporter = _setup

    @node
    def retriever(state):
        return {"docs": ["doc1"]}

    retriever({"query": "hello"})

    finished = exporter.get_finished_spans()
    assert any("retriever" in s.name for s in finished)


def test_node_span_kind_agent(_setup):
    exporter = _setup

    @node
    def my_node(state):
        return state

    my_node({})

    span = next(s for s in exporter.get_finished_spans() if "my_node" in s.name)
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == "AGENT"


def test_node_with_name(_setup):
    exporter = _setup

    @node("custom-retriever")
    def retriever(state):
        return state

    retriever({})
    assert any(s.name == "custom-retriever" for s in exporter.get_finished_spans())


def test_node_auto_registers_builtin_adapter(_setup):
    clear_adapters()

    @node("auto-adapter")
    def my_node(state):
        return state

    my_node({"query": "hello"})

    assert get_adapter("langgraph") is not None


def test_workflow_decorator_creates_span(_setup):
    exporter = _setup

    @workflow
    def my_pipeline(input_data):
        return {"result": "done"}

    my_pipeline({"query": "test"})

    finished = exporter.get_finished_spans()
    assert any("my_pipeline" in s.name for s in finished)


def test_workflow_span_kind_chain(_setup):
    exporter = _setup

    @workflow
    def my_pipeline(input_data):
        return input_data

    my_pipeline({})

    span = next(s for s in exporter.get_finished_spans() if "my_pipeline" in s.name)
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == "CHAIN"


def test_node_auto_io(_setup):
    exporter = _setup

    @node
    def my_node(state):
        return {"processed": True}

    my_node({"query": "hello"})

    span = next(s for s in exporter.get_finished_spans() if "my_node" in s.name)
    assert INPUT_VALUE in span.attributes
    assert OUTPUT_VALUE in span.attributes


def test_node_auto_io_disabled(_setup):
    exporter = _setup

    @node(auto_track_io=False)
    def my_node(state):
        return {"processed": True}

    my_node({"query": "hello"})

    span = next(s for s in exporter.get_finished_spans() if "my_node" in s.name)
    assert INPUT_VALUE not in span.attributes
    assert OUTPUT_VALUE not in span.attributes


def test_nested_nodes_correct_parenting(_setup):
    exporter = _setup

    @node("parent-node")
    def parent(state):
        return child(state)

    @node("child-node")
    def child(state):
        return state

    parent({"x": 1})

    finished = exporter.get_finished_spans()
    parent_span = next(s for s in finished if s.name == "parent-node")
    child_span = next(s for s in finished if s.name == "child-node")

    assert child_span.context.trace_id == parent_span.context.trace_id
    assert child_span.parent.span_id == parent_span.context.span_id


@pytest.mark.asyncio
async def test_node_async(_setup):
    exporter = _setup

    @node("async-node")
    async def async_node(state):
        return {"result": "async"}

    result = await async_node({})
    assert result == {"result": "async"}
    assert any(s.name == "async-node" for s in exporter.get_finished_spans())


@pytest.mark.asyncio
async def test_workflow_async(_setup):
    exporter = _setup

    @workflow("async-pipeline")
    async def async_pipeline(data):
        return data

    await async_pipeline({"q": "test"})
    assert any(s.name == "async-pipeline" for s in exporter.get_finished_spans())
