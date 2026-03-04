"""Tests for ragwatch.adapters.crewai — @node and @endpoint decorators."""

from __future__ import annotations

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch.adapters.crewai import endpoint, node
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.semconv import OPENINFERENCE_SPAN_KIND


@pytest.fixture(autouse=True)
def _setup():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    yield exporter
    reset_tracer_provider()


def test_node_decorator_sync(_setup):
    exporter = _setup

    @node
    def researcher(task):
        return {"findings": "data"}

    result = researcher("find info")
    assert result == {"findings": "data"}

    finished = exporter.get_finished_spans()
    assert any("researcher" in s.name for s in finished)


@pytest.mark.asyncio
async def test_node_decorator_async(_setup):
    exporter = _setup

    @node("async-researcher")
    async def researcher(task):
        return {"findings": "async data"}

    result = await researcher("find info")
    assert result == {"findings": "async data"}
    assert any(s.name == "async-researcher" for s in exporter.get_finished_spans())


def test_node_span_kind(_setup):
    exporter = _setup

    @node
    def my_agent(task):
        return task

    my_agent("test")

    span = next(s for s in exporter.get_finished_spans() if "my_agent" in s.name)
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == "AGENT"


def test_endpoint_decorator(_setup):
    exporter = _setup

    @endpoint
    def run_crew(input_data):
        return {"output": "done"}

    result = run_crew("start")
    assert result == {"output": "done"}

    span = next(s for s in exporter.get_finished_spans() if "run_crew" in s.name)
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == "CHAIN"


def test_endpoint_with_name(_setup):
    exporter = _setup

    @endpoint("research-crew")
    def run_crew(input_data):
        return {"output": "done"}

    run_crew("start")
    assert any(s.name == "research-crew" for s in exporter.get_finished_spans())


@pytest.mark.asyncio
async def test_endpoint_async(_setup):
    exporter = _setup

    @endpoint("async-crew")
    async def run_crew(data):
        return data

    await run_crew({"q": "test"})
    assert any(s.name == "async-crew" for s in exporter.get_finished_spans())
