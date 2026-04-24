"""Tests for ragwatch.adapters.crewai — @node and @endpoint decorators."""

from __future__ import annotations

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch.adapters.crewai import CrewAIAdapter, endpoint, node
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


class TestCrewAINormalizeResult:
    def test_normalize_task_output_dict(self):
        adapter = CrewAIAdapter()
        result = {"task_output": "The answer is 42", "status": "success"}
        norm = adapter.normalize_result(result, {})
        assert norm is not None
        assert norm["agent_answer"] == "The answer is 42"
        assert norm["is_fallback"] is False

    def test_normalize_output_key(self):
        adapter = CrewAIAdapter()
        result = {"output": "Summary of findings", "status": "done"}
        norm = adapter.normalize_result(result, {})
        assert norm["agent_answer"] == "Summary of findings"

    def test_normalize_tools_used(self):
        adapter = CrewAIAdapter()
        result = {"task_output": "answer", "tools_used": ["search", "calculator"]}
        norm = adapter.normalize_result(result, {})
        assert norm["tool_calls"] == [{"name": "search"}, {"name": "calculator"}]

    def test_normalize_tools_used_as_dicts(self):
        adapter = CrewAIAdapter()
        tc = [{"name": "search", "args": {"q": "test"}}]
        result = {"task_output": "answer", "tools_used": tc}
        norm = adapter.normalize_result(result, {})
        assert norm["tool_calls"] == tc

    def test_normalize_fallback_on_error_status(self):
        adapter = CrewAIAdapter()
        result = {"task_output": "", "status": "error"}
        norm = adapter.normalize_result(result, {})
        assert norm["is_fallback"] is True

    def test_normalize_task_output_object_with_raw(self):
        """CrewAI TaskOutput-like object with .raw attribute."""

        class FakeTaskOutput:
            raw = "The raw answer"

        adapter = CrewAIAdapter()
        norm = adapter.normalize_result(FakeTaskOutput(), {})
        assert norm is not None
        assert norm["agent_answer"] == "The raw answer"

    def test_normalize_task_output_object_with_output(self):
        """CrewAI TaskOutput-like object with .output attribute."""

        class FakeTaskOutput:
            output = "The output"

        adapter = CrewAIAdapter()
        norm = adapter.normalize_result(FakeTaskOutput(), {})
        assert norm["agent_answer"] == "The output"

    def test_normalize_plain_string_returns_none(self):
        adapter = CrewAIAdapter()
        assert adapter.normalize_result("just a string", {}) is None

    def test_normalize_empty_dict_returns_none(self):
        adapter = CrewAIAdapter()
        assert adapter.normalize_result({}, {}) is None

    def test_normalize_empty_tools_list_not_included(self):
        adapter = CrewAIAdapter()
        result = {"task_output": "answer", "tools_used": []}
        norm = adapter.normalize_result(result, {})
        assert "tool_calls" not in norm
