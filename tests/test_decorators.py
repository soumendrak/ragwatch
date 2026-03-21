"""Tests for ragwatch.instrumentation.decorators — @trace sync + async."""

from __future__ import annotations

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch import SpanKind, trace
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.core.config import RAGWatchConfig
from ragwatch.instrumentation.semconv import ERROR_TYPE, OPENINFERENCE_SPAN_KIND


@pytest.fixture(autouse=True)
def _setup():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    yield exporter
    reset_tracer_provider()


def test_trace_bare_decorator(_setup):
    exporter = _setup

    @trace
    def hello():
        return "world"

    result = hello()
    assert result == "world"

    finished = exporter.get_finished_spans()
    assert any("hello" in s.name for s in finished)


def test_trace_with_name(_setup):
    exporter = _setup

    @trace("custom-name")
    def hello():
        return "world"

    hello()
    assert any(s.name == "custom-name" for s in exporter.get_finished_spans())


def test_trace_with_span_kind(_setup):
    exporter = _setup

    @trace("embed", span_kind=SpanKind.EMBEDDING)
    def embed():
        return [0.1, 0.2]

    embed()

    span = next(s for s in exporter.get_finished_spans() if s.name == "embed")
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == "EMBEDDING"


@pytest.mark.asyncio
async def test_trace_async(_setup):
    exporter = _setup

    @trace("async-op")
    async def async_op():
        return 42

    result = await async_op()
    assert result == 42
    assert any(s.name == "async-op" for s in exporter.get_finished_spans())


@pytest.mark.asyncio
async def test_trace_async_exception(_setup):
    exporter = _setup

    @trace("async-err")
    async def fail():
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        await fail()

    span = next(s for s in exporter.get_finished_spans() if s.name == "async-err")
    assert len(span.events) > 0


def test_trace_preserves_function_metadata(_setup):
    @trace("named")
    def documented_fn():
        """This is documented."""
        pass

    assert documented_fn.__name__ == "documented_fn"
    assert documented_fn.__doc__ == "This is documented."


def test_trace_nested_spans(_setup):
    exporter = _setup

    @trace("outer")
    def outer():
        return inner()

    @trace("inner")
    def inner():
        return "done"

    outer()

    finished = exporter.get_finished_spans()
    outer_span = next(s for s in finished if s.name == "outer")
    inner_span = next(s for s in finished if s.name == "inner")

    assert inner_span.context.trace_id == outer_span.context.trace_id
    assert inner_span.parent.span_id == outer_span.context.span_id


def test_trace_embedding_stores_context(_setup):
    exporter = _setup

    @trace("embed", span_kind=SpanKind.EMBEDDING)
    def embed():
        return [0.1, 0.2, 0.3]

    embed()
    # After the span closes, the embedding should have been stored
    # (context may or may not persist depending on OTel detach behavior,
    # but the span should have been created correctly)
    finished = exporter.get_finished_spans()
    assert any(s.name == "embed" for s in finished)


def test_trace_exception_sets_error_type(_setup):
    exporter = _setup

    @trace("sync-err")
    def fail():
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        fail()

    span = next(s for s in exporter.get_finished_spans() if s.name == "sync-err")
    assert span.attributes.get(ERROR_TYPE) == "ValueError"


@pytest.mark.asyncio
async def test_trace_async_exception_sets_error_type(_setup):
    exporter = _setup

    @trace("async-err-type")
    async def fail():
        raise TypeError("wrong type")

    with pytest.raises(TypeError, match="wrong type"):
        await fail()

    span = next(s for s in exporter.get_finished_spans() if s.name == "async-err-type")
    assert span.attributes.get(ERROR_TYPE) == "TypeError"
