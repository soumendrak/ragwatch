"""Tests for ragwatch.core.tracer — TracerProvider setup and singleton."""

from __future__ import annotations

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from tests.conftest import InMemorySpanExporter

from ragwatch import RAGWatchConfig, RAGWatchRuntime, configure
from ragwatch.core.tracer import (
    configure_tracer,
    get_tracer,
    get_tracer_provider,
    reset_tracer_provider,
)


@pytest.fixture(autouse=True)
def _reset():
    """Ensure a clean tracer state for every test."""
    yield
    reset_tracer_provider()


def test_configure_creates_tracer_provider():
    config = RAGWatchConfig(service_name="test-svc")
    provider = configure_tracer(config, _force_flush=True)
    assert isinstance(provider, TracerProvider)


def test_configure_with_custom_exporter():
    exporter = InMemorySpanExporter()
    config = RAGWatchConfig(service_name="test-svc", exporter=exporter)
    configure_tracer(config, _force_flush=True)

    tracer = get_tracer()
    with tracer.start_as_current_span("test-span"):
        pass

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "test-span"


def test_configure_service_name_set():
    exporter = InMemorySpanExporter()
    config = RAGWatchConfig(service_name="my-rag-app", exporter=exporter)
    provider = configure_tracer(config, _force_flush=True)

    resource_attrs = dict(provider.resource.attributes)
    assert resource_attrs["service.name"] == "my-rag-app"


def test_trace_sync_function():
    from ragwatch import trace

    exporter = InMemorySpanExporter()
    config = RAGWatchConfig(service_name="test", exporter=exporter)
    configure_tracer(config, _force_flush=True)

    @trace("sync-fn")
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

    finished = exporter.get_finished_spans()
    assert any(s.name == "sync-fn" for s in finished)


@pytest.mark.asyncio
async def test_trace_async_function():
    from ragwatch import trace

    exporter = InMemorySpanExporter()
    config = RAGWatchConfig(service_name="test", exporter=exporter)
    configure_tracer(config, _force_flush=True)

    @trace("async-fn")
    async def add(a, b):
        return a + b

    result = await add(1, 2)
    assert result == 3

    finished = exporter.get_finished_spans()
    assert any(s.name == "async-fn" for s in finished)


def test_trace_creates_span_with_name():
    from ragwatch import trace

    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    @trace("my-custom-span")
    def noop():
        pass

    noop()

    names = [s.name for s in exporter.get_finished_spans()]
    assert "my-custom-span" in names


def test_trace_span_has_correct_parent():
    from ragwatch import trace

    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    @trace("parent")
    def parent():
        child()

    @trace("child")
    def child():
        pass

    parent()

    finished = exporter.get_finished_spans()
    parent_span = next(s for s in finished if s.name == "parent")
    child_span = next(s for s in finished if s.name == "child")

    assert child_span.context.trace_id == parent_span.context.trace_id
    assert child_span.parent.span_id == parent_span.context.span_id


def test_trace_span_attributes_set():
    from ragwatch import trace, SpanKind
    from ragwatch.instrumentation.semconv import OPENINFERENCE_SPAN_KIND

    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    @trace("attr-span", span_kind=SpanKind.RETRIEVER)
    def noop():
        pass

    noop()

    span = next(s for s in exporter.get_finished_spans() if s.name == "attr-span")
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == "RETRIEVER"


def test_trace_exception_recorded():
    from ragwatch import trace

    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    @trace("error-span")
    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        fail()

    span = next(s for s in exporter.get_finished_spans() if s.name == "error-span")
    assert span.status.status_code == otel_trace.StatusCode.ERROR
    assert len(span.events) > 0


def test_get_tracer_provider_returns_configured():
    config = RAGWatchConfig(service_name="t")
    configure_tracer(config, _force_flush=True)
    assert get_tracer_provider() is not None


def test_public_configure():
    exporter = InMemorySpanExporter()
    runtime = configure(RAGWatchConfig(service_name="pub", exporter=exporter))
    assert get_tracer_provider() is not None
    assert isinstance(runtime, RAGWatchRuntime)


# ---------------------------------------------------------------------------
# Sampler support (P1)
# ---------------------------------------------------------------------------


def test_default_sampler_records_all_spans():
    """With no sampler configured (ALWAYS_ON default), every span is exported."""
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    tracer = get_tracer()
    with tracer.start_as_current_span("span-a"):
        pass
    with tracer.start_as_current_span("span-b"):
        pass

    names = [s.name for s in exporter.get_finished_spans()]
    assert "span-a" in names
    assert "span-b" in names


def test_always_off_sampler_drops_all_spans():
    """ALWAYS_OFF sampler should result in zero exported spans."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter, sampler=ALWAYS_OFF),
        _force_flush=True,
    )

    tracer = get_tracer()
    with tracer.start_as_current_span("dropped"):
        pass

    assert exporter.get_finished_spans() == []


def test_trace_id_ratio_sampler_wired():
    """TraceIdRatioBased sampler is accepted and wired into the TracerProvider."""
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    sampler = TraceIdRatioBased(0.5)
    exporter = InMemorySpanExporter()
    provider = configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter, sampler=sampler),
        _force_flush=True,
    )

    assert provider.sampler is sampler


def test_sampler_none_uses_sdk_default():
    """When sampler=None, TracerProvider uses its built-in ParentBased(ALWAYS_ON) default."""
    from opentelemetry.sdk.trace.sampling import ParentBased

    exporter = InMemorySpanExporter()
    provider = configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter, sampler=None),
        _force_flush=True,
    )

    assert isinstance(provider.sampler, ParentBased)
