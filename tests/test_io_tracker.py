"""Tests for ragwatch.instrumentation.io_tracker — auto I/O tracking."""

from __future__ import annotations

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.io_tracker import (
    _MAX_IO_BYTES,
    _safe_serialize,
)
from ragwatch.instrumentation.semconv import INPUT_VALUE, OUTPUT_VALUE


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_tracer_provider()


def test_io_tracker_captures_args():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    from ragwatch import trace

    @trace("io-test")
    def greet(name, greeting="hello"):
        return f"{greeting} {name}"

    greet("world", greeting="hi")

    span = next(s for s in exporter.get_finished_spans() if s.name == "io-test")
    assert INPUT_VALUE in span.attributes
    assert "world" in span.attributes[INPUT_VALUE]
    assert "hi" in span.attributes[INPUT_VALUE]


def test_io_tracker_captures_return():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    from ragwatch import trace

    @trace("io-test")
    def add(a, b):
        return a + b

    add(3, 4)

    span = next(s for s in exporter.get_finished_spans() if s.name == "io-test")
    assert OUTPUT_VALUE in span.attributes
    assert "7" in span.attributes[OUTPUT_VALUE]


def test_io_tracker_truncates_over_4kb():
    big_value = "x" * 10000
    serialized = _safe_serialize(big_value)
    assert len(serialized) <= _MAX_IO_BYTES + len("...[truncated]") + 2


def test_io_tracker_disabled():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    from ragwatch import trace

    @trace("no-io", auto_track_io=False)
    def add(a, b):
        return a + b

    add(1, 2)

    span = next(s for s in exporter.get_finished_spans() if s.name == "no-io")
    assert INPUT_VALUE not in span.attributes
    assert OUTPUT_VALUE not in span.attributes


def test_io_tracker_handles_complex_types():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )

    from ragwatch import trace

    @trace("complex-io")
    def process(data):
        return {"result": data["items"], "count": len(data["items"])}

    process({"items": [1, 2, 3]})

    span = next(s for s in exporter.get_finished_spans() if s.name == "complex-io")
    assert INPUT_VALUE in span.attributes
    assert OUTPUT_VALUE in span.attributes
    assert "items" in span.attributes[INPUT_VALUE]
    assert "count" in span.attributes[OUTPUT_VALUE]


def test_safe_serialize_fallback():
    class Unserializable:
        def __repr__(self):
            return "<Unserializable>"

    result = _safe_serialize(Unserializable())
    assert "Unserializable" in result
