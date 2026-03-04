"""Shared pytest fixtures for RAGWatch tests."""

from __future__ import annotations

import threading
from typing import Sequence

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from ragwatch.core.tracer import reset_tracer_provider


class InMemorySpanExporter(SpanExporter):
    """Minimal in-memory span exporter for testing."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        self.clear()


@pytest.fixture()
def span_exporter():
    """In-memory span exporter that collects finished spans."""
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(span_exporter):
    """A TracerProvider wired to the in-memory exporter.

    Automatically resets the global tracer provider after each test.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    otel_trace.set_tracer_provider(provider)

    yield provider

    provider.shutdown()
    reset_tracer_provider()


@pytest.fixture()
def spans(span_exporter):
    """Convenience accessor — returns the list of finished spans."""
    return span_exporter.get_finished_spans()
