"""Tests for failure isolation — broken hooks/extractors/transformers must not crash user code."""

from __future__ import annotations

from typing import Any, List, Optional

import pytest
from tests.conftest import InMemorySpanExporter

import ragwatch
from ragwatch import SpanKind, trace
from ragwatch.adapters.base import clear_adapters, register_adapter
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.extractors import (
    get_default_registry,
    reset_default_registry,
)
from ragwatch.instrumentation.span_hooks import clear_global_hooks, register_global_hook


@pytest.fixture(autouse=True)
def _setup():
    clear_global_hooks()
    clear_adapters()
    reset_default_registry()
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(strict_mode=False)
    yield exporter
    ragwatch._ACTIVE_CONFIG = None
    reset_tracer_provider()
    clear_global_hooks()
    clear_adapters()
    reset_default_registry()


# ---------------------------------------------------------------------------
# Broken hook does not crash user code
# ---------------------------------------------------------------------------

class _BrokenStartHook:
    def on_start(self, span, args, kwargs):
        raise RuntimeError("hook start exploded")

    def on_end(self, span, result):
        pass


class _BrokenEndHook:
    def on_start(self, span, args, kwargs):
        pass

    def on_end(self, span, result):
        raise RuntimeError("hook end exploded")


def test_broken_on_start_hook_does_not_crash():
    """A broken on_start hook should not prevent the function from running."""
    @trace("test-fn", span_hooks=[_BrokenStartHook()])
    def my_fn():
        return 42

    result = my_fn()
    assert result == 42


def test_broken_on_end_hook_does_not_crash():
    """A broken on_end hook should not prevent the function from returning."""
    @trace("test-fn", span_hooks=[_BrokenEndHook()])
    def my_fn():
        return 42

    result = my_fn()
    assert result == 42


def test_broken_global_hook_does_not_crash():
    """A broken global hook should not prevent function execution."""
    register_global_hook(_BrokenStartHook())

    @trace("test-fn")
    def my_fn():
        return 99

    result = my_fn()
    assert result == 99


def test_broken_hook_records_error_event(_setup):
    """Broken hook should record an error event on the span."""
    exporter = _setup

    @trace("test-fn", span_hooks=[_BrokenStartHook()])
    def my_fn():
        return 1

    my_fn()

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    events = spans[-1].events
    hook_errors = [e for e in events if e.name == "ragwatch.hook_error"]
    assert len(hook_errors) >= 1
    assert "hook start exploded" in hook_errors[0].attributes["error"]


# ---------------------------------------------------------------------------
# Broken extractor does not crash user code
# ---------------------------------------------------------------------------

class _BrokenExtractor:
    name = "broken_ext"

    def extract(self, span, span_name, args, result, state):
        raise RuntimeError("extractor exploded")


def test_broken_extractor_does_not_crash():
    """A broken extractor should not prevent the function from returning."""
    get_default_registry().register(_BrokenExtractor())

    @trace("test-fn", telemetry=["broken_ext"])
    def my_fn():
        return {"data": True}

    result = my_fn()
    assert result == {"data": True}


def test_broken_extractor_records_error_event(_setup):
    """Broken extractor should record an error event on the span."""
    exporter = _setup
    get_default_registry().register(_BrokenExtractor())

    @trace("test-fn", telemetry=["broken_ext"])
    def my_fn():
        return {"data": True}

    my_fn()

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    events = spans[-1].events
    ext_errors = [e for e in events if e.name == "ragwatch.extractor_error"]
    assert len(ext_errors) >= 1
    assert "extractor exploded" in ext_errors[0].attributes["error"]


# ---------------------------------------------------------------------------
# Broken result transformer does not crash user code
# ---------------------------------------------------------------------------

def _bad_formatter(result):
    raise RuntimeError("formatter exploded")


def test_broken_result_formatter_does_not_crash():
    """A broken result_formatter should fall back to raw result."""
    raw = {"value": 42}  # non-string so formatter gets invoked

    @trace("test-fn", span_kind=SpanKind.TOOL, result_formatter=_bad_formatter)
    def my_fn():
        return raw

    result = my_fn()
    assert result == raw


def test_broken_result_formatter_records_error_event(_setup):
    """Broken formatter should record a transform error event."""
    exporter = _setup

    @trace("test-fn", span_kind=SpanKind.TOOL, result_formatter=_bad_formatter)
    def my_fn():
        return {"value": 42}  # non-string so formatter gets invoked

    my_fn()

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    events = spans[-1].events
    transform_errors = [e for e in events if e.name == "ragwatch.transform_error"]
    assert len(transform_errors) >= 1
    assert "formatter exploded" in transform_errors[0].attributes["error"]


# ---------------------------------------------------------------------------
# strict_mode re-raises extension exceptions
# ---------------------------------------------------------------------------

def test_strict_mode_reraises_hook_error():
    """In strict_mode, broken hooks re-raise instead of swallowing."""
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(strict_mode=True)

    @trace("test-fn", span_hooks=[_BrokenStartHook()])
    def my_fn():
        return 42

    with pytest.raises(RuntimeError, match="hook start exploded"):
        my_fn()


def test_strict_mode_reraises_extractor_error():
    """In strict_mode, broken extractors re-raise."""
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(strict_mode=True)
    get_default_registry().register(_BrokenExtractor())

    @trace("test-fn", telemetry=["broken_ext"])
    def my_fn():
        return {"data": True}

    with pytest.raises(RuntimeError, match="extractor exploded"):
        my_fn()


def test_strict_mode_reraises_formatter_error():
    """In strict_mode, broken result_formatter re-raises."""
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(strict_mode=True)

    @trace("test-fn", span_kind=SpanKind.TOOL, result_formatter=_bad_formatter)
    def my_fn():
        return {"value": 42}  # non-string so formatter gets invoked

    with pytest.raises(RuntimeError, match="formatter exploded"):
        my_fn()


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broken_hook_does_not_crash_async():
    """Broken hook in async context should not crash."""
    @trace("test-fn", span_hooks=[_BrokenStartHook()])
    async def my_fn():
        return 42

    result = await my_fn()
    assert result == 42


@pytest.mark.asyncio
async def test_broken_extractor_does_not_crash_async():
    """Broken extractor in async context should not crash."""
    get_default_registry().register(_BrokenExtractor())

    @trace("test-fn", telemetry=["broken_ext"])
    async def my_fn():
        return {"data": True}

    result = await my_fn()
    assert result == {"data": True}
