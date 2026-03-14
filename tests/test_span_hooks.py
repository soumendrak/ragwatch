"""Tests for ragwatch.instrumentation.span_hooks — SpanHook lifecycle."""

from __future__ import annotations

from typing import Any

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch import SpanKind, trace
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.span_hooks import (
    clear_global_hooks,
    get_global_hooks,
    register_global_hook,
)


@pytest.fixture(autouse=True)
def _setup():
    clear_global_hooks()
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    yield exporter
    reset_tracer_provider()
    clear_global_hooks()


# ---------------------------------------------------------------------------
# Local hooks (per-decorator)
# ---------------------------------------------------------------------------

class _RecordingHook:
    """Hook that records calls for assertion."""

    def __init__(self) -> None:
        self.starts: list[tuple] = []
        self.ends: list[tuple] = []

    def on_start(self, span, args, kwargs):
        self.starts.append((span.name, args, kwargs))

    def on_end(self, span, result):
        self.ends.append((span.name, result))


def test_local_hook_on_start_and_on_end(_setup):
    exporter = _setup
    hook = _RecordingHook()

    @trace("hooked-fn", span_hooks=[hook])
    def my_func(x):
        return x * 2

    result = my_func(5)

    assert result == 10
    assert len(hook.starts) == 1
    assert hook.starts[0][0] == "hooked-fn"
    assert len(hook.ends) == 1
    assert hook.ends[0][1] == 10  # result


def test_local_hook_receives_args_and_kwargs(_setup):
    hook = _RecordingHook()

    @trace("args-fn", span_hooks=[hook])
    def my_func(a, b, key="val"):
        return a + b

    my_func(1, 2, key="test")

    assert hook.starts[0][1] == (1, 2)
    assert hook.starts[0][2] == {"key": "test"}


def test_multiple_local_hooks(_setup):
    hook1 = _RecordingHook()
    hook2 = _RecordingHook()

    @trace("multi-hook", span_hooks=[hook1, hook2])
    def my_func():
        return "done"

    my_func()

    assert len(hook1.starts) == 1
    assert len(hook2.starts) == 1
    assert len(hook1.ends) == 1
    assert len(hook2.ends) == 1


# ---------------------------------------------------------------------------
# Local hook sets custom attributes
# ---------------------------------------------------------------------------

class _AttributeHook:
    """Adds custom attributes at start and end."""

    def on_start(self, span, args, kwargs):
        span.set_attribute("custom.hook.started", True)

    def on_end(self, span, result):
        span.set_attribute("custom.hook.result_type", type(result).__name__)


def test_local_hook_sets_span_attributes(_setup):
    exporter = _setup

    @trace("attr-fn", span_hooks=[_AttributeHook()])
    def my_func():
        return {"key": "value"}

    my_func()

    span = next(s for s in exporter.get_finished_spans() if s.name == "attr-fn")
    assert span.attributes["custom.hook.started"] is True
    assert span.attributes["custom.hook.result_type"] == "dict"


# ---------------------------------------------------------------------------
# Global hooks
# ---------------------------------------------------------------------------

def test_global_hook_runs_on_all_spans(_setup):
    exporter = _setup
    hook = _RecordingHook()
    register_global_hook(hook)

    @trace("fn-a")
    def fn_a():
        return "a"

    @trace("fn-b")
    def fn_b():
        return "b"

    fn_a()
    fn_b()

    assert len(hook.starts) == 2
    assert {s[0] for s in hook.starts} == {"fn-a", "fn-b"}
    assert len(hook.ends) == 2


def test_global_and_local_hooks_both_run(_setup):
    exporter = _setup
    global_hook = _RecordingHook()
    local_hook = _RecordingHook()
    register_global_hook(global_hook)

    @trace("combined", span_hooks=[local_hook])
    def my_func():
        return 42

    my_func()

    assert len(global_hook.starts) == 1
    assert len(local_hook.starts) == 1
    assert len(global_hook.ends) == 1
    assert len(local_hook.ends) == 1


def test_get_global_hooks():
    clear_global_hooks()
    assert get_global_hooks() == []

    hook = _RecordingHook()
    register_global_hook(hook)
    assert len(get_global_hooks()) == 1


def test_clear_global_hooks():
    register_global_hook(_RecordingHook())
    clear_global_hooks()
    assert get_global_hooks() == []


# ---------------------------------------------------------------------------
# Async support
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_hook_async(_setup):
    exporter = _setup
    hook = _RecordingHook()

    @trace("async-hooked", span_hooks=[hook])
    async def my_async():
        return "async-result"

    result = await my_async()

    assert result == "async-result"
    assert len(hook.starts) == 1
    assert hook.starts[0][0] == "async-hooked"
    assert len(hook.ends) == 1
    assert hook.ends[0][1] == "async-result"


# ---------------------------------------------------------------------------
# No hooks (backward compat)
# ---------------------------------------------------------------------------

def test_no_hooks_still_works(_setup):
    exporter = _setup

    @trace("plain")
    def plain():
        return "ok"

    assert plain() == "ok"
    span = next(s for s in exporter.get_finished_spans() if s.name == "plain")
    assert span is not None
