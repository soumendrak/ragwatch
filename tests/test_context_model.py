"""Tests for InstrumentationContext + on_error hook pathway."""

from __future__ import annotations

from typing import Any

import pytest
from tests.conftest import InMemorySpanExporter

import ragwatch
from ragwatch import SpanKind, trace
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.context_model import InstrumentationContext
from ragwatch.instrumentation.span_hooks import clear_global_hooks, register_global_hook


@pytest.fixture(autouse=True)
def _setup():
    clear_global_hooks()
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(strict_mode=False)
    yield exporter
    ragwatch._ACTIVE_CONFIG = None
    reset_tracer_provider()
    clear_global_hooks()


# ---------------------------------------------------------------------------
# InstrumentationContext dataclass basics
# ---------------------------------------------------------------------------

def test_context_default_fields():
    """InstrumentationContext fields have correct defaults."""
    from unittest.mock import MagicMock
    span = MagicMock()
    ctx = InstrumentationContext(
        span=span, span_name="test", span_kind=SpanKind.CHAIN,
        func_name="my_func",
    )
    assert ctx.args == ()
    assert ctx.kwargs == {}
    assert ctx.adapter is None
    assert ctx.state is None
    assert ctx.raw_result is None
    assert ctx.result is None
    assert ctx.exception is None


def test_context_mutable_fields():
    """InstrumentationContext fields can be updated incrementally."""
    from unittest.mock import MagicMock
    span = MagicMock()
    ctx = InstrumentationContext(
        span=span, span_name="test", span_kind=SpanKind.CHAIN,
        func_name="my_func",
    )
    ctx.raw_result = {"data": 1}
    ctx.result = "transformed"
    ctx.exception = ValueError("oops")
    assert ctx.raw_result == {"data": 1}
    assert ctx.result == "transformed"
    assert isinstance(ctx.exception, ValueError)


# ---------------------------------------------------------------------------
# Hook receives context when it accepts it
# ---------------------------------------------------------------------------

class _ContextCapturingHook:
    """Hook that captures the InstrumentationContext."""
    start_ctx = None
    end_ctx = None

    def on_start(self, span, args, kwargs, *, context=None):
        _ContextCapturingHook.start_ctx = context

    def on_end(self, span, result, *, context=None):
        _ContextCapturingHook.end_ctx = context


def test_hook_receives_context():
    """Hooks that accept a context= kwarg get the InstrumentationContext."""
    _ContextCapturingHook.start_ctx = None
    _ContextCapturingHook.end_ctx = None

    @trace("ctx-test", span_hooks=[_ContextCapturingHook()])
    def my_fn(x):
        return x + 1

    result = my_fn(10)
    assert result == 11

    ctx = _ContextCapturingHook.start_ctx
    assert ctx is not None
    assert isinstance(ctx, InstrumentationContext)
    assert ctx.span_name == "ctx-test"
    assert ctx.span_kind == SpanKind.CHAIN
    assert ctx.args == (10,)

    end_ctx = _ContextCapturingHook.end_ctx
    assert end_ctx is not None
    assert end_ctx.result == 11
    assert end_ctx.raw_result == 11


def test_hook_without_context_still_works():
    """Old-style hooks without context= parameter still work."""
    calls = []

    class _OldStyleHook:
        def on_start(self, span, args, kwargs):
            calls.append("start")

        def on_end(self, span, result):
            calls.append("end")

    @trace("old-hook-test", span_hooks=[_OldStyleHook()])
    def my_fn():
        return 42

    result = my_fn()
    assert result == 42
    assert calls == ["start", "end"]


# ---------------------------------------------------------------------------
# on_error hook pathway
# ---------------------------------------------------------------------------

class _ErrorCapturingHook:
    """Hook that captures on_error calls."""
    error_ctx = None
    error_exc = None

    def on_start(self, span, args, kwargs):
        pass

    def on_end(self, span, result):
        pass

    def on_error(self, span, exception, *, context=None):
        _ErrorCapturingHook.error_exc = exception
        _ErrorCapturingHook.error_ctx = context


def test_on_error_called_on_exception():
    """on_error is called when the decorated function raises."""
    _ErrorCapturingHook.error_exc = None
    _ErrorCapturingHook.error_ctx = None

    @trace("error-test", span_hooks=[_ErrorCapturingHook()])
    def my_fn():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        my_fn()

    assert _ErrorCapturingHook.error_exc is not None
    assert str(_ErrorCapturingHook.error_exc) == "boom"
    ctx = _ErrorCapturingHook.error_ctx
    assert ctx is not None
    assert isinstance(ctx, InstrumentationContext)
    assert ctx.exception is not None
    assert str(ctx.exception) == "boom"


def test_on_error_not_called_on_success():
    """on_error is NOT called when the function succeeds."""
    _ErrorCapturingHook.error_exc = None
    _ErrorCapturingHook.error_ctx = None

    @trace("no-error-test", span_hooks=[_ErrorCapturingHook()])
    def my_fn():
        return 42

    my_fn()

    assert _ErrorCapturingHook.error_exc is None
    assert _ErrorCapturingHook.error_ctx is None


def test_on_error_global_hook():
    """Global hooks with on_error are called on exception."""
    _ErrorCapturingHook.error_exc = None
    _ErrorCapturingHook.error_ctx = None
    register_global_hook(_ErrorCapturingHook())

    @trace("global-error-test")
    def my_fn():
        raise RuntimeError("global boom")

    with pytest.raises(RuntimeError, match="global boom"):
        my_fn()

    assert _ErrorCapturingHook.error_exc is not None
    assert str(_ErrorCapturingHook.error_exc) == "global boom"


def test_hooks_without_on_error_are_skipped():
    """Hooks that don't define on_error are silently skipped."""
    class _NoErrorHook:
        def on_start(self, span, args, kwargs):
            pass

        def on_end(self, span, result):
            pass

    @trace("skip-on-error-test", span_hooks=[_NoErrorHook()])
    def my_fn():
        raise RuntimeError("skip")

    with pytest.raises(RuntimeError, match="skip"):
        my_fn()


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hook_receives_context_async():
    """Async functions also pass InstrumentationContext to hooks."""
    _ContextCapturingHook.start_ctx = None
    _ContextCapturingHook.end_ctx = None

    @trace("async-ctx-test", span_hooks=[_ContextCapturingHook()])
    async def my_fn(x):
        return x * 2

    result = await my_fn(5)
    assert result == 10

    ctx = _ContextCapturingHook.start_ctx
    assert ctx is not None
    assert ctx.span_name == "async-ctx-test"
    assert ctx.args == (5,)

    end_ctx = _ContextCapturingHook.end_ctx
    assert end_ctx is not None
    assert end_ctx.result == 10


@pytest.mark.asyncio
async def test_on_error_called_async():
    """on_error is called for async functions that raise."""
    _ErrorCapturingHook.error_exc = None
    _ErrorCapturingHook.error_ctx = None

    @trace("async-error-test", span_hooks=[_ErrorCapturingHook()])
    async def my_fn():
        raise ValueError("async boom")

    with pytest.raises(ValueError, match="async boom"):
        await my_fn()

    assert _ErrorCapturingHook.error_exc is not None
    assert str(_ErrorCapturingHook.error_exc) == "async boom"


# ---------------------------------------------------------------------------
# ctx.state is populated from first dict arg (legacy convention)
# ---------------------------------------------------------------------------

def test_context_state_populated_from_dict_arg():
    """ctx.state should be auto-populated from first dict positional arg."""
    _ContextCapturingHook.start_ctx = None

    @trace("state-test", span_hooks=[_ContextCapturingHook()])
    def my_fn(state):
        return state

    my_fn({"question": "what?"})

    ctx = _ContextCapturingHook.start_ctx
    assert ctx is not None
    assert ctx.state == {"question": "what?"}


def test_context_state_none_when_no_dict_arg():
    """ctx.state should be None when no dict arg is passed."""
    _ContextCapturingHook.start_ctx = None

    @trace("no-state-test", span_hooks=[_ContextCapturingHook()])
    def my_fn(x):
        return x

    my_fn(42)

    ctx = _ContextCapturingHook.start_ctx
    assert ctx is not None
    assert ctx.state is None


def test_context_state_populated_via_adapter():
    """ctx.state should use adapter.extract_state() when adapter is set."""
    from ragwatch.adapters.base import register_adapter, clear_adapters

    class _KwargsAdapter:
        name = "test_kwargs"
        def extract_state(self, args, kwargs):
            return kwargs.get("my_state")
        def default_extractors(self):
            return []

    clear_adapters()
    register_adapter(_KwargsAdapter())
    _ContextCapturingHook.start_ctx = None

    @trace("adapter-state-test", span_hooks=[_ContextCapturingHook()], adapter="test_kwargs")
    def my_fn(x, my_state=None):
        return x

    my_fn("hello", my_state={"from_adapter": True})

    ctx = _ContextCapturingHook.start_ctx
    assert ctx is not None
    assert ctx.state == {"from_adapter": True}
    clear_adapters()


# ---------------------------------------------------------------------------
# ctx.set_attribute() — policy-enforced writer
# ---------------------------------------------------------------------------

def test_context_set_attribute(_setup):
    """ctx.set_attribute() should delegate to safe_set_attribute."""
    exporter = _setup

    class _SetAttrHook:
        def on_start(self, span, args, kwargs, *, context=None):
            context.set_attribute("custom.from_hook", "hello")

        def on_end(self, span, result, *, context=None):
            context.set_attribute("custom.result_val", result)

    @trace("set-attr-test", span_hooks=[_SetAttrHook()])
    def my_fn():
        return 42

    my_fn()

    span = next(s for s in exporter.get_finished_spans() if s.name == "set-attr-test")
    assert span.attributes["custom.from_hook"] == "hello"
    assert span.attributes["custom.result_val"] == 42


def test_context_set_attribute_respects_policy(_setup):
    """ctx.set_attribute() should enforce the active AttributePolicy."""
    from ragwatch.instrumentation.attribute_policy import AttributePolicy

    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(
        attribute_policy=AttributePolicy(redact_keys=["secret"]),
    )
    exporter = _setup

    class _SecretHook:
        def on_start(self, span, args, kwargs, *, context=None):
            context.set_attribute("api.secret", "my_token_123")
            context.set_attribute("api.name", "visible")

        def on_end(self, span, result):
            pass

    @trace("policy-test", span_hooks=[_SecretHook()])
    def my_fn():
        return 1

    my_fn()

    span = next(s for s in exporter.get_finished_spans() if s.name == "policy-test")
    assert span.attributes["api.secret"] == "[REDACTED]"
    assert span.attributes["api.name"] == "visible"
