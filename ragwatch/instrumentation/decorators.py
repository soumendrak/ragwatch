"""The ``@trace`` decorator — core instrumentation primitive for RAGWatch.

Supports both ``sync`` and ``async`` functions.  Automatically sets
``openinference.span.kind`` and optionally tracks I/O.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, TypeVar, overload

from opentelemetry import trace as otel_trace

from ragwatch.core.span_kinds import SpanKind
from ragwatch.core.tracer import get_tracer
from ragwatch.instrumentation.embedding import store_embedding_in_context
from ragwatch.instrumentation.io_tracker import track_input, track_output
from ragwatch.instrumentation.semconv import (
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
    OPENINFERENCE_SPAN_KIND,
)

F = TypeVar("F", bound=Callable[..., Any])


@overload
def trace(func: F) -> F: ...


@overload
def trace(
    span_name: str | None = None,
    *,
    span_kind: SpanKind = SpanKind.CHAIN,
    auto_track_io: bool = True,
) -> Callable[[F], F]: ...


def trace(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    span_kind: SpanKind = SpanKind.CHAIN,
    auto_track_io: bool = True,
) -> F | Callable[[F], F]:
    """Decorator to trace a function as an OTel span.

    Can be used with or without arguments::

        @trace
        def my_func(): ...

        @trace("my-span", span_kind=SpanKind.RETRIEVER)
        def my_func(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        span_name: Explicit span name.  Defaults to the function's
            ``__qualname__``.
        span_kind: OpenInference span kind.
        auto_track_io: Whether to auto-capture input/output (default ON).

    Returns:
        A decorated function (or a decorator, when called with arguments).
    """
    # Called as @trace (no parentheses) — func is the decorated function
    if callable(func):
        return _make_wrapper(
            func, span_name=None, span_kind=span_kind, auto_track_io=auto_track_io
        )

    # Called as @trace("name", ...) — func is actually the span_name string
    actual_span_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return _make_wrapper(
            fn,
            span_name=actual_span_name,
            span_kind=span_kind,
            auto_track_io=auto_track_io,
        )

    return decorator  # type: ignore[return-value]


def _make_wrapper(
    func: F,
    *,
    span_name: str | None,
    span_kind: SpanKind,
    auto_track_io: bool,
) -> F:
    """Build the sync or async wrapper around *func*."""
    resolved_name = span_name or func.__qualname__

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(resolved_name) as span:
                span.set_attribute(OPENINFERENCE_SPAN_KIND, span_kind.value)
                if auto_track_io:
                    track_input(span, args, kwargs)
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise
                if auto_track_io:
                    track_output(span, result)
                _handle_embedding_context(span_kind, result)
                _extract_token_usage(span, result)
                return result

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer()
        with tracer.start_as_current_span(resolved_name) as span:
            span.set_attribute(OPENINFERENCE_SPAN_KIND, span_kind.value)
            if auto_track_io:
                track_input(span, args, kwargs)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise
            if auto_track_io:
                track_output(span, result)
            _handle_embedding_context(span_kind, result)
            _extract_token_usage(span, result)
            return result

    return sync_wrapper  # type: ignore[return-value]


def _handle_embedding_context(span_kind: SpanKind, result: Any) -> None:
    """If this is an EMBEDDING span, store the result in OTel context."""
    if span_kind is SpanKind.EMBEDDING:
        store_embedding_in_context(result)


def _extract_token_usage(span: Any, result: Any) -> None:
    """Scan *result* for LangChain AIMessage.usage_metadata and set token
    count attributes on *span*.

    LangChain nodes return dicts like ``{"messages": [AIMessage(...)]}``.
    ``AIMessage.usage_metadata`` is a plain dict::

        {"input_tokens": 379, "output_tokens": 45, "total_tokens": 424}

    Duck-typed so RAGWatch stays framework-agnostic (no LangChain import).
    """
    if not span.is_recording():
        return

    candidates: list = []
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                candidates.extend(v)
            else:
                candidates.append(v)
    elif isinstance(result, list):
        candidates = result
    else:
        candidates = [result]

    prompt_tokens = completion_tokens = total_tokens = 0
    found = False
    for obj in candidates:
        usage = getattr(obj, "usage_metadata", None)
        if isinstance(usage, dict):
            found = True
            prompt_tokens     += usage.get("input_tokens", 0)
            completion_tokens += usage.get("output_tokens", 0)
            total_tokens      += usage.get("total_tokens", 0)

    if found:
        span.set_attribute(LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
        span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
        span.set_attribute(LLM_TOKEN_COUNT_TOTAL, total_tokens)
