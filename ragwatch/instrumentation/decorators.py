"""The ``@trace`` decorator — core instrumentation primitive for RAGWatch.

All telemetry extraction is driven by decorator parameters.
Zero observability code is required inside decorated functions.

  telemetry=["tool_calls", "routing"]   — auto-extract from result/state
  result_formatter=my_fn                 — convert raw result → string for caller
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, List, Optional, TypeVar, overload

from opentelemetry import trace as otel_trace

from ragwatch.core.span_kinds import SpanKind
from ragwatch.core.tracer import get_tracer
from ragwatch.instrumentation.context_model import InstrumentationContext
from ragwatch.instrumentation.embedding import store_embedding_in_context
from ragwatch.instrumentation.io_tracker import track_input, track_output
from ragwatch.instrumentation.result_transformers import transform_result
from ragwatch.instrumentation.attributes import safe_set_attribute
from ragwatch.instrumentation.semconv import ERROR_TYPE, OPENINFERENCE_SPAN_KIND
from ragwatch.instrumentation.span_hooks import (
    SpanHook,
    run_on_end,
    run_on_error,
    run_on_start,
)
from ragwatch.instrumentation.token_usage import extract_token_usage

_logger = logging.getLogger(__name__)


def _safe_normalize(ctx: InstrumentationContext) -> Any:
    """Call adapter.normalize_result() with failure isolation.

    Matches the same three-step pattern used by ``_safe_transform`` and
    ``_safe_extract_tokens``: log warning → span event → strict re-raise.
    """
    if ctx.adapter is None:
        return None
    try:
        from ragwatch.adapters.base import normalize_result

        return normalize_result(ctx.adapter, ctx.raw_result, ctx.state)
    except Exception as exc:
        _logger.warning("Result normalization failed: %s", exc)
        ctx.span.add_event("ragwatch.normalize_error", {"error": str(exc)})
        if _is_strict_mode():
            raise
        return None


def _is_strict_mode() -> bool:
    try:
        from ragwatch import get_active_config

        cfg = get_active_config()
        return cfg.strict_mode if cfg is not None else False
    except Exception:
        return False


def _effective_auto_track_io(per_decorator: bool) -> bool:
    """Combine per-decorator flag with global config."""
    if not per_decorator:
        return False
    try:
        from ragwatch import get_active_config

        cfg = get_active_config()
        if cfg is not None and not cfg.global_auto_track_io:
            return False
    except Exception:
        pass
    return True


F = TypeVar("F", bound=Callable[..., Any])


def _resolve_adapter(name: Optional[str]):
    """Look up a registered FrameworkAdapter by name, or return None."""
    if name is None:
        return None
    from ragwatch.adapters.base import get_adapter, register_adapter

    adapter = get_adapter(name)
    if adapter is not None:
        return adapter
    if name == "langgraph":
        from ragwatch.adapters.langgraph.adapter import LangGraphAdapter

        adapter = LangGraphAdapter()
    elif name == "crewai":
        from ragwatch.adapters.crewai.adapter import CrewAIAdapter

        adapter = CrewAIAdapter()
    else:
        return None
    register_adapter(adapter)
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


@overload
def trace(func: F) -> F: ...


@overload
def trace(
    span_name: str | None = None,
    *,
    span_kind: SpanKind = SpanKind.CHAIN,
    auto_track_io: bool = True,
    telemetry: Optional[List[str]] = None,
    result_formatter: Optional[Callable] = None,
    span_hooks: Optional[List[SpanHook]] = None,
    adapter: Optional[str] = None,
) -> Callable[[F], F]: ...


def trace(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    span_kind: SpanKind = SpanKind.CHAIN,
    auto_track_io: bool = True,
    telemetry: Optional[List[str]] = None,
    result_formatter: Optional[Callable] = None,
    span_hooks: Optional[List[SpanHook]] = None,
    adapter: Optional[str] = None,
) -> F | Callable[[F], F]:
    """Decorator that traces a function as an OTel span.

    Can be used with or without arguments::

        @trace
        def my_func(): ...

        @trace("my-span", span_kind=SpanKind.RETRIEVER,
               result_formatter=my_formatter)
        def my_func(): ...

    Args:
        func: The function to decorate (bare ``@trace`` usage).
        span_name: Explicit span name (defaults to ``__qualname__``).
        span_kind: OpenInference span kind.
        auto_track_io: Auto-capture ``input.value`` / ``output.value``.
        telemetry: Extraction strategies applied after the call returns.
            Valid items: ``"tool_calls"``, ``"routing"``,
            ``"agent_completion"``, ``"query_rewrite"``, ``"compression"``.
        result_formatter: Callable ``(raw_result) -> str`` for RETRIEVER /
            TOOL spans.  The decorated function may return raw data; the
            decorator records telemetry and converts to string for the caller.
        span_hooks: Per-decorator :class:`SpanHook` instances.  Their
            ``on_start`` / ``on_end`` methods run in addition to any global
            hooks registered via :func:`configure`.
        adapter: Name of a registered :class:`FrameworkAdapter` whose
            ``extract_state()`` drives state resolution for extractors.
    """
    if callable(func):
        return _make_wrapper(
            func,
            span_name=None,
            span_kind=span_kind,
            auto_track_io=auto_track_io,
            telemetry=telemetry,
            result_formatter=result_formatter,
            span_hooks=span_hooks,
            adapter=adapter,
        )

    actual_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return _make_wrapper(
            fn,
            span_name=actual_name,
            span_kind=span_kind,
            auto_track_io=auto_track_io,
            telemetry=telemetry,
            result_formatter=result_formatter,
            span_hooks=span_hooks,
            adapter=adapter,
        )

    return decorator  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper factory
# ─────────────────────────────────────────────────────────────────────────────


def _make_wrapper(
    func: F,
    *,
    span_name: str | None,
    span_kind: SpanKind,
    auto_track_io: bool,
    telemetry: Optional[List[str]],
    result_formatter: Optional[Callable],
    span_hooks: Optional[List[SpanHook]],
    adapter: Optional[str] = None,
) -> F:
    resolved_name = span_name or func.__qualname__
    _tel = list(telemetry) if telemetry else []
    _hooks = list(span_hooks) if span_hooks else None
    _adapter_name = adapter

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(resolved_name) as span:
                safe_set_attribute(span, OPENINFERENCE_SPAN_KIND, span_kind.value)
                resolved_adapter = _resolve_adapter(_adapter_name)
                state = _resolve_state(resolved_adapter, args, kwargs)
                ctx = InstrumentationContext(
                    span=span,
                    span_name=resolved_name,
                    span_kind=span_kind,
                    func_name=func.__qualname__,
                    args=args,
                    kwargs=kwargs,
                    adapter=resolved_adapter,
                    state=state,
                )
                run_on_start(span, args, kwargs, local_hooks=_hooks, context=ctx)
                _do_io = _effective_auto_track_io(auto_track_io)
                if _do_io:
                    track_input(span, args, kwargs)
                try:
                    raw = await func(*args, **kwargs)
                except Exception as exc:
                    ctx.exception = exc
                    span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    safe_set_attribute(span, ERROR_TYPE, type(exc).__name__)
                    run_on_error(span, exc, local_hooks=_hooks, context=ctx)
                    raise
                ctx.raw_result = raw
                ctx.normalized = _safe_normalize(ctx)
                # Telemetry on raw result (before any transformation)
                if _tel:
                    _extract_node_telemetry(ctx, _tel)
                # Result transformation (RETRIEVER / TOOL only)
                result = _safe_transform(ctx, raw, result_formatter)
                ctx.result = result
                if _do_io:
                    track_output(span, result)
                _handle_embedding_context(span_kind, result)
                _safe_extract_tokens(ctx, result)
                run_on_end(span, result, local_hooks=_hooks, context=ctx)
                return result

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer()
        with tracer.start_as_current_span(resolved_name) as span:
            safe_set_attribute(span, OPENINFERENCE_SPAN_KIND, span_kind.value)
            resolved_adapter = _resolve_adapter(_adapter_name)
            state = _resolve_state(resolved_adapter, args, kwargs)
            ctx = InstrumentationContext(
                span=span,
                span_name=resolved_name,
                span_kind=span_kind,
                func_name=func.__qualname__,
                args=args,
                kwargs=kwargs,
                adapter=resolved_adapter,
                state=state,
            )
            run_on_start(span, args, kwargs, local_hooks=_hooks, context=ctx)
            _do_io = _effective_auto_track_io(auto_track_io)
            if _do_io:
                track_input(span, args, kwargs)
            try:
                raw = func(*args, **kwargs)
            except Exception as exc:
                ctx.exception = exc
                span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                safe_set_attribute(span, ERROR_TYPE, type(exc).__name__)
                run_on_error(span, exc, local_hooks=_hooks, context=ctx)
                raise
            ctx.raw_result = raw
            ctx.normalized = _safe_normalize(ctx)
            # Telemetry on raw result (before any transformation)
            if _tel:
                _extract_node_telemetry(ctx, _tel)
            # Result transformation (RETRIEVER / TOOL only)
            result = _safe_transform(ctx, raw, result_formatter)
            ctx.result = result
            if _do_io:
                track_output(span, result)
            _handle_embedding_context(span_kind, result)
            _safe_extract_tokens(ctx, result)
            run_on_end(span, result, local_hooks=_hooks, context=ctx)
            return result

    return sync_wrapper  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Node-level telemetry extraction (delegated to ExtractorRegistry)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_state(adapter: Any, args: tuple, kwargs: dict) -> Any:
    """Resolve framework state once from the adapter or legacy convention."""
    if adapter is not None:
        return adapter.extract_state(args, kwargs)
    return next((a for a in args if isinstance(a, dict)), None)


def _safe_transform(
    ctx: InstrumentationContext,
    raw: Any,
    result_formatter: Any,
) -> Any:
    """Call transform_result with failure isolation."""
    try:
        return transform_result(
            ctx.span,
            ctx.span_kind,
            ctx.args,
            ctx.kwargs,
            raw,
            result_formatter,
            context=ctx,
        )
    except Exception as exc:
        _logger.warning("Result transformation failed: %s", exc)
        ctx.span.add_event("ragwatch.transform_error", {"error": str(exc)})
        if _is_strict_mode():
            raise
        return raw  # fall back to raw result


def _safe_extract_tokens(ctx: InstrumentationContext, result: Any) -> None:
    """Call extract_token_usage with failure isolation."""
    try:
        extract_token_usage(ctx.span, result, context=ctx)
    except Exception as exc:
        _logger.warning("Token usage extraction failed: %s", exc)
        ctx.span.add_event("ragwatch.token_error", {"error": str(exc)})
        if _is_strict_mode():
            raise


def _extract_node_telemetry(
    ctx: InstrumentationContext,
    telemetry: List[str],
) -> None:
    """Delegate telemetry extraction to the pluggable registry."""
    from ragwatch.instrumentation.extractors import get_default_registry

    get_default_registry().extract_all(
        telemetry,
        ctx.span,
        ctx.span_name,
        ctx.args,
        ctx.raw_result,
        kwargs=ctx.kwargs,
        adapter=ctx.adapter,
        context=ctx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Existing hooks (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def _handle_embedding_context(span_kind: SpanKind, result: Any) -> None:
    if span_kind is SpanKind.EMBEDDING:
        store_embedding_in_context(result)
