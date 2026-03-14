"""Pluggable span lifecycle hooks.

Provides a ``SpanHook`` protocol and a global hook registry so that users
can inject custom attribute enrichment at span start and end without
modifying SDK internals.

Usage (context-first — recommended)::

    from ragwatch.instrumentation.span_hooks import SpanHook

    class TimingHook:
        def on_start(self, span, args, kwargs, *, context=None):
            import time
            context.set_attribute("custom.start_ts", time.time())

        def on_end(self, span, result, *, context=None):
            pass

    # Per-decorator:
    @trace("my-span", span_hooks=[TimingHook()])
    def my_func(): ...

    # Global (all spans):
    ragwatch.configure(RAGWatchConfig(..., global_span_hooks=[TimingHook()]))

Note: Always prefer ``context.set_attribute()`` over raw
``span.set_attribute()`` to ensure policy enforcement (truncation,
redaction, naming validation).
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, List, Optional, Protocol, runtime_checkable

from opentelemetry import trace as otel_trace

_logger = logging.getLogger(__name__)


def _is_strict_mode() -> bool:
    """Check if the SDK is in strict mode (re-raise extension errors)."""
    try:
        from ragwatch import get_active_config
        cfg = get_active_config()
        return cfg.strict_mode if cfg is not None else False
    except Exception:
        return False


@runtime_checkable
class SpanHook(Protocol):
    """Contract for span lifecycle hooks.

    Implementors may define ``on_start`` and/or ``on_end`` to enrich spans
    with custom attributes, events, or side-effects.
    """

    def on_start(
        self,
        span: otel_trace.Span,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Called immediately after the span is created.

        Args:
            span: The newly started OTel span.
            args: Positional arguments passed to the decorated function.
            kwargs: Keyword arguments passed to the decorated function.
        """
        ...

    def on_end(
        self,
        span: otel_trace.Span,
        result: Any,
    ) -> None:
        """Called after the decorated function returns (before span closes).

        Args:
            span: The active OTel span.
            result: Return value of the decorated function (post-transformation).
        """
        ...


# ---------------------------------------------------------------------------
# Global hook registry
# ---------------------------------------------------------------------------

_GLOBAL_HOOKS: List[SpanHook] = []


def register_global_hook(hook: SpanHook) -> None:
    """Add a hook that runs on every traced span."""
    _GLOBAL_HOOKS.append(hook)


def get_global_hooks() -> List[SpanHook]:
    """Return the current list of global hooks."""
    return list(_GLOBAL_HOOKS)


def clear_global_hooks() -> None:
    """Remove all global hooks.  Intended for testing only."""
    _GLOBAL_HOOKS.clear()


def _accepts_context(method: Any) -> bool:
    """Return True if *method* accepts a ``context`` keyword argument."""
    try:
        sig = inspect.signature(method)
        return "context" in sig.parameters
    except (ValueError, TypeError):
        return False


def _run_hooks(
    hooks: List[Any],
    method_name: str,
    span: otel_trace.Span,
    positional_args: tuple,
    *,
    context: Any = None,
) -> None:
    """Run *method_name* on each hook with failure isolation."""
    strict = _is_strict_mode()
    for hook in hooks:
        method = getattr(hook, method_name, None)
        if method is None:
            continue
        try:
            if context is not None and _accepts_context(method):
                method(*positional_args, context=context)
            else:
                method(*positional_args)
        except Exception as exc:
            _logger.warning("SpanHook.%s failed: %s", method_name, exc)
            span.add_event("ragwatch.hook_error", {
                "hook": type(hook).__name__, "method": method_name,
                "error": str(exc),
            })
            if strict:
                raise


def run_on_start(
    span: otel_trace.Span,
    args: tuple,
    kwargs: dict,
    *,
    local_hooks: Optional[List[SpanHook]] = None,
    context: Any = None,
) -> None:
    """Execute ``on_start`` for all global + local hooks."""
    positional = (span, args, kwargs)
    _run_hooks(_GLOBAL_HOOKS, "on_start", span, positional, context=context)
    if local_hooks:
        _run_hooks(local_hooks, "on_start", span, positional, context=context)


def run_on_end(
    span: otel_trace.Span,
    result: Any,
    *,
    local_hooks: Optional[List[SpanHook]] = None,
    context: Any = None,
) -> None:
    """Execute ``on_end`` for all global + local hooks."""
    positional = (span, result)
    _run_hooks(_GLOBAL_HOOKS, "on_end", span, positional, context=context)
    if local_hooks:
        _run_hooks(local_hooks, "on_end", span, positional, context=context)


def run_on_error(
    span: otel_trace.Span,
    exception: BaseException,
    *,
    local_hooks: Optional[List[SpanHook]] = None,
    context: Any = None,
) -> None:
    """Execute ``on_error`` for hooks that implement it."""
    positional = (span, exception)
    _run_hooks(_GLOBAL_HOOKS, "on_error", span, positional, context=context)
    if local_hooks:
        _run_hooks(local_hooks, "on_error", span, positional, context=context)
