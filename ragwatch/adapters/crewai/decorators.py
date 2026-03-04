"""CrewAI-specific decorators.

- ``@node``     — wraps a CrewAI agent function with ``SpanKind.AGENT``.
- ``@endpoint`` — wraps a crew endpoint with ``SpanKind.CHAIN``.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from ragwatch.core.span_kinds import SpanKind
from ragwatch.instrumentation.decorators import trace

F = TypeVar("F", bound=Callable[..., Any])


def node(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    auto_track_io: bool = True,
) -> F | Callable[[F], F]:
    """Decorate a CrewAI agent function with ``SpanKind.AGENT``.

    Can be used with or without arguments::

        @node
        def researcher(task): ...

        @node("researcher-agent")
        def researcher(task): ...

    Args:
        func: The function to decorate (when used without parentheses).
        span_name: Explicit span name.
        auto_track_io: Whether to auto-capture input/output.
    """
    if callable(func):
        return trace(func, span_kind=SpanKind.AGENT, auto_track_io=auto_track_io)

    actual_span_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return trace(
            actual_span_name,
            span_kind=SpanKind.AGENT,
            auto_track_io=auto_track_io,
        )(fn)

    return decorator  # type: ignore[return-value]


def endpoint(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    auto_track_io: bool = True,
) -> F | Callable[[F], F]:
    """Decorate a CrewAI endpoint with ``SpanKind.CHAIN``.

    Can be used with or without arguments::

        @endpoint
        def run_crew(input): ...

        @endpoint("research-crew")
        def run_crew(input): ...

    Args:
        func: The function to decorate (when used without parentheses).
        span_name: Explicit span name.
        auto_track_io: Whether to auto-capture input/output.
    """
    if callable(func):
        return trace(func, span_kind=SpanKind.CHAIN, auto_track_io=auto_track_io)

    actual_span_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return trace(
            actual_span_name,
            span_kind=SpanKind.CHAIN,
            auto_track_io=auto_track_io,
        )(fn)

    return decorator  # type: ignore[return-value]
