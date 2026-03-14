"""LangGraph-specific decorators.

- ``@node``     — wraps a LangGraph node function with ``SpanKind.AGENT``.
- ``@workflow`` — wraps a workflow orchestrator with ``SpanKind.CHAIN``.
- ``@tool``     — wraps a tool implementation with ``SpanKind.TOOL``.

All decorators accept the same ``telemetry`` and ``result_formatter``
parameters as the core ``@trace`` — telemetry extraction is entirely
driven by decorator parameters; no code inside the decorated function
is required.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, TypeVar

from ragwatch.core.span_kinds import SpanKind
from ragwatch.instrumentation.decorators import trace
from ragwatch.instrumentation.span_hooks import SpanHook

F = TypeVar("F", bound=Callable[..., Any])


def node(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    auto_track_io: bool = True,
    telemetry: Optional[List[str]] = None,
    result_formatter: Optional[Callable] = None,
    span_hooks: Optional[List[SpanHook]] = None,
) -> F | Callable[[F], F]:
    """Decorate a LangGraph node with ``SpanKind.AGENT``.

    Usage::

        @node("orchestrator", telemetry=["tool_calls", "routing"])
        def orchestrator(state, llm_with_tools): ...

        @node("collect-answer", telemetry=["agent_completion"])
        def collect_answer(state): ...

    Args:
        telemetry: Extraction strategies — any of: ``"tool_calls"``,
            ``"routing"``, ``"agent_completion"``, ``"query_rewrite"``,
            ``"compression"``.  The decorator inspects the LangGraph state
            dict (first positional arg) and the return value to emit rich
            telemetry without any code inside the function body.
        result_formatter: Not typically needed for nodes; included for
            completeness.
    """
    if callable(func):
        return trace(func, span_kind=SpanKind.AGENT, auto_track_io=auto_track_io,
                     telemetry=telemetry, result_formatter=result_formatter,
                     span_hooks=span_hooks, adapter="langgraph")

    actual_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return trace(actual_name, span_kind=SpanKind.AGENT,
                     auto_track_io=auto_track_io, telemetry=telemetry,
                     result_formatter=result_formatter,
                     span_hooks=span_hooks, adapter="langgraph")(fn)

    return decorator  # type: ignore[return-value]


def workflow(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    auto_track_io: bool = True,
    telemetry: Optional[List[str]] = None,
    result_formatter: Optional[Callable] = None,
    span_hooks: Optional[List[SpanHook]] = None,
) -> F | Callable[[F], F]:
    """Decorate a LangGraph workflow with ``SpanKind.CHAIN``."""
    if callable(func):
        return trace(func, span_kind=SpanKind.CHAIN, auto_track_io=auto_track_io,
                     telemetry=telemetry, result_formatter=result_formatter,
                     span_hooks=span_hooks, adapter="langgraph")

    actual_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return trace(actual_name, span_kind=SpanKind.CHAIN,
                     auto_track_io=auto_track_io, telemetry=telemetry,
                     result_formatter=result_formatter,
                     span_hooks=span_hooks, adapter="langgraph")(fn)

    return decorator  # type: ignore[return-value]


def tool(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    auto_track_io: bool = True,
    telemetry: Optional[List[str]] = None,
    result_formatter: Optional[Callable] = None,
    span_hooks: Optional[List[SpanHook]] = None,
) -> F | Callable[[F], F]:
    """Decorate a LangGraph tool implementation with ``SpanKind.TOOL``.

    Usage::

        @tool("retrieve-parent-chunks",
              result_formatter=_format_parent_chunk)
        def _retrieve_parent_chunks(self, parent_id: str):
            ...
            return raw_parent_dict   # decorator records attrs + formats

    """
    if callable(func):
        return trace(func, span_kind=SpanKind.TOOL, auto_track_io=auto_track_io,
                     telemetry=telemetry, result_formatter=result_formatter,
                     span_hooks=span_hooks, adapter="langgraph")

    actual_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return trace(actual_name, span_kind=SpanKind.TOOL,
                     auto_track_io=auto_track_io, telemetry=telemetry,
                     result_formatter=result_formatter,
                     span_hooks=span_hooks, adapter="langgraph")(fn)

    return decorator  # type: ignore[return-value]
