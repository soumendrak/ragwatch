"""LlamaIndex-specific decorators.

- ``@query_engine`` — wraps a query function with ``SpanKind.CHAIN``.
- ``@retriever``    — wraps a retrieval function with ``SpanKind.RETRIEVER``.
- ``@chat_engine``  — wraps a chat function with ``SpanKind.AGENT``.
- ``@ingestion``    — wraps an ingestion/embedding run with ``SpanKind.EMBEDDING``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, TypeVar

from ragwatch.core.span_kinds import SpanKind
from ragwatch.instrumentation.attributes import safe_set_attribute
from ragwatch.instrumentation.decorators import _effective_auto_track_io, trace
from ragwatch.instrumentation.io_tracker import (
    track_input,
)
from ragwatch.instrumentation.semconv import OUTPUT_VALUE
from ragwatch.instrumentation.span_hooks import SpanHook

F = TypeVar("F", bound=Callable[..., Any])


class _StreamingSafeIO:
    """Auto I/O tracking that never reads a streaming response.

    The generic output tracker serializes results with ``str()``, which
    drains LlamaIndex's ``StreamingResponse`` generator before the caller
    can iterate it.  Streaming results get no ``output.value``.
    """

    def on_start(self, span: Any, args: tuple, kwargs: dict, *, context=None) -> None:
        if _effective_auto_track_io(True):
            track_input(span, args, kwargs)

    def on_end(self, span: Any, result: Any, *, context=None) -> None:
        from ragwatch.adapters.llamaindex.hooks import _io_text, _is_streaming

        if not _effective_auto_track_io(True) or _is_streaming(result):
            return
        # Scrub, serialize and cap at 4 KB (strings included).
        safe_set_attribute(span, OUTPUT_VALUE, _io_text(result))


def _make(kind: SpanKind, doc: str) -> Callable[..., Any]:
    def decorator_factory(
        func: F | str | None = None,
        *,
        span_name: str | None = None,
        auto_track_io: bool = True,
        span_hooks: Optional[List[SpanHook]] = None,
    ) -> F | Callable[[F], F]:
        hooks = list(span_hooks or [])
        if auto_track_io:
            hooks.append(_StreamingSafeIO())
        options = dict(
            span_kind=kind,
            auto_track_io=False,  # handled by _StreamingSafeIO
            span_hooks=hooks,
            adapter="llamaindex",
        )
        if callable(func):
            return trace(func, **options)

        actual_span_name = func if isinstance(func, str) else span_name

        def decorator(fn: F) -> F:
            return trace(actual_span_name, **options)(fn)

        return decorator  # type: ignore[return-value]

    decorator_factory.__doc__ = doc
    return decorator_factory


query_engine = _make(
    SpanKind.CHAIN,
    """Decorate a LlamaIndex query function (``QueryEngine.query``) as CHAIN.

    Usable as ``@query_engine`` or ``@query_engine("name")``.
    """,
)
retriever = _make(
    SpanKind.RETRIEVER,
    """Decorate a LlamaIndex retrieval function (``BaseRetriever.retrieve``).

    Usable as ``@retriever`` or ``@retriever("name")``.
    """,
)
chat_engine = _make(
    SpanKind.AGENT,
    """Decorate a LlamaIndex chat function (``ChatEngine.chat``) as AGENT.

    Usable as ``@chat_engine`` or ``@chat_engine("name")``.
    """,
)
ingestion = _make(
    SpanKind.EMBEDDING,
    """Decorate a LlamaIndex ingestion run (``IngestionPipeline.run``).

    Usable as ``@ingestion`` or ``@ingestion("name")``.
    """,
)
