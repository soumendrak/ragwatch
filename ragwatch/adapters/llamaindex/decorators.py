"""LlamaIndex-specific decorators.

- ``@query_engine`` — wraps a query function with ``SpanKind.CHAIN``.
- ``@retriever``    — wraps a retrieval function with ``SpanKind.RETRIEVER``.
- ``@chat_engine``  — wraps a chat function with ``SpanKind.AGENT``.
- ``@ingestion``    — wraps an ingestion/embedding run with ``SpanKind.EMBEDDING``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, TypeVar

from ragwatch.core.span_kinds import SpanKind
from ragwatch.instrumentation.decorators import trace
from ragwatch.instrumentation.span_hooks import SpanHook

F = TypeVar("F", bound=Callable[..., Any])


def _make(kind: SpanKind, doc: str) -> Callable[..., Any]:
    def decorator_factory(
        func: F | str | None = None,
        *,
        span_name: str | None = None,
        auto_track_io: bool = True,
        span_hooks: Optional[List[SpanHook]] = None,
    ) -> F | Callable[[F], F]:
        options = dict(
            span_kind=kind,
            auto_track_io=auto_track_io,
            span_hooks=span_hooks,
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
