"""Result transformation for RETRIEVER and TOOL spans.

Extracted from ``decorators.py`` to keep span-lifecycle orchestration
separate from result-specific logic.  All functions are internal — the
public surface remains ``@trace(result_formatter=...)``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from ragwatch.core.span_kinds import SpanKind
from ragwatch.instrumentation.attributes import safe_set_attribute


def _accepts_context_transform(method: Any) -> bool:
    """Return True if *method* accepts a single ``context`` positional arg."""
    try:
        sig = inspect.signature(method)
        params = [
            p for p in sig.parameters.values()
            if p.name != "self" and p.default is inspect.Parameter.empty
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(params) == 1 and params[0].name == "context"
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Protocol + Registry
# ---------------------------------------------------------------------------

@runtime_checkable
class ResultTransformer(Protocol):
    """Contract for pluggable result transformers.

    Implementors provide a ``span_kind`` they handle and a ``transform``
    method that converts the raw result into whatever the caller expects.
    """

    @property
    def span_kind(self) -> SpanKind:
        """The span kind this transformer handles."""
        ...

    def transform(
        self,
        span: Any,
        args: tuple,
        kwargs: dict,
        result: Any,
        result_formatter: Optional[Callable],
    ) -> Any:
        """Transform *result* and optionally record telemetry on *span*."""
        ...


class ResultTransformerRegistry:
    """Registry of result transformers keyed by :class:`SpanKind`."""

    def __init__(self) -> None:
        self._transformers: Dict[SpanKind, ResultTransformer] = {}

    def register(self, transformer: ResultTransformer) -> None:
        self._transformers[transformer.span_kind] = transformer

    def get(self, span_kind: SpanKind) -> Optional[ResultTransformer]:
        return self._transformers.get(span_kind)

    def clear(self) -> None:
        self._transformers.clear()


_DEFAULT_TRANSFORMER_REGISTRY = ResultTransformerRegistry()


def get_default_transformer_registry() -> ResultTransformerRegistry:
    return _DEFAULT_TRANSFORMER_REGISTRY


def reset_default_transformer_registry() -> None:
    """Reset the global registry.  Intended for testing only."""
    _DEFAULT_TRANSFORMER_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Core transform dispatch
# ---------------------------------------------------------------------------

def transform_result(
    span: Any,
    span_kind: SpanKind,
    args: tuple,
    kwargs: dict,
    result: Any,
    result_formatter: Optional[Callable],
    *,
    context: Any = None,
) -> Any:
    """Record chunk telemetry and convert raw results to strings.

    If a custom :class:`ResultTransformer` is registered for *span_kind*,
    it takes precedence over the built-in handlers.  When a *context*
    (:class:`InstrumentationContext`) is provided and the transformer
    accepts it, the context-first path is used.

    AGENT/CHAIN spans: result is never modified (Command and state-update
    dicts must reach LangGraph unchanged).
    """
    # Check for user-registered custom transformer first
    custom = _DEFAULT_TRANSFORMER_REGISTRY.get(span_kind)
    if custom is not None:
        if context is not None and _accepts_context_transform(custom.transform):
            return custom.transform(context)
        return custom.transform(span, args, kwargs, result, result_formatter)

    if span_kind is SpanKind.RETRIEVER:
        return _handle_retriever_result(span, args, kwargs, result, result_formatter)
    if span_kind is SpanKind.TOOL:
        return _handle_tool_result(span, result, result_formatter)
    # AGENT / CHAIN / EMBEDDING — pass through untouched
    return result


def _handle_retriever_result(
    span: Any, args: tuple, kwargs: dict, result: Any,
    result_formatter: Optional[Callable],
) -> Any:
    """Auto-record chunks when result is a raw ``[(Document, score)]`` list."""
    if not _is_raw_retrieval(result):
        return result  # already a string (e.g. "NO_RELEVANT_CHUNKS")

    from ragwatch.instrumentation.helpers import record_chunks  # lazy

    query: str = kwargs.get("query", "")
    if not query:
        for arg in args:
            if isinstance(arg, str) and arg:
                query = arg
                break

    record_chunks(result, query=query, span=span)

    if result_formatter is not None:
        return result_formatter(result)

    # Built-in default formatter
    return "\n\n".join(
        f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
        f"File Name: {doc.metadata.get('source', '')}\n"
        f"Content: {doc.page_content.strip()}"
        for doc, _ in result
    )


def _handle_tool_result(
    span: Any, result: Any, result_formatter: Optional[Callable],
) -> Any:
    """Auto-record parent-chunk attributes from raw dict / list-of-dict results.

    Built-in formatters are provided for the two canonical parent-chunk shapes
    so application code never needs ``result_formatter=``.
    """
    if isinstance(result, dict) and "parent_id" in result:
        _record_parent_chunk_attrs(span, [result])
        if result_formatter is not None:
            return result_formatter(result)
        # Built-in default: single parent chunk → tool string
        return (
            f"Parent ID: {result.get('parent_id', 'n/a')}\n"
            f"File Name: {result.get('metadata', {}).get('source', 'unknown')}\n"
            f"Content: {result.get('content', '').strip()}"
        )

    if (isinstance(result, list) and result
            and isinstance(result[0], dict) and "parent_id" in result[0]):
        _record_parent_chunk_attrs(span, result)
        if result_formatter is not None:
            return result_formatter(result)
        # Built-in default: list of parent chunks → tool string
        return "\n\n".join(
            f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
            f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
            f"Content: {doc.get('content', '').strip()}"
            for doc in result
        )

    # String, error sentinel, or unrecognised type
    if result_formatter is not None and not isinstance(result, str):
        return result_formatter(result)
    return result


def _record_parent_chunk_attrs(span: Any, docs: list) -> None:
    if not span.is_recording():
        return
    safe_set_attribute(span, "retrieval.parent_chunk_count", len(docs))
    for i, doc in enumerate(docs):
        content = doc.get("content", "").strip()
        pfx = f"retrieval.parent.{i}"
        safe_set_attribute(span, f"{pfx}.parent_id", str(doc.get("parent_id", "")))
        safe_set_attribute(span, f"{pfx}.source",
                           str(doc.get("metadata", {}).get("source", "")))
        safe_set_attribute(span, f"{pfx}.char_count", len(content))
        safe_set_attribute(span, f"{pfx}.content",
                           content[:600] + (" …[truncated]" if len(content) > 600 else ""))


def _is_raw_retrieval(result: Any) -> bool:
    """True if result looks like ``[(Document, float)]``."""
    return (isinstance(result, list) and bool(result)
            and isinstance(result[0], tuple) and len(result[0]) == 2
            and isinstance(result[0][1], (int, float)))
