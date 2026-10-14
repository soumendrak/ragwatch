"""LlamaIndex ``CallbackManager`` → OpenTelemetry bridge.

Register :class:`RagWatchCallbackHandler` once and every LlamaIndex event
(query, retrieve, synthesize, embedding, function call, …) becomes a RAGWatch
span; LLM events are skipped and left to OpenLLMetry::

    from llama_index.core import Settings
    from ragwatch.adapters.llamaindex import RagWatchCallbackHandler

    Settings.callback_manager.add_handler(RagWatchCallbackHandler())

The handler never imports LlamaIndex at module load, so ``llama-index-core``
stays an optional dependency.  Spans are keyed by LlamaIndex event IDs and
parented through LlamaIndex's own ``parent_id``, which LlamaIndex tracks in a
``contextvars`` stack, so concurrent asyncio queries never share spans.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span, StatusCode

from ragwatch.core.context import get_query_embedding
from ragwatch.instrumentation.decorators import _effective_auto_track_io
from ragwatch.core.span_kinds import SpanKind
from ragwatch.core.tracer import get_tracer
from ragwatch.instrumentation.attributes import safe_set_attribute
from ragwatch.instrumentation.embedding import (
    set_embedding_attributes,
    store_embedding_in_context,
)
from ragwatch.instrumentation.evaluators import _cosine_similarity
from ragwatch.instrumentation.helpers import record_chunks
from ragwatch.instrumentation.retrieval import set_retrieval_attributes
from ragwatch.instrumentation.semconv import (
    ERROR_TYPE,
    INPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_VALUE,
    RESPONSE_LENGTH,
)
from ragwatch.instrumentation.span_hooks import _is_strict_mode

try:  # Subclass the real base when LlamaIndex is installed (isinstance checks).
    from llama_index.core.callbacks.base_handler import (
        BaseCallbackHandler as _HandlerBase,
    )
except ImportError:  # pragma: no cover - exercised only without the extra
    _HandlerBase = object  # type: ignore[assignment,misc]

_logger = logging.getLogger(__name__)

# LlamaIndex event type (``CBEventType`` value) → OpenInference span kind.
_SPAN_KINDS: Dict[str, str] = {
    "query": SpanKind.CHAIN.value,
    "retrieve": SpanKind.RETRIEVER.value,
    "reranking": SpanKind.RETRIEVER.value,
    "synthesize": SpanKind.CHAIN.value,
    "embedding": SpanKind.EMBEDDING.value,
    "function_call": SpanKind.TOOL.value,
}
# LLM calls are left to OpenLLMetry (see AGENTS.md); spans for them would
# duplicate the provider spans.  Children of a skipped event are re-parented
# onto its nearest traced ancestor.
_DEFAULT_SKIPPED = frozenset({"llm"})
# Events whose single embedding is the query embedding, not an ingestion batch.
_QUERY_ANCESTORS = frozenset({"query", "retrieve"})


class _Chunk:
    """Adapt a LlamaIndex node to the ``page_content``/``metadata`` shape."""

    def __init__(self, node: Any) -> None:
        get_content = getattr(node, "get_content", None)
        self.page_content = get_content() if callable(get_content) else ""
        metadata = dict(getattr(node, "metadata", None) or {})
        metadata.setdefault("source", metadata.get("file_name", ""))
        metadata.setdefault("parent_id", getattr(node, "ref_doc_id", "") or "")
        self.metadata = metadata


def record_nodes(
    span: Span,
    nodes: Any,
    *,
    query: str = "",
    query_embedding: Optional[List[float]] = None,
) -> None:
    """Record ``NodeWithScore`` results as RAGWatch chunk telemetry.

    Uses the retriever's scores.  When any score is missing, falls back to
    cosine similarity between each node's embedding and *query_embedding*
    (``None`` means: use the one stored in the current context); if neither
    is available, only the returned-chunk count is recorded.
    """
    if not isinstance(nodes, (list, tuple)):
        return
    if nodes and not all(hasattr(n, "node") for n in nodes):
        return
    set_retrieval_attributes(span, chunks_returned=len(nodes))
    if not nodes:
        return
    scores: Sequence[Optional[float]] = [getattr(n, "score", None) for n in nodes]
    if any(s is None for s in scores):
        if query_embedding is None:
            query_embedding = get_query_embedding()
        embeddings = [getattr(n.node, "embedding", None) for n in nodes]
        if not query_embedding or any(e is None for e in embeddings):
            return
        scores = [_cosine_similarity(query_embedding, e) for e in embeddings]
    results: List[Tuple[Any, float]] = [
        (_Chunk(n.node), float(s)) for n, s in zip(nodes, scores)
    ]
    record_chunks(results, query=query, span=span)


class RagWatchCallbackHandler(_HandlerBase):  # type: ignore[misc,valid-type]
    """LlamaIndex callback handler that emits RAGWatch/OpenInference spans."""

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[Any]] = None,
        event_ends_to_ignore: Optional[List[Any]] = None,
    ) -> None:
        self.event_starts_to_ignore = tuple(event_starts_to_ignore or ())
        self.event_ends_to_ignore = tuple(event_ends_to_ignore or ())
        # event_id → (span, event type, parent event_id, query string)
        self._events: Dict[str, Tuple[Span, str, str, str]] = {}
        # skipped event_id → its parent event_id (for re-parenting children)
        self._skipped: Dict[str, str] = {}
        # query/retrieve event_id → the query embedding captured beneath it
        self._query_embeddings: Dict[str, List[float]] = {}

    # -- CallbackManager protocol -------------------------------------------

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        try:
            self._start(_value(event_type), payload or {}, event_id, parent_id)
        except Exception as exc:
            self._isolate("start", exc)
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        self._skipped.pop(event_id, None)
        entry = self._events.get(event_id)
        if entry is None:
            return
        span = entry[0]
        try:
            self._end(event_id, entry, payload or {})
        except Exception as exc:
            self._isolate("end", exc)
        finally:
            self._events.pop(event_id, None)
            self._query_embeddings.pop(event_id, None)
            span.end()

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Traces map onto the caller's current OTel span; nothing to do."""

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Spans are ended per event in :meth:`on_event_end`."""

    # -- internals -----------------------------------------------------------

    def _start(
        self, kind: str, payload: Dict[str, Any], event_id: str, parent_id: str
    ) -> None:
        if kind in _DEFAULT_SKIPPED:
            self._skipped[event_id] = parent_id
            return
        parent_id = self._resolve_parent(parent_id)
        parent = self._events.get(parent_id)
        context = otel_trace.set_span_in_context(parent[0]) if parent else None
        span = get_tracer().start_span(f"llama_index.{kind}", context=context)
        safe_set_attribute(
            span, OPENINFERENCE_SPAN_KIND, _SPAN_KINDS.get(kind, SpanKind.CHAIN.value)
        )
        query = _payload(payload, "query_str")
        query = str(query) if query is not None else ""
        track_io = _effective_auto_track_io(True)
        if query and track_io:
            safe_set_attribute(span, INPUT_VALUE, query)
        elif kind == "function_call":
            tool = _payload(payload, "tool")
            name = getattr(tool, "name", None)
            if name:
                safe_set_attribute(span, "tool.name", str(name))
            args = _payload(payload, "function_call")
            if args is not None and track_io:
                safe_set_attribute(span, INPUT_VALUE, str(args))
        if kind == "embedding":
            serialized = _payload(payload, "serialized") or {}
            model = (
                serialized.get("model_name") if isinstance(serialized, dict) else None
            )
            set_embedding_attributes(span, model_name=model)
        self._events[event_id] = (span, kind, parent_id, query)

    def _resolve_parent(self, parent_id: str) -> str:
        """Skip over events that were not traced (e.g. ``llm``)."""
        seen = set()
        while parent_id in self._skipped and parent_id not in seen:
            seen.add(parent_id)
            parent_id = self._skipped[parent_id]
        return parent_id

    def _end(
        self,
        event_id: str,
        entry: Tuple[Span, str, str, str],
        payload: Dict[str, Any],
    ) -> None:
        span, kind, parent_id, query = entry
        exc = _payload(payload, "exception")
        if isinstance(exc, BaseException):
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            safe_set_attribute(span, ERROR_TYPE, type(exc).__name__)
            return
        if kind in ("retrieve", "reranking"):
            record_nodes(
                span,
                _payload(payload, "nodes"),
                query=query,
                # [] (not None) so the ambient context is never consulted.
                query_embedding=self._scoped_query_embedding(event_id) or [],
            )
        elif kind == "embedding":
            self._end_embedding(span, parent_id, _payload(payload, "embeddings"))
        else:
            output = _payload(payload, "response")
            if output is None:
                output = _payload(payload, "function_call_response")
            text = _materialized_text(output)
            if text is None:
                return
            if _effective_auto_track_io(True):
                safe_set_attribute(span, OUTPUT_VALUE, text)
            if kind in ("query", "synthesize"):
                safe_set_attribute(span, RESPONSE_LENGTH, len(text))

    def _end_embedding(self, span: Span, parent_id: str, embeddings: Any) -> None:
        if not isinstance(embeddings, list) or not embeddings:
            return
        set_embedding_attributes(span, dimensions=len(embeddings[0]))
        if len(embeddings) == 1:
            ancestor = self._query_ancestor(parent_id)
            if ancestor is not None:
                embedding = list(embeddings[0])
                self._query_embeddings[ancestor] = embedding
                store_embedding_in_context(embedding)

    def _query_ancestor(self, event_id: str) -> Optional[str]:
        """Nearest ``query``/``retrieve`` event at or above *event_id*."""
        seen = set()
        while event_id in self._events and event_id not in seen:
            seen.add(event_id)
            _, kind, parent_id, _ = self._events[event_id]
            if kind in _QUERY_ANCESTORS:
                return event_id
            event_id = parent_id
        return None

    def _scoped_query_embedding(self, event_id: str) -> Optional[List[float]]:
        """Query embedding captured under this retrieval or its query.

        Scoped to LlamaIndex events rather than the ambient context, so a
        later retrieval never scores its nodes against a previous query.
        """
        seen = set()
        while event_id in self._events and event_id not in seen:
            seen.add(event_id)
            if event_id in self._query_embeddings:
                return self._query_embeddings[event_id]
            event_id = self._events[event_id][2]
        return None

    @staticmethod
    def _isolate(phase: str, exc: Exception) -> None:
        _logger.warning("RAGWatch LlamaIndex callback %s failed: %s", phase, exc)
        if _is_strict_mode():
            raise exc


def _value(event_type: Any) -> str:
    """Return the string value of a ``CBEventType`` (a ``str`` enum)."""
    return str(getattr(event_type, "value", event_type))


def _payload(payload: Dict[str, Any], key: str) -> Any:
    """Look up an ``EventPayload`` key; its members are ``str`` enums."""
    return payload.get(key)


def _materialized_text(output: Any) -> Optional[str]:
    """Text of an already-materialized response, else ``None``.

    Streaming responses are never stringified: doing so would drain the
    token generator before the caller can iterate it.
    """
    if output is None:
        return None
    if hasattr(output, "response_gen") or hasattr(output, "async_response_gen"):
        return None
    response = getattr(output, "response", None)
    if isinstance(response, str):
        return response
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    return str(output)
