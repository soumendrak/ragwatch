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

import contextvars

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span, StatusCode

from ragwatch.core.context import (
    clear_query_embedding,
    get_query_embedding,
    set_query_embedding,
)
from ragwatch.instrumentation.decorators import _effective_auto_track_io
from ragwatch.instrumentation.io_tracker import (
    _MAX_IO_BYTES,
    _get_io_policy,
    _safe_serialize,
)
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
    "agent_step": SpanKind.AGENT.value,
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
    if not span.is_recording():
        return  # sampled out: skip node materialization and scoring
    if nodes is None or isinstance(nodes, (str, bytes, dict)):
        return
    try:
        nodes = list(nodes)  # e.g. dict_values from RouterRetriever
    except TypeError:
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
        missing = [n for n, s in zip(nodes, scores) if s is None]
        if not query_embedding or any(
            getattr(n.node, "embedding", None) is None for n in missing
        ):
            return
        # Keep the retriever's/reranker's scores; only fill the gaps.
        scores = [
            s
            if s is not None
            else _cosine_similarity(query_embedding, n.node.embedding)
            for n, s in zip(nodes, scores)
        ]
    results: List[Tuple[Any, float]] = [
        (_Chunk(n.node), float(s)) for n, s in zip(nodes, scores)
    ]
    record_chunks(results, query=query, span=span)


_CURRENT_TRACE: contextvars.ContextVar[Optional[object]] = contextvars.ContextVar(
    "ragwatch_llamaindex_trace", default=None
)


class RagWatchCallbackHandler(_HandlerBase):  # type: ignore[misc,valid-type]
    """LlamaIndex callback handler that emits RAGWatch/OpenInference spans."""

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[Any]] = None,
        event_ends_to_ignore: Optional[List[Any]] = None,
    ) -> None:
        self.event_starts_to_ignore = tuple(event_starts_to_ignore or ())
        self.event_ends_to_ignore = tuple(event_ends_to_ignore or ())
        self._ignored_ends = frozenset(_value(e) for e in self.event_ends_to_ignore)
        # event_id → (span, event type, parent event_id, query string)
        self._events: Dict[str, Tuple[Span, str, str, str]] = {}
        # skipped event_id → its parent event_id (for re-parenting children)
        self._skipped: Dict[str, str] = {}
        # query/retrieve event_id → the query embedding captured beneath it
        self._query_embeddings: Dict[str, Optional[List[float]]] = {}
        # event_id → OTel context token (span is current for the event)
        self._tokens: Dict[str, Any] = {}
        # query/retrieve event_id → ambient query embedding to restore at end
        self._previous_embeddings: Dict[str, Optional[List[float]]] = {}
        # event_id → trace marker active (task-locally) when it started
        self._owners: Dict[str, object] = {}
        # trace marker → token restoring the previous marker at end_trace
        self._trace_tokens: Dict[int, Any] = {}

    # -- CallbackManager protocol -------------------------------------------

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        owner = _CURRENT_TRACE.get()
        if owner is not None:
            self._owners[event_id] = owner
        try:
            self._start(_value(event_type), payload or {}, event_id, parent_id)
        except Exception as exc:
            entry = self._events.get(event_id)
            self._isolate("start", exc, entry[0] if entry else None)
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
            self._owners.pop(event_id, None)
            return
        span = entry[0]
        try:
            self._end(event_id, entry, payload or {})
        except Exception as exc:
            self._isolate("end", exc, span)
        finally:
            self._finish(event_id)

    def _finish(self, event_id: str) -> None:
        """End the event's span and undo its context changes."""
        entry = self._events.pop(event_id, None)
        self._owners.pop(event_id, None)
        self._query_embeddings.pop(event_id, None)
        if event_id in self._previous_embeddings:
            previous = self._previous_embeddings.pop(event_id)
            if previous is None:
                clear_query_embedding()
            else:
                set_query_embedding(previous)
        token = self._tokens.pop(event_id, None)
        if token is not None:
            try:
                otel_context.detach(token)
            except Exception:  # ended in a different context; nothing to undo
                pass
        if entry is not None:
            entry[0].end()

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Mark events started in this task as belonging to a new trace.

        The marker lives in a ``ContextVar`` so concurrent ``aquery()``
        tasks sharing one ``CallbackManager`` keep separate ownership.
        """
        marker = object()
        self._trace_tokens[id(marker)] = (_CURRENT_TRACE.set(marker), marker)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End this trace's spans whose events never ended.

        Only events started under this task's ``start_trace`` marker are
        finalized. *trace_map* is ignored: LlamaIndex keeps it on the shared
        manager, so it can list another concurrent task's events.
        """
        marker = _CURRENT_TRACE.get()
        if marker is None:
            return
        owned = [e for e, m in self._owners.items() if m is marker]
        for event_id in reversed(list(self._events)):
            if event_id in owned:
                self._finish(event_id)
        for event_id in owned:
            self._skipped.pop(event_id, None)
            self._owners.pop(event_id, None)
        token, _ = self._trace_tokens.pop(id(marker), (None, None))
        if token is not None:
            try:
                _CURRENT_TRACE.reset(token)
            except ValueError:  # ended in a different context
                _CURRENT_TRACE.set(None)

    # -- internals -----------------------------------------------------------

    def _start(
        self, kind: str, payload: Dict[str, Any], event_id: str, parent_id: str
    ) -> None:
        if kind in _DEFAULT_SKIPPED or kind in self._ignored_ends:
            # No span when its end will never arrive (LlamaIndex suppresses
            # on_event_end for event_ends_to_ignore); children re-parent.
            self._skipped[event_id] = parent_id
            return
        parent_id = self._resolve_parent(parent_id)
        if kind == "exception":
            # Start-only event: record on the nearest traced span, no new span.
            exc = _payload(payload, "exception")
            parent = self._events.get(parent_id)
            if parent is not None and isinstance(exc, BaseException):
                _record_error(parent[0], exc)
            return
        parent = self._events.get(parent_id)
        context = otel_trace.set_span_in_context(parent[0]) if parent else None
        span = get_tracer().start_span(f"llama_index.{kind}", context=context)
        # Make the span current for the event's lifetime so provider spans
        # (e.g. OpenLLMetry) started during it nest beneath it.
        self._tokens[event_id] = otel_context.attach(
            otel_trace.set_span_in_context(span)
        )
        # Register before any fallible payload processing so a failure
        # below still leaves the span reachable by on_event_end/end_trace.
        self._events[event_id] = (span, kind, parent_id, "")
        if kind in _QUERY_ANCESTORS:
            # Snapshot before fallible work so _finish can always restore it.
            self._previous_embeddings[event_id] = get_query_embedding()
        safe_set_attribute(
            span, OPENINFERENCE_SPAN_KIND, _SPAN_KINDS.get(kind, SpanKind.CHAIN.value)
        )
        query = _payload(payload, "query_str")
        query = str(query) if query is not None else ""
        track_io = _effective_auto_track_io(True)
        if query and track_io:
            safe_set_attribute(span, INPUT_VALUE, _io_text(query))
        elif kind == "function_call":
            tool = _payload(payload, "tool")
            name = getattr(tool, "name", None)
            if name:
                safe_set_attribute(span, "tool.name", str(name))
            args = _payload(payload, "function_call")
            if args is not None and track_io:
                safe_set_attribute(span, INPUT_VALUE, _io_text(args))
        top_k = _payload(payload, "top_k")
        if isinstance(top_k, int) and kind in ("retrieve", "reranking"):
            set_retrieval_attributes(span, top_k=top_k)
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
            _record_error(span, exc)
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
                # Scrub the raw structure so key-based redaction still sees
                # nested keys; only plain text/response objects use the text.
                raw = output if isinstance(output, (dict, list, tuple)) else text
                safe_set_attribute(span, OUTPUT_VALUE, _io_text(raw))
            if kind in ("query", "synthesize"):
                safe_set_attribute(span, RESPONSE_LENGTH, len(text))

    def _end_embedding(self, span: Span, parent_id: str, embeddings: Any) -> None:
        if not isinstance(embeddings, list) or not embeddings:
            return
        set_embedding_attributes(span, dimensions=len(embeddings[0]))
        if len(embeddings) == 1:
            ancestor = self._query_ancestor(parent_id)
            if ancestor is None:
                return
            if ancestor in self._query_embeddings:
                # Several query embeddings (e.g. multiple custom_embedding_strs)
                # are aggregated by LlamaIndex; we can't reproduce that, so
                # don't score against any single one of them.
                self._query_embeddings[ancestor] = None
                return
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
                return self._query_embeddings[event_id]  # None if ambiguous
            event_id = self._events[event_id][2]
        return None

    @staticmethod
    def _isolate(phase: str, exc: Exception, span: Optional[Span] = None) -> None:
        _logger.warning("RAGWatch LlamaIndex callback %s failed: %s", phase, exc)
        if span is not None:
            try:  # make missing telemetry distinguishable from a clean event
                span.add_event(
                    "ragwatch.callback_error",
                    {
                        "phase": phase,
                        ERROR_TYPE: type(exc).__name__,
                        "error.message": str(exc),
                    },
                )
            except Exception:
                pass
        if _is_strict_mode():
            raise exc


def _record_error(span: Span, exc: BaseException) -> None:
    span.set_status(StatusCode.ERROR, str(exc))
    span.record_exception(exc)
    safe_set_attribute(span, ERROR_TYPE, type(exc).__name__)


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
    if isinstance(output, str):
        return output
    if _is_streaming(output):
        return None
    response = getattr(output, "response", None)
    if isinstance(response, str):
        return response
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    return str(output)


def _is_streaming(output: Any) -> bool:
    """True for a LlamaIndex streaming response whose text is not yet read."""
    if output is None or isinstance(output, str):
        return False
    # StreamingResponse, AsyncStreamingResponse, StreamingAgentChatResponse.
    # The last one sets .response to a string while it still streams.
    if "Streaming" in type(output).__name__:
        return True
    # AgentChatResponse exposes response_gen for optional dummy streaming
    # but already has its text, so a str .response means materialized.
    if isinstance(getattr(output, "response", None), str):
        return False
    return hasattr(output, "response_gen") or hasattr(output, "async_response_gen")


def _io_text(value: Any) -> str:
    """Scrub *value* with the I/O policy and cap it like auto I/O tracking."""
    policy = _get_io_policy()
    if policy is not None:
        value = policy.scrub_io_payload(value)
    text = value if isinstance(value, str) else _safe_serialize(value)
    if len(text) > _MAX_IO_BYTES:
        text = text[:_MAX_IO_BYTES] + "...[truncated]"
    return text
