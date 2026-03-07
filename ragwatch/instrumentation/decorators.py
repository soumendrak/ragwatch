"""The ``@trace`` decorator — core instrumentation primitive for RAGWatch.

All telemetry extraction is driven by decorator parameters.
Zero observability code is required inside decorated functions.

  telemetry=["tool_calls", "routing"]   — auto-extract from result/state
  result_formatter=my_fn                 — convert raw result → string for caller
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, List, Optional, TypeVar, overload

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
) -> Callable[[F], F]: ...


def trace(
    func: F | str | None = None,
    *,
    span_name: str | None = None,
    span_kind: SpanKind = SpanKind.CHAIN,
    auto_track_io: bool = True,
    telemetry: Optional[List[str]] = None,
    result_formatter: Optional[Callable] = None,
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
    """
    if callable(func):
        return _make_wrapper(func, span_name=None, span_kind=span_kind,
                             auto_track_io=auto_track_io, telemetry=telemetry,
                             result_formatter=result_formatter)

    actual_name = func if isinstance(func, str) else span_name

    def decorator(fn: F) -> F:
        return _make_wrapper(fn, span_name=actual_name, span_kind=span_kind,
                             auto_track_io=auto_track_io, telemetry=telemetry,
                             result_formatter=result_formatter)

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
) -> F:
    resolved_name = span_name or func.__qualname__
    _tel = list(telemetry) if telemetry else []

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(resolved_name) as span:
                span.set_attribute(OPENINFERENCE_SPAN_KIND, span_kind.value)
                if auto_track_io:
                    track_input(span, args, kwargs)
                try:
                    raw = await func(*args, **kwargs)
                except Exception as exc:
                    span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise
                # Telemetry on raw result (before any transformation)
                if _tel:
                    _extract_node_telemetry(span, resolved_name, args, raw, _tel)
                # Result transformation (RETRIEVER / TOOL only)
                result = _transform_result(span, span_kind, args, kwargs, raw, result_formatter)
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
                raw = func(*args, **kwargs)
            except Exception as exc:
                span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise
            # Telemetry on raw result (before any transformation)
            if _tel:
                _extract_node_telemetry(span, resolved_name, args, raw, _tel)
            # Result transformation (RETRIEVER / TOOL only)
            result = _transform_result(span, span_kind, args, kwargs, raw, result_formatter)
            if auto_track_io:
                track_output(span, result)
            _handle_embedding_context(span_kind, result)
            _extract_token_usage(span, result)
            return result

    return sync_wrapper  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Result transformation (RETRIEVER / TOOL spans)
# ─────────────────────────────────────────────────────────────────────────────

def _transform_result(
    span: Any,
    span_kind: SpanKind,
    args: tuple,
    kwargs: dict,
    result: Any,
    result_formatter: Optional[Callable],
) -> Any:
    """Record chunk telemetry and convert raw results to strings.

    AGENT/CHAIN spans: result is never modified (Command and state-update
    dicts must reach LangGraph unchanged).
    """
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
    span.set_attribute("retrieval.parent_chunk_count", len(docs))
    for i, doc in enumerate(docs):
        content = doc.get("content", "").strip()
        pfx = f"retrieval.parent.{i}"
        span.set_attribute(f"{pfx}.parent_id", str(doc.get("parent_id", "")))
        span.set_attribute(f"{pfx}.source",
                           str(doc.get("metadata", {}).get("source", "")))
        span.set_attribute(f"{pfx}.char_count", len(content))
        span.set_attribute(f"{pfx}.content",
                           content[:600] + (" …[truncated]" if len(content) > 600 else ""))


def _is_raw_retrieval(result: Any) -> bool:
    """True if result looks like ``[(Document, float)]``."""
    return (isinstance(result, list) and bool(result)
            and isinstance(result[0], tuple) and len(result[0]) == 2
            and isinstance(result[0][1], (int, float)))


# ─────────────────────────────────────────────────────────────────────────────
# Node-level telemetry extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_node_telemetry(
    span: Any,
    span_name: str,
    args: tuple,
    result: Any,
    telemetry: List[str],
) -> None:
    """Inspect state (args) and result to emit structured telemetry.

    Fully duck-typed — no LangGraph / LangChain imports in SDK core.
    All helper imports are lazy to avoid circular dependencies.
    """
    from ragwatch.instrumentation.helpers import (  # lazy
        record_agent_completion,
        record_context_compression,
        record_query_rewrite,
        record_routing,
        record_tool_calls,
    )

    # Find the LangGraph state dict in positional args
    state: Optional[dict] = next(
        (a for a in args if isinstance(a, dict)), None
    )

    # ── tool_calls + routing (orchestrator) ───────────────────────────────────
    if "tool_calls" in telemetry or ("routing" in telemetry and isinstance(result, dict)):
        tc_list = _tool_calls_from_messages(result)

        if "tool_calls" in telemetry:
            record_tool_calls(tc_list, span=span)

        if "routing" in telemetry and isinstance(result, dict):
            if tc_list:
                names = [tc.get("name", "") for tc in tc_list]
                record_routing(span_name, "tools",
                                reason=f"calling: {names}", span=span)
            else:
                record_routing(span_name, "collect_answer",
                                reason="answer ready, no tool calls", span=span)

    # ── routing from Command (should_compress_context) ────────────────────────
    if "routing" in telemetry and _is_command(result):
        goto = str(result.goto)
        reason = f"tokens≈{_rough_tokens(state)} → {goto}" if state else goto
        record_routing(span_name, goto, reason=reason, span=span)

    # ── agent_completion (collect_answer) ─────────────────────────────────────
    if "agent_completion" in telemetry and isinstance(result, dict):
        agent_answers = result.get("agent_answers", [])
        final_answer  = result.get("final_answer", "")
        answer = (agent_answers[-1].get("answer", final_answer)
                  if agent_answers and isinstance(agent_answers[-1], dict)
                  else final_answer)
        is_fallback = (answer == "Unable to generate an answer."
                       or not bool((answer or "").strip()))
        record_agent_completion(
            status="fallback" if is_fallback else "success",
            iteration_count=state.get("iteration_count", 0) if state else 0,
            tool_call_count=state.get("tool_call_count", 0) if state else 0,
            question=state.get("question", "") if state else "",
            question_index=state.get("question_index", 0) if state else 0,
            answer_length=len(answer or ""),
            is_fallback=is_fallback,
            span=span,
        )

    # ── query_rewrite (rewrite_query) ─────────────────────────────────────────
    if "query_rewrite" in telemetry and isinstance(result, dict):
        rewritten  = result.get("rewrittenQuestions", [])
        is_clear   = result.get("questionIsClear", False)
        original   = result.get("originalQuery", "")
        if not original and state:
            msgs = state.get("messages", [])
            original = getattr(msgs[-1], "content", "") if msgs else ""
        record_query_rewrite(original, rewritten, is_clear, span=span)

    # ── compression (compress_context) ────────────────────────────────────────
    if "compression" in telemetry and isinstance(result, dict) and state is not None:
        msgs        = state.get("messages", [])
        old_summary = state.get("context_summary", "") or ""
        new_summary = result.get("context_summary", "") or ""
        tokens_before = (
            sum(len(getattr(m, "content", "") or "") for m in msgs) // 4
            + len(old_summary) // 4
        )
        tokens_after = len(new_summary) // 4
        retrieved_ids = state.get("retrieval_keys", set()) or set()
        parents = sorted(r.replace("parent::", "") for r in retrieved_ids
                         if r.startswith("parent::"))
        queries = sorted(r.replace("search::", "") for r in retrieved_ids
                         if r.startswith("search::"))
        record_context_compression(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            queries_run=queries or None,
            parents_retrieved=parents or None,
            span=span,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tool_calls_from_messages(result: Any) -> list:
    if not isinstance(result, dict):
        return []
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "tool_calls"):
            return msg.tool_calls or []
    return []


def _is_command(obj: Any) -> bool:
    return hasattr(obj, "goto") and hasattr(obj, "update")


def _rough_tokens(state: Optional[dict]) -> int:
    if state is None:
        return 0
    msgs = state.get("messages", [])
    summary = state.get("context_summary", "") or ""
    return sum(len(getattr(m, "content", "") or "") for m in msgs) // 4 + len(summary) // 4


# ─────────────────────────────────────────────────────────────────────────────
# Existing hooks (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_embedding_context(span_kind: SpanKind, result: Any) -> None:
    if span_kind is SpanKind.EMBEDDING:
        store_embedding_in_context(result)


def _extract_token_usage(span: Any, result: Any) -> None:
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

    prompt = completion = total = 0
    found = False
    for obj in candidates:
        usage = getattr(obj, "usage_metadata", None)
        if isinstance(usage, dict):
            found = True
            prompt     += usage.get("input_tokens", 0)
            completion += usage.get("output_tokens", 0)
            total      += usage.get("total_tokens", 0)
    if found:
        span.set_attribute(LLM_TOKEN_COUNT_PROMPT, prompt)
        span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, completion)
        span.set_attribute(LLM_TOKEN_COUNT_TOTAL, total)
