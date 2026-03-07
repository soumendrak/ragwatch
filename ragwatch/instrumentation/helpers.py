"""Convenience helpers for recording rich structured telemetry.

These functions are intentionally **framework-agnostic** — they only depend on
the OpenTelemetry API and RAGWatch's own semconv.  Call them from any node,
tool, or edge function to add minute-level observability without modifying
business logic:

    from ragwatch import record_chunks, record_agent_completion, record_routing
    from ragwatch import record_tool_calls, record_context_compression

All helpers are also exported from the top-level ``ragwatch`` package.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from opentelemetry import trace as otel_trace

from ragwatch.instrumentation.semconv import (
    AGENT_ANSWER_LENGTH,
    AGENT_COMPLETION_STATUS,
    AGENT_IS_FALLBACK,
    AGENT_ITERATION_COUNT,
    AGENT_QUESTION,
    AGENT_QUESTION_INDEX,
    AGENT_TOOL_CALL_COUNT,
    CHUNK_AVG_SCORE,
    CHUNK_COUNT,
    CHUNK_MAX_SCORE,
    CHUNK_MIN_SCORE,
    CHUNK_PREFIX,
    CHUNK_QUERY,
    CHUNK_RELEVANCE_SCORE,
    CHUNK_RELEVANCE_SCORES,
    COMPRESSION_PARENTS_RETRIEVED,
    COMPRESSION_QUERIES_RUN,
    COMPRESSION_RATIO,
    COMPRESSION_TOKENS_AFTER,
    COMPRESSION_TOKENS_BEFORE,
    COMPRESSION_UNIQUE_PARENTS,
    COMPRESSION_UNIQUE_QUERIES,
    LLM_TOOL_CALL_PREFIX,
    LLM_TOOL_CALLS_COUNT,
    LLM_TOOL_CALLS_NAMES,
    QUERY_IS_CLEAR,
    QUERY_ORIGINAL,
    QUERY_REWRITTEN_COUNT,
    QUERY_REWRITTEN_QUESTIONS,
    ROUTING_FROM_NODE,
    ROUTING_REASON,
    ROUTING_TO_NODE,
)

# Maximum characters of chunk content stored per chunk — keeps spans lean.
_DEFAULT_CHUNK_CONTENT_CHARS = 600


# ---------------------------------------------------------------------------
# record_chunks
# ---------------------------------------------------------------------------

def record_chunks(
    results: Sequence[Tuple[Any, float]],
    *,
    query: str = "",
    span: Optional[otel_trace.Span] = None,
    max_content_chars: int = _DEFAULT_CHUNK_CONTENT_CHARS,
) -> None:
    """Record per-chunk retrieval telemetry on the current span.

    Designed for LangChain ``similarity_search_with_relevance_scores`` output
    but works with any ``(document, score)`` sequence where the document has
    ``.page_content`` and ``.metadata``.

    Sets the following attributes::

        retrieval.chunk_count           = N
        retrieval.chunk.avg_score       = 0.87
        retrieval.chunk.min_score       = 0.72
        retrieval.chunk.max_score       = 0.93
        retrieval.query                 = "what is …"
        retrieval.chunk.0.content       = "The quick brown fox …"
        retrieval.chunk.0.score         = 0.93
        retrieval.chunk.0.source        = "doc_name.md"
        retrieval.chunk.0.parent_id     = "abc123"
        retrieval.chunk.0.char_count    = 487
        retrieval.chunk.1.*             = …

    These attributes are visible in Phoenix's span detail panel as a rich
    table; Jaeger shows them under "Tags".

    Args:
        results: Sequence of ``(Document, relevance_score)`` tuples.
        query: The search query string (optional but highly recommended).
        span: OTel span to record on.  Defaults to the current active span.
        max_content_chars: Truncate content at this many chars to keep spans
            from growing too large.  Default: 600 chars.
    """
    sp = span or otel_trace.get_current_span()
    if not sp.is_recording() or not results:
        return

    scores = [float(score) for _, score in results]

    # --- Aggregate stats ---------------------------------------------------
    sp.set_attribute(CHUNK_COUNT, len(results))
    sp.set_attribute(CHUNK_RELEVANCE_SCORES, scores)       # legacy array
    avg = sum(scores) / len(scores)
    sp.set_attribute(CHUNK_RELEVANCE_SCORE, round(avg, 4))  # legacy scalar
    sp.set_attribute(CHUNK_AVG_SCORE, round(avg, 4))
    sp.set_attribute(CHUNK_MIN_SCORE, round(min(scores), 4))
    sp.set_attribute(CHUNK_MAX_SCORE, round(max(scores), 4))

    if query:
        sp.set_attribute(CHUNK_QUERY, query)

    # --- Per-chunk attributes ----------------------------------------------
    for i, (doc, score) in enumerate(results):
        pfx = f"{CHUNK_PREFIX}.{i}"
        content = getattr(doc, "page_content", "") or ""
        content = content.strip()
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + " …[truncated]"

        metadata = getattr(doc, "metadata", {}) or {}

        sp.set_attribute(f"{pfx}.content",   content)
        sp.set_attribute(f"{pfx}.score",     round(float(score), 4))
        sp.set_attribute(f"{pfx}.source",    str(metadata.get("source", "")))
        sp.set_attribute(f"{pfx}.parent_id", str(metadata.get("parent_id", "")))
        sp.set_attribute(f"{pfx}.char_count", len(getattr(doc, "page_content", "") or ""))


# ---------------------------------------------------------------------------
# record_agent_completion
# ---------------------------------------------------------------------------

def record_agent_completion(
    status: str,
    *,
    iteration_count: int = 0,
    tool_call_count: int = 0,
    question: str = "",
    question_index: int = 0,
    answer_length: int = 0,
    is_fallback: bool = False,
    span: Optional[otel_trace.Span] = None,
) -> None:
    """Record agent task completion metadata on the current span.

    Call this from ``collect_answer`` (or equivalent) to capture whether the
    agent succeeded, fell back, or hit its iteration ceiling.

    Sets::

        agent.completion_status   = "success" | "fallback" | "max_iterations" | "error"
        agent.iteration_count     = 3
        agent.tool_call_count     = 5
        agent.question            = "What is the revenue for Q3?"
        agent.question_index      = 1
        agent.answer_length       = 342
        agent.is_fallback         = false

    Args:
        status: Completion disposition. Use one of: ``"success"``,
            ``"fallback"``, ``"max_iterations"``, ``"error"``.
        iteration_count: Total LLM→tool loop iterations for this sub-agent.
        tool_call_count: Total tool invocations across all iterations.
        question: The sub-question this agent was answering.
        question_index: Position in the decomposed question list (0-based).
        answer_length: Character length of the final answer.
        is_fallback: ``True`` if the agent used the fallback response path.
        span: OTel span.  Defaults to the current active span.
    """
    sp = span or otel_trace.get_current_span()
    if not sp.is_recording():
        return

    sp.set_attribute(AGENT_COMPLETION_STATUS, status)
    sp.set_attribute(AGENT_ITERATION_COUNT, iteration_count)
    sp.set_attribute(AGENT_TOOL_CALL_COUNT, tool_call_count)
    sp.set_attribute(AGENT_IS_FALLBACK, is_fallback)

    if question:
        sp.set_attribute(AGENT_QUESTION, question)
    if question_index:
        sp.set_attribute(AGENT_QUESTION_INDEX, question_index)
    if answer_length:
        sp.set_attribute(AGENT_ANSWER_LENGTH, answer_length)

    # Emit a span event so the timeline shows the completion point clearly
    sp.add_event(
        "agent.task_completed",
        {
            "status": status,
            "iterations": iteration_count,
            "tool_calls": tool_call_count,
            "is_fallback": is_fallback,
        },
    )


# ---------------------------------------------------------------------------
# record_routing
# ---------------------------------------------------------------------------

def record_routing(
    from_node: str,
    to_node: str,
    reason: str = "",
    *,
    span: Optional[otel_trace.Span] = None,
) -> None:
    """Record a routing (edge) decision on the current span.

    Call this at the end of any node that controls flow to capture *where*
    execution is being directed and *why*.  In Phoenix this appears as a span
    event on the timeline, making the agent path immediately visible.

    Sets::

        routing.from_node = "should-compress-context"
        routing.to_node   = "orchestrator"
        routing.reason    = "tokens=1842 < threshold=2000, context ok"

    Plus a span **event** ``routing.decision`` visible as a timeline point.

    Args:
        from_node: Name of the node making the routing decision.
        to_node: Name of the destination node.
        reason: Human-readable reason for the routing choice.
        span: OTel span.  Defaults to the current active span.
    """
    sp = span or otel_trace.get_current_span()
    if not sp.is_recording():
        return

    sp.set_attribute(ROUTING_FROM_NODE, from_node)
    sp.set_attribute(ROUTING_TO_NODE, to_node)
    if reason:
        sp.set_attribute(ROUTING_REASON, reason)

    sp.add_event(
        "routing.decision",
        {
            "from": from_node,
            "to": to_node,
            "reason": reason or "—",
        },
    )


# ---------------------------------------------------------------------------
# record_tool_calls
# ---------------------------------------------------------------------------

def record_tool_calls(
    tool_calls: List[dict],
    *,
    span: Optional[otel_trace.Span] = None,
) -> None:
    """Record the LLM's tool-call decisions on the current orchestrator span.

    Call this **inside the orchestrator node** right after ``llm.invoke()``
    returns, passing ``response.tool_calls``.  This makes the orchestrator
    span in Phoenix show exactly which tools the model decided to call and
    with what arguments — without needing a separate tool-dispatch span.

    Sets::

        llm.tool_calls.count    = 1
        llm.tool_calls.names    = ["search_child_chunks"]
        llm.tool_call.0.name    = "search_child_chunks"
        llm.tool_call.0.args    = "{'query': 'RAG latency', 'limit': 5}"
        llm.tool_call.0.id      = "call_abc123"

    Plus a span **event** ``llm.tool_calls_decided`` on the timeline.

    Args:
        tool_calls: List of tool-call dicts from ``AIMessage.tool_calls``.
            Each entry has ``name``, ``args`` (dict), and ``id`` keys.
        span: OTel span.  Defaults to the current active span.
    """
    sp = span or otel_trace.get_current_span()
    if not sp.is_recording() or not tool_calls:
        return

    tool_names: List[str] = [tc.get("name", "") for tc in tool_calls]
    sp.set_attribute(LLM_TOOL_CALLS_COUNT, len(tool_calls))
    sp.set_attribute(LLM_TOOL_CALLS_NAMES, tool_names)

    for i, tc in enumerate(tool_calls):
        pfx = f"{LLM_TOOL_CALL_PREFIX}.{i}"
        sp.set_attribute(f"{pfx}.name", tc.get("name", ""))
        sp.set_attribute(f"{pfx}.args", str(tc.get("args", {})))
        sp.set_attribute(f"{pfx}.id",   str(tc.get("id", "")))

    sp.add_event(
        "llm.tool_calls_decided",
        {
            "tools": str(tool_names),
            "count": len(tool_calls),
        },
    )


# ---------------------------------------------------------------------------
# record_context_compression
# ---------------------------------------------------------------------------

def record_context_compression(
    tokens_before: int,
    tokens_after: int = 0,
    *,
    queries_run: Optional[List[str]] = None,
    parents_retrieved: Optional[List[str]] = None,
    span: Optional[otel_trace.Span] = None,
) -> None:
    """Record context compression statistics on the current span.

    Call this from the ``compress_context`` node to surface compression
    efficiency and deduplication info in Phoenix.

    Sets::

        context.compression.tokens_before       = 4200
        context.compression.tokens_after        = 1100
        context.compression.ratio               = 0.262
        context.compression.queries_already_run = ["what is RAG?", …]
        context.compression.unique_queries       = 2
        context.compression.parents_retrieved    = ["abc", "def"]
        context.compression.unique_parents       = 2

    Args:
        tokens_before: Token count in the context before compression.
        tokens_after: Token count after compression (estimated).
        queries_run: List of search query strings already executed
            (used for deduplication tracking).
        parents_retrieved: List of parent chunk IDs already fetched.
        span: OTel span.  Defaults to the current active span.
    """
    sp = span or otel_trace.get_current_span()
    if not sp.is_recording():
        return

    sp.set_attribute(COMPRESSION_TOKENS_BEFORE, tokens_before)

    if tokens_after:
        sp.set_attribute(COMPRESSION_TOKENS_AFTER, tokens_after)
        if tokens_before > 0:
            ratio = round(tokens_after / tokens_before, 3)
            sp.set_attribute(COMPRESSION_RATIO, ratio)

    if queries_run:
        sp.set_attribute(COMPRESSION_QUERIES_RUN, queries_run)
        sp.set_attribute(COMPRESSION_UNIQUE_QUERIES, len(queries_run))

    if parents_retrieved:
        sp.set_attribute(COMPRESSION_PARENTS_RETRIEVED, parents_retrieved)
        sp.set_attribute(COMPRESSION_UNIQUE_PARENTS, len(parents_retrieved))

    sp.add_event(
        "context.compressed",
        {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction_pct": round((1 - tokens_after / max(tokens_before, 1)) * 100, 1),
        },
    )


# ---------------------------------------------------------------------------
# record_query_rewrite
# ---------------------------------------------------------------------------

def record_query_rewrite(
    original_query: str,
    rewritten_questions: List[str],
    is_clear: bool,
    *,
    span: Optional[otel_trace.Span] = None,
) -> None:
    """Record query decomposition telemetry on the rewrite-query span.

    Shows what the LLM understood from the user's query, how many
    sub-questions were created, and whether it requested clarification.

    Sets::

        query.original            = "What is the total revenue and profit margin?"
        query.rewritten_count     = 2
        query.rewritten_questions = ["What is the total revenue?", "…margin?"]
        query.is_clear            = true

    Args:
        original_query: The raw user message.
        rewritten_questions: The decomposed question list.
        is_clear: Whether the query was deemed answerable.
        span: OTel span.
    """
    sp = span or otel_trace.get_current_span()
    if not sp.is_recording():
        return

    sp.set_attribute(QUERY_ORIGINAL, original_query[:1000])
    sp.set_attribute(QUERY_REWRITTEN_COUNT, len(rewritten_questions))
    sp.set_attribute(QUERY_REWRITTEN_QUESTIONS, rewritten_questions)
    sp.set_attribute(QUERY_IS_CLEAR, is_clear)

    sp.add_event(
        "query.rewritten",
        {
            "original": original_query[:200],
            "sub_questions": len(rewritten_questions),
            "is_clear": is_clear,
        },
    )
