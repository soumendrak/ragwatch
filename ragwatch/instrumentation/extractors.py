"""Pluggable telemetry extractor registry.

Provides a ``TelemetryExtractor`` protocol and an ``ExtractorRegistry`` that
replaces the hardcoded ``if "x" in telemetry`` switchboard.  Built-in
extractors wrap the existing ``helpers.record_*()`` functions — zero behaviour
change, but users can now register custom extractors without touching SDK
internals.

Usage — context-first extractor (recommended)::

    from ragwatch.instrumentation.extractors import (
        TelemetryExtractor, get_default_registry,
    )

    class LatencyExtractor:
        name = "latency"
        def extract(self, context):
            result = context.raw_result
            if isinstance(result, dict) and "latency" in result:
                context.set_attribute("custom.latency_ms", result["latency"])

    get_default_registry().register(LatencyExtractor())

Usage — legacy extractor (still supported)::

    class LegacyExtractor:
        name = "latency"
        def extract(self, span, span_name, args, result, state):
            from ragwatch.instrumentation.attributes import safe_set_attribute
            safe_set_attribute(span, "custom.latency_ms", result.get("latency"))

Note: Always prefer ``context.set_attribute()`` over raw
``span.set_attribute()`` to ensure policy enforcement.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from opentelemetry import trace as otel_trace

from ragwatch.core.span_kinds import SpanKind

_logger = logging.getLogger(__name__)


def _accepts_context(method: Any) -> bool:
    """Return True if *method* accepts a single ``context`` positional arg.

    Used to detect new-style extractors whose ``extract`` signature is
    ``extract(self, context)`` vs the legacy
    ``extract(self, span, span_name, args, result, state)``.
    """
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


def _is_strict_mode() -> bool:
    """Check if the SDK is in strict mode (re-raise extension errors)."""
    try:
        from ragwatch import get_active_config
        cfg = get_active_config()
        return cfg.strict_mode if cfg is not None else False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TelemetryExtractor(Protocol):
    """Contract for pluggable telemetry extractors.

    Implementors must provide a ``name`` attribute and an ``extract`` method.
    The ``name`` is the string users pass in ``telemetry=["name"]``.

    Two signatures are supported — the registry auto-detects which one
    your extractor implements.

    **Canonical (context-first — recommended):**

    .. code-block:: python

        class MyExtractor:
            name = "my_ext"
            def extract(self, context: InstrumentationContext) -> None:
                result = context.raw_result
                context.set_attribute("my.key", result.get("val"))

    **Legacy (still supported):**

    .. code-block:: python

        class MyExtractor:
            name = "my_ext"
            def extract(self, span, span_name, args, result, state) -> None:
                from ragwatch.instrumentation.attributes import safe_set_attribute
                safe_set_attribute(span, "my.key", result.get("val"))

    Context-first extractors receive the full
    :class:`~ragwatch.instrumentation.context_model.InstrumentationContext`
    which provides ``span``, ``state``, ``raw_result``, ``adapter``, and the
    policy-enforced :meth:`set_attribute` writer.
    """

    name: str

    def extract(self, *args: Any, **kwargs: Any) -> None:
        """Extract telemetry — see class docstring for supported signatures."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ExtractorRegistry:
    """Name→extractor mapping with batch execution."""

    def __init__(self) -> None:
        self._extractors: Dict[str, TelemetryExtractor] = {}

    def register(self, extractor: TelemetryExtractor) -> None:
        """Register an extractor (overwrites any existing one with same name)."""
        self._extractors[extractor.name] = extractor

    def unregister(self, name: str) -> None:
        """Remove an extractor by name.  No-op if not found."""
        self._extractors.pop(name, None)

    def get(self, name: str) -> Optional[TelemetryExtractor]:
        """Return the extractor registered under *name*, or ``None``."""
        return self._extractors.get(name)

    def names(self) -> List[str]:
        """Return the list of registered extractor names."""
        return list(self._extractors.keys())

    def extract_all(
        self,
        names: List[str],
        span: otel_trace.Span,
        span_name: str,
        args: tuple,
        result: Any,
        *,
        kwargs: Optional[dict] = None,
        adapter: Any = None,
        context: Any = None,
    ) -> None:
        """Run every extractor whose name is in *names*.

        If a *context* (:class:`InstrumentationContext`) is provided and
        the extractor's ``extract`` method accepts a single ``context``
        parameter, the context-first path is used.  Otherwise falls back
        to the legacy ``(span, span_name, args, result, state)`` call.

        State resolution: uses ``context.state`` when available, else
        ``adapter.extract_state()``, else first ``dict`` in *args*.
        """
        if context is not None and context.state is not None:
            state = context.state
        elif adapter is not None:
            state = adapter.extract_state(args, kwargs or {})
        else:
            state: Optional[dict] = next(
                (a for a in args if isinstance(a, dict)), None
            )
        strict = _is_strict_mode()
        for name in names:
            ext = self._extractors.get(name)
            if ext is not None:
                try:
                    if context is not None and _accepts_context(ext.extract):
                        ext.extract(context)
                    else:
                        ext.extract(span, span_name, args, result, state)
                except Exception as exc:
                    _logger.warning("Extractor %r failed: %s", name, exc)
                    span.add_event("ragwatch.extractor_error", {
                        "extractor": name, "error": str(exc),
                    })
                    if strict:
                        raise


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------

def _tool_calls_from_messages(result: Any) -> list:
    """Extract tool calls from the last message in ``result["messages"]``."""
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


class ToolCallsExtractor:
    """Extract LLM tool-call decisions from orchestrator results.

    Reads ``ctx.normalized["tool_calls"]`` when available (adapter-driven),
    otherwise falls back to scanning ``result["messages"]`` directly.
    """

    name = "tool_calls"

    def extract(self, *args: Any, **kwargs: Any) -> None:
        from ragwatch.instrumentation.helpers import record_tool_calls  # lazy

        # Context-first path
        if len(args) == 1 and hasattr(args[0], "normalized"):
            ctx = args[0]
            norm = ctx.normalized or {}
            tc_list = norm.get("tool_calls") or _tool_calls_from_messages(ctx.raw_result)
            record_tool_calls(tc_list, span=ctx.span)
            return

        # Legacy path: (span, span_name, args, result, state)
        span, span_name, fn_args, result, state = args
        tc_list = _tool_calls_from_messages(result)
        record_tool_calls(tc_list, span=span)


class RoutingExtractor:
    """Extract routing decisions from dict results or Command objects.

    Reads ``ctx.normalized["routing_target"]`` when available (adapter-driven),
    otherwise falls back to framework-specific result inspection.
    """

    name = "routing"

    def extract(self, *args: Any, **kwargs: Any) -> None:
        from ragwatch.instrumentation.helpers import record_routing  # lazy

        # Context-first path
        if len(args) == 1 and hasattr(args[0], "normalized"):
            ctx = args[0]
            norm = ctx.normalized or {}
            if "routing_target" in norm:
                record_routing(
                    ctx.span_name, norm["routing_target"],
                    reason=norm.get("routing_reason", ""), span=ctx.span,
                )
                return
            # Fall through to legacy logic with ctx fields
            self._legacy_extract(
                ctx.span, ctx.span_name, ctx.raw_result, ctx.state,
            )
            return

        # Legacy path: (span, span_name, args, result, state)
        span, span_name, fn_args, result, state = args
        self._legacy_extract(span, span_name, result, state)

    @staticmethod
    def _legacy_extract(span, span_name, result, state):
        from ragwatch.instrumentation.helpers import record_routing

        if isinstance(result, dict):
            tc_list = _tool_calls_from_messages(result)
            if tc_list:
                names = [tc.get("name", "") for tc in tc_list]
                record_routing(span_name, "tools",
                               reason=f"calling: {names}", span=span)
            else:
                record_routing(span_name, "collect_answer",
                               reason="answer ready, no tool calls", span=span)

        if _is_command(result):
            goto = str(result.goto)
            reason = f"tokens≈{_rough_tokens(state)} → {goto}" if state else goto
            record_routing(span_name, goto, reason=reason, span=span)


class AgentCompletionExtractor:
    """Extract agent task completion metadata.

    Reads ``ctx.normalized["agent_answer"]`` and ``ctx.normalized["is_fallback"]``
    when available, otherwise falls back to framework-specific field reading.
    """

    name = "agent_completion"

    def extract(self, *args: Any, **kwargs: Any) -> None:
        from ragwatch.instrumentation.helpers import record_agent_completion  # lazy

        # Context-first path
        if len(args) == 1 and hasattr(args[0], "normalized"):
            ctx = args[0]
            norm = ctx.normalized or {}
            if "agent_answer" in norm:
                answer = norm["agent_answer"]
                is_fallback = norm.get("is_fallback", False)
                state = ctx.state
                record_agent_completion(
                    status="fallback" if is_fallback else "success",
                    iteration_count=state.get("iteration_count", 0) if state else 0,
                    tool_call_count=state.get("tool_call_count", 0) if state else 0,
                    question=state.get("question", "") if state else "",
                    question_index=state.get("question_index", 0) if state else 0,
                    answer_length=len(answer or ""),
                    is_fallback=is_fallback,
                    span=ctx.span,
                )
                return
            # Fall through to legacy logic
            self._legacy_extract(ctx.span, ctx.raw_result, ctx.state)
            return

        # Legacy path: (span, span_name, args, result, state)
        span, span_name, fn_args, result, state = args
        self._legacy_extract(span, result, state)

    @staticmethod
    def _legacy_extract(span, result, state):
        from ragwatch.instrumentation.helpers import record_agent_completion

        if not isinstance(result, dict):
            return

        agent_answers = result.get("agent_answers", [])
        final_answer = result.get("final_answer", "")
        answer = (
            agent_answers[-1].get("answer", final_answer)
            if agent_answers and isinstance(agent_answers[-1], dict)
            else final_answer
        )
        is_fallback = (
            answer == "Unable to generate an answer."
            or not bool((answer or "").strip())
        )
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


class QueryRewriteExtractor:
    """Extract query decomposition telemetry.

    Reads ``ctx.normalized["rewritten_questions"]`` when available,
    otherwise falls back to framework-specific field reading.
    """

    name = "query_rewrite"

    def extract(self, *args: Any, **kwargs: Any) -> None:
        from ragwatch.instrumentation.helpers import record_query_rewrite  # lazy

        # Context-first path
        if len(args) == 1 and hasattr(args[0], "normalized"):
            ctx = args[0]
            norm = ctx.normalized or {}
            if "rewritten_questions" in norm:
                record_query_rewrite(
                    norm.get("original_query", ""),
                    norm["rewritten_questions"],
                    norm.get("is_clear", False),
                    span=ctx.span,
                )
                return
            # Fall through to legacy
            self._legacy_extract(ctx.span, ctx.raw_result, ctx.state)
            return

        # Legacy path: (span, span_name, args, result, state)
        span, span_name, fn_args, result, state = args
        self._legacy_extract(span, result, state)

    @staticmethod
    def _legacy_extract(span, result, state):
        from ragwatch.instrumentation.helpers import record_query_rewrite

        if not isinstance(result, dict):
            return

        rewritten = result.get("rewrittenQuestions", [])
        is_clear = result.get("questionIsClear", False)
        original = result.get("originalQuery", "")
        if not original and state:
            msgs = state.get("messages", [])
            original = getattr(msgs[-1], "content", "") if msgs else ""
        record_query_rewrite(original, rewritten, is_clear, span=span)


class CompressionExtractor:
    """Extract context compression statistics.

    Reads ``ctx.normalized["compression_tokens_before"]`` etc. when available,
    otherwise falls back to framework-specific field reading.
    """

    name = "compression"

    def extract(self, *args: Any, **kwargs: Any) -> None:
        from ragwatch.instrumentation.helpers import record_context_compression  # lazy

        # Context-first path
        if len(args) == 1 and hasattr(args[0], "normalized"):
            ctx = args[0]
            norm = ctx.normalized or {}
            if "compression_tokens_before" in norm:
                record_context_compression(
                    tokens_before=norm["compression_tokens_before"],
                    tokens_after=norm.get("compression_tokens_after", 0),
                    queries_run=norm.get("queries_run") or None,
                    parents_retrieved=norm.get("parents_retrieved") or None,
                    span=ctx.span,
                )
                return
            # Fall through to legacy
            self._legacy_extract(ctx.span, ctx.raw_result, ctx.state)
            return

        # Legacy path: (span, span_name, args, result, state)
        span, span_name, fn_args, result, state = args
        self._legacy_extract(span, result, state)

    @staticmethod
    def _legacy_extract(span, result, state):
        from ragwatch.instrumentation.helpers import record_context_compression

        if not isinstance(result, dict) or state is None:
            return

        msgs = state.get("messages", [])
        old_summary = state.get("context_summary", "") or ""
        new_summary = result.get("context_summary", "") or ""
        tokens_before = (
            sum(len(getattr(m, "content", "") or "") for m in msgs) // 4
            + len(old_summary) // 4
        )
        tokens_after = len(new_summary) // 4
        retrieved_ids = state.get("retrieval_keys", set()) or set()
        parents = sorted(
            r.replace("parent::", "") for r in retrieved_ids if r.startswith("parent::")
        )
        queries = sorted(
            r.replace("search::", "") for r in retrieved_ids if r.startswith("search::")
        )
        record_context_compression(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            queries_run=queries or None,
            parents_retrieved=parents or None,
            span=span,
        )


# ---------------------------------------------------------------------------
# Default registry singleton
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: Optional[ExtractorRegistry] = None


def get_default_registry() -> ExtractorRegistry:
    """Return the default registry, creating it on first call.

    Pre-loaded with the five built-in extractors.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ExtractorRegistry()
        for ext_cls in (
            ToolCallsExtractor,
            RoutingExtractor,
            AgentCompletionExtractor,
            QueryRewriteExtractor,
            CompressionExtractor,
        ):
            _DEFAULT_REGISTRY.register(ext_cls())
    return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """Reset the default registry.  Intended for testing only."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = None
