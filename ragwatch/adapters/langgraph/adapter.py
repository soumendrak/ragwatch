"""LangGraph framework adapter.

Provides a formal ``FrameworkAdapter`` implementation for LangGraph that
extracts state from the first positional dict argument and supplies the
standard LangGraph telemetry extractors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragwatch.adapters.base import FrameworkAdapter
from ragwatch.instrumentation.extractors import (
    AgentCompletionExtractor,
    CompressionExtractor,
    QueryRewriteExtractor,
    RoutingExtractor,
    TelemetryExtractor,
    ToolCallsExtractor,
)


class LangGraphAdapter:
    """Adapter for LangGraph workflows.

    State convention: the first positional ``dict`` argument is the LangGraph
    state dict.
    """

    name = "langgraph"

    def extract_state(
        self,
        args: tuple,
        kwargs: dict,
    ) -> Optional[dict]:
        """Return the first dict in *args* (LangGraph state convention)."""
        return next((a for a in args if isinstance(a, dict)), None)

    def default_extractors(self) -> List[TelemetryExtractor]:
        """Return all built-in extractors relevant to LangGraph."""
        return [
            ToolCallsExtractor(),
            RoutingExtractor(),
            AgentCompletionExtractor(),
            QueryRewriteExtractor(),
            CompressionExtractor(),
        ]

    def capabilities(self) -> set:
        """LangGraph supports all core telemetry capabilities."""
        return {
            "routing", "tool_calls", "agent_completion",
            "query_rewrite", "compression", "message_history",
        }

    def normalize_result(
        self, raw_result: Any, state: Optional[dict],
    ) -> Optional[Dict[str, Any]]:
        """Translate LangGraph-specific result shapes into semantic keys.

        Returns a dict with well-known keys that extractors can consume
        without knowing LangGraph internals.  Returns ``None`` if the
        result is not a dict.
        """
        if not isinstance(raw_result, dict):
            # Check for Command objects (routing via goto)
            if hasattr(raw_result, "goto") and hasattr(raw_result, "update"):
                tokens = self._rough_tokens(state)
                goto = str(raw_result.goto)
                return {
                    "routing_target": goto,
                    "routing_reason": f"tokens≈{tokens} → {goto}" if state else goto,
                }
            return None

        norm: Dict[str, Any] = {}

        # Tool calls from messages
        tc = self._extract_tool_calls(raw_result)
        if tc:
            norm["tool_calls"] = tc

        # Agent completion
        agent_answers = raw_result.get("agent_answers", [])
        final_answer = raw_result.get("final_answer", "")
        if agent_answers or final_answer:
            answer = (
                agent_answers[-1].get("answer", final_answer)
                if agent_answers and isinstance(agent_answers[-1], dict)
                else final_answer
            )
            is_fallback = (
                answer == "Unable to generate an answer."
                or not bool((answer or "").strip())
            )
            norm["agent_answer"] = answer
            norm["is_fallback"] = is_fallback

        # Query rewrite
        rewritten = raw_result.get("rewrittenQuestions", [])
        if rewritten:
            norm["rewritten_questions"] = rewritten
            norm["is_clear"] = raw_result.get("questionIsClear", False)
            original = raw_result.get("originalQuery", "")
            if not original and state:
                msgs = state.get("messages", [])
                original = getattr(msgs[-1], "content", "") if msgs else ""
            norm["original_query"] = original

        # Compression
        new_summary = raw_result.get("context_summary", "")
        if new_summary and state is not None:
            msgs = state.get("messages", [])
            old_summary = state.get("context_summary", "") or ""
            tokens_before = (
                sum(len(getattr(m, "content", "") or "") for m in msgs) // 4
                + len(old_summary) // 4
            )
            norm["compression_tokens_before"] = tokens_before
            norm["compression_tokens_after"] = len(new_summary) // 4
            norm["context_summary"] = new_summary

            retrieved_ids = state.get("retrieval_keys", set()) or set()
            norm["parents_retrieved"] = sorted(
                r.replace("parent::", "") for r in retrieved_ids if r.startswith("parent::")
            )
            norm["queries_run"] = sorted(
                r.replace("search::", "") for r in retrieved_ids if r.startswith("search::")
            )

        return norm if norm else None

    @staticmethod
    def _extract_tool_calls(result: dict) -> list:
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "tool_calls"):
                return msg.tool_calls or []
        return []

    @staticmethod
    def _rough_tokens(state: Optional[dict]) -> int:
        if state is None:
            return 0
        msgs = state.get("messages", [])
        summary = state.get("context_summary", "") or ""
        return sum(len(getattr(m, "content", "") or "") for m in msgs) // 4 + len(summary) // 4
