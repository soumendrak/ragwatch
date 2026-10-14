"""LlamaIndex framework adapter.

Provides a ``FrameworkAdapter`` implementation for LlamaIndex that extracts
state from call arguments and normalizes LlamaIndex response objects
(``Response``, ``AgentChatResponse``, ``CompletionResponse``) into
RAGWatch's semantic keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragwatch.instrumentation.extractors import TelemetryExtractor


class LlamaIndexAdapter:
    """Adapter for LlamaIndex query, chat and ingestion functions.

    State convention: LlamaIndex entry points take a query string or a
    ``QueryBundle``; keyword arguments (or the first dict argument) are
    used as state.
    """

    name = "llamaindex"

    def extract_state(
        self,
        args: tuple,
        kwargs: dict,
    ) -> Optional[dict]:
        """Return kwargs as state if non-empty, else first dict in args."""
        if kwargs:
            return dict(kwargs)
        return next((a for a in args if isinstance(a, dict)), None)

    def default_extractors(self) -> List[TelemetryExtractor]:
        """LlamaIndex has no built-in extractors beyond the core defaults."""
        return []

    def capabilities(self) -> set:
        """LlamaIndex supports agent completion and tool call tracking."""
        return {"agent_completion", "tool_calls"}

    def normalize_result(
        self,
        raw_result: Any,
        state: Optional[dict],
    ) -> Optional[Dict[str, Any]]:
        """Translate LlamaIndex-style results into normalized semantic keys.

        Recognized shapes:

        - ``.response`` (``Response`` / ``AgentChatResponse``) or ``.text``
          (``CompletionResponse``) → ``agent_answer``
        - ``.sources`` (``ToolOutput`` list) → ``tool_calls``
        - dicts with ``response`` / ``tool_calls`` keys
        """
        if isinstance(raw_result, dict):
            answer = raw_result.get("response")
            tools = raw_result.get("tool_calls")
        else:
            answer = getattr(raw_result, "response", None)
            if answer is None:
                answer = getattr(raw_result, "text", None)
            tools = getattr(raw_result, "sources", None)

        norm: Dict[str, Any] = {}
        if answer is not None:
            norm["agent_answer"] = str(answer)
            norm["is_fallback"] = not norm["agent_answer"]
        if isinstance(tools, list) and tools:
            norm["tool_calls"] = [_tool_call(t) for t in tools]
        return norm or None


def _tool_call(tool: Any) -> Dict[str, Any]:
    if isinstance(tool, dict):
        return tool
    name = getattr(tool, "tool_name", None)
    return {"name": str(name if name is not None else tool)}
