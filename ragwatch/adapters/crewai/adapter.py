"""CrewAI framework adapter.

Provides a formal ``FrameworkAdapter`` implementation for CrewAI that
extracts state from keyword arguments and supplies minimal default
extractors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragwatch.instrumentation.extractors import TelemetryExtractor


class CrewAIAdapter:
    """Adapter for CrewAI workflows.

    State convention: CrewAI passes task data as positional args or kwargs.
    No dict-based state convention like LangGraph.
    """

    name = "crewai"

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
        """CrewAI has no built-in extractors beyond the core defaults."""
        return []

    def capabilities(self) -> set:
        """CrewAI supports agent completion tracking."""
        return {"agent_completion"}

    def normalize_result(
        self,
        raw_result: Any,
        state: Optional[dict],
    ) -> Optional[Dict[str, Any]]:
        """Translate CrewAI-style results into normalized semantic keys.

        Recognized shapes:

        - ``task_output`` or ``output`` → ``agent_answer``
        - ``status`` → ``is_fallback`` (True if status is ``"error"`` or empty)
        - ``tools_used`` → ``tool_calls`` (list of ``{"name": ...}`` dicts)
        """
        if not isinstance(raw_result, dict):
            # CrewAI TaskOutput objects may expose .dict() or attributes
            if hasattr(raw_result, "raw"):
                raw_result = {"task_output": raw_result.raw}
            elif hasattr(raw_result, "output"):
                raw_result = {"output": raw_result.output}
            else:
                return None

        norm: Dict[str, Any] = {}

        # Agent answer — use 'is not None' to preserve empty strings for fallback
        answer = raw_result.get("task_output")
        if answer is None:
            answer = raw_result.get("output")
        if answer is not None:
            norm["agent_answer"] = str(answer)

        # Fallback detection
        status = raw_result.get("status", "")
        if "agent_answer" in norm:
            norm["is_fallback"] = status in ("error", "") and not norm["agent_answer"]

        # Tool calls
        tools = raw_result.get("tools_used")
        if isinstance(tools, list) and tools:
            norm["tool_calls"] = [
                t if isinstance(t, dict) else {"name": str(t)} for t in tools
            ]

        return norm or None
