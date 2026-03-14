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
