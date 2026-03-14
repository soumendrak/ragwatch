"""CrewAI framework adapter.

Provides a formal ``FrameworkAdapter`` implementation for CrewAI that
extracts state from keyword arguments and supplies minimal default
extractors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragwatch.adapters.base import FrameworkAdapter
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
        """CrewAI supports task completion tracking."""
        return {"task_completion"}
