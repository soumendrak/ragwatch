"""Base adapter contract for framework integrations.

Provides a ``FrameworkAdapter`` protocol that formalises how framework-specific
adapters extract state, supply default extractors, and declare their identity.
This replaces duck-typing with explicit contracts.

Usage — implementing a new adapter::

    from ragwatch.adapters.base import FrameworkAdapter

    class MyFrameworkAdapter:
        name = "myframework"

        def extract_state(self, args, kwargs):
            return args[0] if args and isinstance(args[0], dict) else None

        def default_extractors(self):
            return [MyCustomExtractor()]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ragwatch.instrumentation.extractors import TelemetryExtractor


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Contract for framework-specific adapters.

    Implementors declare how to extract state from function arguments and
    which telemetry extractors are relevant by default.

    **Mandatory members** (required for ``isinstance`` / ``runtime_checkable``):

    - ``name: str`` — unique adapter identifier (e.g. ``"langgraph"``).
    - ``extract_state(args, kwargs) -> Optional[dict]`` — pull the
      framework's state dict from decorated function arguments.
    - ``default_extractors() -> List[TelemetryExtractor]`` — extractors
      registered when the adapter is activated via ``configure(adapters=[...])``.

    **Optional methods** (safe to omit; detected via ``getattr`` at runtime):

    - ``capabilities() -> set`` — declares supported telemetry features
      (e.g. ``{"routing", "tool_calls"}``).  Accessed via
      :func:`get_capabilities`.
    - ``normalize_result(raw_result, state) -> Optional[dict]`` — translates
      framework-specific result shapes into well-known semantic keys.
      When implemented, the return dict is stored as ``ctx.normalized``
      and preferred by built-in extractors over raw result inspection.

      Expected semantic keys (all optional):

      - ``tool_calls``: ``list[dict]`` — LLM tool-call decisions
      - ``routing_target``: ``str`` — destination node name
      - ``routing_reason``: ``str`` — human-readable reason
      - ``agent_answer``: ``str`` — final answer string
      - ``is_fallback``: ``bool`` — whether the answer is a fallback
      - ``rewritten_questions``: ``list[str]`` — decomposed queries
      - ``is_clear``: ``bool`` — query clarity flag
      - ``original_query``: ``str`` — original user query
      - ``compression_tokens_before``: ``int``
      - ``compression_tokens_after``: ``int``
      - ``context_summary``: ``str``
      - ``queries_run``: ``list[str]`` — search queries executed
      - ``parents_retrieved``: ``list[str]`` — parent document IDs retrieved

    Optional methods are **not** included in the Protocol to preserve
    ``@runtime_checkable`` compatibility with minimal adapters.
    """

    name: str

    def extract_state(
        self,
        args: tuple,
        kwargs: dict,
    ) -> Optional[dict]:
        """Extract the framework's state dict from function arguments.

        Args:
            args: Positional arguments passed to the decorated function.
            kwargs: Keyword arguments passed to the decorated function.

        Returns:
            The state dict if found, or ``None``.
        """
        ...

    def default_extractors(self) -> List[TelemetryExtractor]:
        """Return the default telemetry extractors for this framework.

        These are registered in the extractor registry when the adapter is
        activated via ``configure(adapters=[...])``.

        Returns:
            List of extractor instances.
        """
        ...


def get_capabilities(adapter: FrameworkAdapter) -> set:
    """Return the capabilities of an adapter, or empty set if not declared.

    Adapters may optionally implement a ``capabilities()`` method that
    returns a set of strings (e.g. ``{"routing", "tool_calls"}``).
    This helper safely falls back to an empty set.
    """
    method = getattr(adapter, "capabilities", None)
    if callable(method):
        return method()
    return set()


def normalize_result(
    adapter: Optional[FrameworkAdapter],
    raw_result: Any,
    state: Optional[dict],
) -> Optional[dict]:
    """Normalize a framework-specific result into semantic keys.

    Adapters may optionally implement ``normalize_result(raw_result, state)``
    returning a dict with well-known semantic keys such as:

    - ``tool_calls``: list of tool call dicts
    - ``routing_target``: destination node name
    - ``routing_reason``: human-readable routing reason
    - ``agent_answer``: final answer string
    - ``is_fallback``: bool
    - ``rewritten_questions``: list of rewritten queries
    - ``is_clear``: bool (query clarity)
    - ``original_query``: str
    - ``compression_tokens_before``: int
    - ``compression_tokens_after``: int
    - ``context_summary``: str

    If the adapter does not implement ``normalize_result``, returns ``None``
    and extractors fall back to framework-specific field reading.
    """
    if adapter is None:
        return None
    method = getattr(adapter, "normalize_result", None)
    if callable(method):
        return method(raw_result, state)
    return None


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTERS: Dict[str, FrameworkAdapter] = {}


def register_adapter(adapter: FrameworkAdapter) -> None:
    """Register a framework adapter globally."""
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
    """Return the adapter registered under *name*, or ``None``."""
    return _ADAPTERS.get(name)


def get_all_adapters() -> Dict[str, FrameworkAdapter]:
    """Return all registered adapters."""
    return dict(_ADAPTERS)


def clear_adapters() -> None:
    """Remove all registered adapters.  Intended for testing only."""
    _ADAPTERS.clear()
