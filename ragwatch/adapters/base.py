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
