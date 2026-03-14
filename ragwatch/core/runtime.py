"""RAGWatchRuntime — scoped object that bundles SDK state.

Reduces reliance on module-level globals by providing a single object
that holds the active config, extractor registry, adapter registry,
and attribute policy.

Usage::

    runtime = RAGWatchRuntime.current()
    runtime.config          # RAGWatchConfig
    runtime.strict_mode     # bool
    runtime.attribute_policy  # AttributePolicy | None
    runtime.auto_track_io   # bool
"""

from __future__ import annotations

from typing import Any, Optional


class RAGWatchRuntime:
    """Scoped accessor for the active RAGWatch SDK state.

    This is a **read-only view** — mutating the runtime is done via
    :func:`ragwatch.configure`.  The runtime simply reads the global
    state that ``configure()`` sets.
    """

    @staticmethod
    def current() -> RAGWatchRuntime:
        """Return a runtime bound to the current global config."""
        return RAGWatchRuntime()

    @property
    def config(self) -> Any:
        """The active :class:`RAGWatchConfig`, or ``None``."""
        from ragwatch import get_active_config
        return get_active_config()

    @property
    def strict_mode(self) -> bool:
        cfg = self.config
        return cfg.strict_mode if cfg is not None else False

    @property
    def attribute_policy(self) -> Any:
        cfg = self.config
        return cfg.attribute_policy if cfg is not None else None

    @property
    def auto_track_io(self) -> bool:
        cfg = self.config
        return cfg.global_auto_track_io if cfg is not None else True

    @property
    def extractor_registry(self) -> Any:
        from ragwatch.instrumentation.extractors import get_default_registry
        return get_default_registry()

    @property
    def adapter_registry(self) -> dict:
        from ragwatch.adapters.base import get_all_adapters
        return get_all_adapters()

    @property
    def transformer_registry(self) -> Any:
        from ragwatch.instrumentation.result_transformers import (
            get_default_transformer_registry,
        )
        return get_default_transformer_registry()

    @property
    def token_extractor_registry(self) -> list:
        from ragwatch.instrumentation.token_usage import get_token_extractors
        return get_token_extractors()

    def __repr__(self) -> str:
        cfg = self.config
        name = cfg.service_name if cfg is not None else "<unconfigured>"
        return f"RAGWatchRuntime(service={name!r})"
