"""RAGWatch — OpenTelemetry-native RAG observability with semantic quality scores.

Public API::

    from ragwatch import (
        configure, trace, record_feedback, RAGWatchConfig, SpanKind,
        # Rich telemetry helpers
        record_chunks, record_agent_completion, record_routing,
        record_tool_calls, record_context_compression, record_query_rewrite,
    )
"""

from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.runtime import RAGWatchRuntime
from ragwatch.core.span_kinds import SpanKind
from ragwatch.core.tracer import configure_tracer as _configure_tracer
from ragwatch.instrumentation.decorators import trace
from ragwatch.instrumentation.evaluators import chunk_relevance_score, record_feedback
from ragwatch.instrumentation.extractors import (
    ExtractorRegistry,
    TelemetryExtractor,
    get_default_registry,
)
from ragwatch.adapters.base import FrameworkAdapter, register_adapter
from ragwatch.instrumentation.attribute_policy import (
    AttributePolicy,
    validate_attribute_name,
)
from ragwatch.instrumentation.attributes import safe_set_attribute, safe_set_attributes
from ragwatch.instrumentation.span_hooks import SpanHook, register_global_hook, clear_global_hooks
from ragwatch.instrumentation.result_transformers import (
    ResultTransformer,
    ResultTransformerRegistry,
    get_default_transformer_registry,
    reset_default_transformer_registry,
)
from ragwatch.instrumentation.token_usage import (
    TokenExtractor,
    register_token_extractor,
    clear_token_extractors,
)
from ragwatch.instrumentation.context_model import InstrumentationContext
from ragwatch.instrumentation.helpers import (
    record_agent_completion,
    record_chunks,
    record_context_compression,
    record_query_rewrite,
    record_routing,
    record_tool_calls,
)


_ACTIVE_CONFIG: RAGWatchConfig | None = None


def get_active_config() -> RAGWatchConfig | None:
    """Return the config passed to the last :func:`configure` call, or ``None``."""
    return _ACTIVE_CONFIG


def configure(config: RAGWatchConfig | None = None, **kwargs) -> None:
    """Initialize the RAGWatch SDK.

    This function is **idempotent-safe**: calling it multiple times resets
    all global registries (hooks, extractors, adapters, transformers, token
    extractors) before re-registering, so repeated calls never accumulate
    duplicates.

    Args:
        config: A :class:`RAGWatchConfig` instance.  If ``None``, a default
            config is created from *kwargs*.
        **kwargs: Forwarded to :class:`RAGWatchConfig` when *config* is
            ``None``.
    """
    global _ACTIVE_CONFIG
    if config is None:
        config = RAGWatchConfig(**kwargs)

    # Reset all global registries to prevent accumulation on repeated calls
    from ragwatch.instrumentation.extractors import reset_default_registry
    from ragwatch.adapters.base import clear_adapters

    clear_global_hooks()
    reset_default_registry()
    clear_adapters()
    clear_token_extractors()
    reset_default_transformer_registry()

    _ACTIVE_CONFIG = config
    _configure_tracer(config)

    # Re-initialize default extractor registry (built-ins are re-created
    # by get_default_registry() on next access after reset)
    registry = get_default_registry()

    # Register user-supplied telemetry extractors
    if config.custom_extractors:
        for ext in config.custom_extractors:
            registry.register(ext)

    # Register global span hooks
    if config.global_span_hooks:
        for hook in config.global_span_hooks:
            register_global_hook(hook)

    # Register framework adapters + their default extractors
    if config.adapters:
        for adapter in config.adapters:
            register_adapter(adapter)
            for ext in adapter.default_extractors():
                registry.register(ext)

    # Register custom result transformers
    if config.custom_transformers:
        tr_registry = get_default_transformer_registry()
        for transformer in config.custom_transformers:
            tr_registry.register(transformer)

    # Register custom token extractors
    if config.custom_token_extractors:
        for ext in config.custom_token_extractors:
            register_token_extractor(ext)


__all__ = [
    # --- Stable API -----------------------------------------------------------
    # Core
    "configure",
    "trace",
    "RAGWatchConfig",
    "SpanKind",
    "InstrumentationContext",
    "RAGWatchRuntime",
    "get_active_config",
    # Quality scores
    "record_feedback",
    "chunk_relevance_score",
    # Extension protocols (context-first canonical, legacy supported)
    "TelemetryExtractor",
    "ExtractorRegistry",
    "get_default_registry",
    "SpanHook",
    "register_global_hook",
    "FrameworkAdapter",
    "register_adapter",
    "ResultTransformer",
    "ResultTransformerRegistry",
    "get_default_transformer_registry",
    "TokenExtractor",
    "register_token_extractor",
    # Attribute policy & writer
    "AttributePolicy",
    "validate_attribute_name",
    "safe_set_attribute",
    "safe_set_attributes",
    # Rich telemetry helpers
    "record_chunks",
    "record_agent_completion",
    "record_routing",
    "record_tool_calls",
    "record_context_compression",
    "record_query_rewrite",
]
