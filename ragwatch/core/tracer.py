"""TracerProvider setup and singleton management."""

from __future__ import annotations

from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from ragwatch.core.config import RAGWatchConfig
from ragwatch.instrumentation.semconv import RAGWATCH_SDK_VERSION

_TRACER_PROVIDER: Optional[TracerProvider] = None
_SDK_VERSION = "0.1.2"
_INSTRUMENTATION_NAME = "ragwatch"


def configure_tracer(
    config: RAGWatchConfig,
    *,
    _force_flush: bool = False,
) -> TracerProvider:
    """Create and register a ``TracerProvider``.

    Args:
        config: RAGWatch configuration.
        _force_flush: If ``True``, use ``SimpleSpanProcessor`` for immediate
            export (useful for tests).  Defaults to ``False``
            (``BatchSpanProcessor``).

    Returns:
        The configured ``TracerProvider``.
    """
    global _TRACER_PROVIDER

    resource = Resource.create(
        {
            "service.name": config.service_name,
            RAGWATCH_SDK_VERSION: _SDK_VERSION,
        }
    )

    provider = TracerProvider(resource=resource)

    # Collect all exporters: prefer the new list API, fall back to legacy single
    all_exporters = list(config.exporters) or (
        [config.exporter] if config.exporter is not None else []
    )
    for exp in all_exporters:
        processor = SimpleSpanProcessor(exp) if _force_flush else BatchSpanProcessor(exp)
        provider.add_span_processor(processor)

    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    _TRACER_PROVIDER = provider
    return provider


def get_tracer() -> trace.Tracer:
    """Return a ``Tracer`` scoped to the RAGWatch instrumentation library.

    If :func:`configure_tracer` has not been called yet, the global
    no-op tracer is returned.
    """
    provider = _TRACER_PROVIDER or trace.get_tracer_provider()
    return provider.get_tracer(_INSTRUMENTATION_NAME, _SDK_VERSION)


def get_tracer_provider() -> Optional[TracerProvider]:
    """Return the current RAGWatch ``TracerProvider``, or ``None``."""
    return _TRACER_PROVIDER


def reset_tracer_provider() -> None:
    """Reset the global tracer provider.  Intended for testing only."""
    global _TRACER_PROVIDER
    if _TRACER_PROVIDER is not None:
        _TRACER_PROVIDER.shutdown()
    _TRACER_PROVIDER = None
