"""RAGWatch configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

from opentelemetry.sdk.trace.export import SpanExporter

if TYPE_CHECKING:
    from ragwatch.adapters.base import FrameworkAdapter
    from ragwatch.instrumentation.attribute_policy import AttributePolicy
    from ragwatch.instrumentation.extractors import TelemetryExtractor
    from ragwatch.instrumentation.span_hooks import SpanHook


@dataclass(frozen=True)
class RAGWatchConfig:
    """Configuration for RAGWatch SDK.

    Args:
        service_name: Name of the service to appear in OTel resource attributes.
        exporters: List of span exporters — one BatchSpanProcessor is created
            per exporter, so each backend (Jaeger, Phoenix, …) has its own
            independent buffer and retry queue.
        exporter: Single span exporter (kept for backwards compatibility).
            Ignored when ``exporters`` is non-empty.
        max_embedding_dims: Maximum embedding dimensions stored in context.
            Defaults to 512.
        custom_extractors: Additional :class:`TelemetryExtractor` instances
            to register in the default extractor registry at configure-time.
        global_span_hooks: :class:`SpanHook` instances that run on every
            traced span (``on_start`` / ``on_end``).
        adapters: :class:`FrameworkAdapter` instances to register at
            configure-time.  Each adapter's default extractors are also
            registered in the extractor registry.
        attribute_policy: An :class:`AttributePolicy` controlling truncation
            and redaction of custom span attribute values.  ``None`` means
            no policy enforcement (default).
        strict_mode: When ``True``, exceptions from hooks, extractors, and
            result transformers are re-raised instead of swallowed.  Useful
            for development and testing.  Default: ``False``.
        global_auto_track_io: When ``False``, disables auto I/O tracking
            globally (overrides per-decorator ``auto_track_io=True``).
            Default: ``True``.

    Example — multiple backends::

        ragwatch.configure(
            RAGWatchConfig(
                service_name="my-rag-app",
                exporters=[
                    OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"),
                    OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"),
                ],
            )
        )
    """

    service_name: str = "ragwatch-service"
    exporters: List[SpanExporter] = field(default_factory=list)
    exporter: Optional[SpanExporter] = None   # legacy single-exporter API
    max_embedding_dims: int = 512
    custom_extractors: List[Any] = field(default_factory=list)  # List[TelemetryExtractor]
    global_span_hooks: List[Any] = field(default_factory=list)  # List[SpanHook]
    adapters: List[Any] = field(default_factory=list)  # List[FrameworkAdapter]
    custom_transformers: List[Any] = field(default_factory=list)  # List[ResultTransformer]
    custom_token_extractors: List[Any] = field(default_factory=list)  # List[TokenExtractor]
    attribute_policy: Any = None  # Optional[AttributePolicy]
    strict_mode: bool = False
    global_auto_track_io: bool = True
