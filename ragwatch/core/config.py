"""RAGWatch configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from opentelemetry.sdk.trace.export import SpanExporter


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
