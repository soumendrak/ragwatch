"""RAGWatch configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from opentelemetry.sdk.trace.export import SpanExporter


@dataclass(frozen=True)
class RAGWatchConfig:
    """Configuration for RAGWatch SDK.

    Args:
        service_name: Name of the service to appear in OTel resource attributes.
        exporter: Optional span exporter. Users bring their own OTel backend.
            If ``None``, spans are created but not exported (useful for testing).
        max_embedding_dims: Maximum embedding dimensions stored in context.
            Defaults to 512.
    """

    service_name: str = "ragwatch-service"
    exporter: Optional[SpanExporter] = None
    max_embedding_dims: int = 512
