"""RAGWatch — OpenTelemetry-native RAG observability with semantic quality scores.

Public API::

    from ragwatch import configure, trace, record_feedback, RAGWatchConfig, SpanKind
"""

from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.span_kinds import SpanKind
from ragwatch.core.tracer import configure_tracer as _configure_tracer
from ragwatch.instrumentation.decorators import trace
from ragwatch.instrumentation.evaluators import chunk_relevance_score, record_feedback


def configure(config: RAGWatchConfig | None = None, **kwargs) -> None:
    """Initialize the RAGWatch SDK.

    Args:
        config: A :class:`RAGWatchConfig` instance.  If ``None``, a default
            config is created from *kwargs*.
        **kwargs: Forwarded to :class:`RAGWatchConfig` when *config* is
            ``None``.
    """
    if config is None:
        config = RAGWatchConfig(**kwargs)
    _configure_tracer(config)


__all__ = [
    "configure",
    "trace",
    "record_feedback",
    "chunk_relevance_score",
    "RAGWatchConfig",
    "SpanKind",
]
