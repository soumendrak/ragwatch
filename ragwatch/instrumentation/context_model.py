"""Unified instrumentation context passed to all extension points.

``InstrumentationContext`` is built incrementally inside ``_make_wrapper``
and provides a single, consistent view of the current span lifecycle to
hooks, extractors, and transformers.

Fields that are not yet available (e.g. ``result`` during ``on_start``)
are set to their default sentinel values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from opentelemetry import trace as otel_trace

from ragwatch.core.span_kinds import SpanKind


@dataclass
class InstrumentationContext:
    """Context object threaded through all RAGWatch extension points.

    Built incrementally — ``result``, ``raw_result``, and ``exception``
    are populated as the decorated function executes.
    """

    span: otel_trace.Span
    span_name: str
    span_kind: SpanKind
    func_name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    adapter: Any = None            # Optional[FrameworkAdapter]
    state: Optional[dict] = None   # adapter.extract_state() result
    raw_result: Any = None         # before transformation
    result: Any = None             # after transformation
    exception: Optional[BaseException] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute through the policy-enforced writer.

        This is the **recommended** way for hooks, extractors, and
        transformers to write attributes — it guarantees that the active
        :class:`AttributePolicy` (truncation, redaction, naming checks)
        is applied automatically.
        """
        from ragwatch.instrumentation.attributes import safe_set_attribute

        safe_set_attribute(self.span, key, value)
