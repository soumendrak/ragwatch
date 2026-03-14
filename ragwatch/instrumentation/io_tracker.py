"""Auto I/O tracking for decorated functions.

Captures function arguments as ``input.value`` and return value as
``output.value`` on the active span.  Values are JSON-serialized and
truncated at 4 KB.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace as otel_trace

from ragwatch.instrumentation.attributes import safe_set_attribute
from ragwatch.instrumentation.semconv import INPUT_VALUE, OUTPUT_VALUE

_MAX_IO_BYTES = 4096


def _safe_serialize(value: Any) -> str:
    """JSON-serialize *value*, falling back to ``repr`` on failure."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError, OverflowError):
        text = repr(value)
    if len(text) > _MAX_IO_BYTES:
        text = text[:_MAX_IO_BYTES] + "...[truncated]"
    return text


def track_input(span: otel_trace.Span, args: tuple, kwargs: dict) -> None:
    """Record function arguments on *span* as ``input.value``.

    Args:
        span: The active OTel span.
        args: Positional arguments passed to the function.
        kwargs: Keyword arguments passed to the function.
    """
    payload: dict[str, Any] = {}
    if args:
        payload["args"] = list(args)
    if kwargs:
        payload["kwargs"] = kwargs
    safe_set_attribute(span, INPUT_VALUE, _safe_serialize(payload))


def track_output(span: otel_trace.Span, result: Any) -> None:
    """Record the return value on *span* as ``output.value``.

    Args:
        span: The active OTel span.
        result: The return value of the decorated function.
    """
    safe_set_attribute(span, OUTPUT_VALUE, _safe_serialize(result))
