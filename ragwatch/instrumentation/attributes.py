"""Centralized attribute writer with policy enforcement.

All internal SDK code should use ``safe_set_attribute`` /
``safe_set_attributes`` instead of calling ``span.set_attribute()``
directly.  This guarantees that attribute naming validation, value
truncation, and redaction policies are applied consistently.

Usage::

    from ragwatch.instrumentation.attributes import safe_set_attribute

    safe_set_attribute(span, "my.custom.key", "some value")
"""

from __future__ import annotations

import logging
import re
import weakref
from typing import Any, Dict, Optional, Set

from opentelemetry import trace as otel_trace

_logger = logging.getLogger(__name__)

# Pattern to detect indexed attribute keys like retrieval.chunk.42.content
_INDEXED_ATTR_RE = re.compile(r"^(.+)\.(\d+)\.")

# WeakKeyDict tracking the set of (prefix, index) pairs written per span.
# Automatically cleaned up when the span is garbage-collected.
_span_indexed_counts: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_active_policy():
    """Return the active AttributePolicy from the SDK config, or None."""
    try:
        from ragwatch import get_active_config
        cfg = get_active_config()
        return cfg.attribute_policy if cfg is not None else None
    except Exception:
        return None


def _is_strict_mode() -> bool:
    """Check if the SDK is in strict mode."""
    try:
        from ragwatch import get_active_config
        cfg = get_active_config()
        return cfg.strict_mode if cfg is not None else False
    except Exception:
        return False


def safe_set_attribute(
    span: otel_trace.Span,
    key: str,
    value: Any,
    *,
    policy: Any = None,
) -> None:
    """Set a span attribute with optional policy enforcement.

    Args:
        span: The OTel span.
        key: Attribute name.
        value: Attribute value.
        policy: An :class:`AttributePolicy` instance.  If ``None``,
            the globally configured policy (from :func:`configure`) is used.
            If no policy is configured anywhere, the value is set as-is.
    """
    if not span.is_recording():
        return

    resolved_policy = policy or _get_active_policy()

    if resolved_policy is not None:
        from ragwatch.instrumentation.attribute_policy import validate_attribute_name

        if not validate_attribute_name(key):
            if _is_strict_mode():
                _logger.warning("Invalid attribute name (strict, skipped): %r", key)
                span.add_event("ragwatch.invalid_attribute", {"key": key})
                return
            _logger.warning("Invalid attribute name: %r", key)

        # Indexed attribute explosion guard
        m = _INDEXED_ATTR_RE.match(key)
        if m is not None:
            prefix = m.group(1)
            idx = int(m.group(2))
            if idx >= resolved_policy.max_indexed_attributes:
                _logger.debug(
                    "Indexed attribute %r exceeds max_indexed_attributes=%d, skipped",
                    key, resolved_policy.max_indexed_attributes,
                )
                # Record event only once per prefix to avoid spam
                seen: Set = _span_indexed_counts.setdefault(span, set())
                if prefix not in seen:
                    seen.add(prefix)
                    span.add_event("ragwatch.indexed_attr_limit", {
                        "prefix": prefix,
                        "limit": resolved_policy.max_indexed_attributes,
                    })
                return

        value = resolved_policy.apply(key, value)

    span.set_attribute(key, value)


def safe_set_attributes(
    span: otel_trace.Span,
    attrs: Dict[str, Any],
    *,
    policy: Any = None,
) -> None:
    """Set multiple span attributes with optional policy enforcement.

    Args:
        span: The OTel span.
        attrs: Mapping of attribute names to values.
        policy: An :class:`AttributePolicy` instance (or ``None`` for global).
    """
    for key, value in attrs.items():
        safe_set_attribute(span, key, value, policy=policy)
