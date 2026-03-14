"""Attribute naming validation, truncation, and redaction policies.

Provides schema rules for custom span attributes so that enterprise users
get consistent, safe attribute naming and value handling across the SDK.

Usage::

    from ragwatch.instrumentation.attribute_policy import (
        AttributePolicy, validate_attribute_name,
    )

    # Validate naming conventions
    assert validate_attribute_name("ragwatch.custom.latency_ms") is True
    assert validate_attribute_name("") is False

    # Apply policy to values
    policy = AttributePolicy(max_value_bytes=2048)
    safe_value = policy.apply("my.attr", "some long string...")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Attribute name validation
# ---------------------------------------------------------------------------

# Dot-separated, lowercase alphanumeric + underscores per segment
_VALID_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*(\.\d+(\.[a-z][a-z0-9_]*)*)*$")
_MAX_KEY_LENGTH = 128


def validate_attribute_name(name: str) -> bool:
    """Check if *name* follows the dot-separated namespace convention.

    Rules:
    - Must be non-empty and ≤128 characters.
    - Must be dot-separated segments of lowercase alphanumeric + underscores.
    - Numeric segments are allowed for indexed attributes (e.g. ``chunk.0.score``).
    - First character of each non-numeric segment must be a letter.

    Args:
        name: The attribute name to validate.

    Returns:
        ``True`` if the name is valid.
    """
    if not name or len(name) > _MAX_KEY_LENGTH:
        return False
    return _VALID_NAME_PATTERN.match(name) is not None


# ---------------------------------------------------------------------------
# AttributePolicy
# ---------------------------------------------------------------------------

@dataclass
class AttributePolicy:
    """Controls truncation and redaction of span attribute values.

    Args:
        max_value_bytes: Maximum byte length for string attribute values.
            Values exceeding this are truncated with a ``...[truncated]``
            suffix.  Default: 4096.
        redact_patterns: List of regex patterns.  If an attribute **value**
            matches any pattern, it is replaced with ``[REDACTED]``.
            Patterns are applied to string values only.
        redact_keys: List of attribute name substrings.  If an attribute
            **name** contains any of these substrings, its value is replaced
            with ``[REDACTED]``.  Useful for blanket PII suppression
            (e.g. ``["password", "secret", "token"]``).
    """

    max_value_bytes: int = 4096
    max_list_length: int = 128
    redact_patterns: List[str] = field(default_factory=list)
    redact_keys: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._compiled_patterns = [re.compile(p) for p in self.redact_patterns]

    def apply(self, key: str, value: Any) -> Any:
        """Apply truncation, redaction, and collection-size rules.

        Args:
            key: The attribute name (used for key-based redaction).
            value: The raw attribute value.

        Returns:
            The sanitized value (possibly truncated, redacted, or capped).
        """
        # Key-based redaction
        if self.redact_keys:
            key_lower = key.lower()
            for rk in self.redact_keys:
                if rk.lower() in key_lower:
                    return "[REDACTED]"

        # List/tuple cardinality cap
        if isinstance(value, (list, tuple)):
            if len(value) > self.max_list_length:
                truncated_list = list(value[:self.max_list_length])
                return truncated_list
            return value

        # Only apply string-level rules to strings
        if not isinstance(value, str):
            return value

        # Pattern-based redaction
        for pattern in self._compiled_patterns:
            if pattern.search(value):
                return "[REDACTED]"

        # Truncation
        if len(value.encode("utf-8", errors="replace")) > self.max_value_bytes:
            # Truncate at character level (approximate) to stay under byte limit
            truncated = value[:self.max_value_bytes]
            while len(truncated.encode("utf-8", errors="replace")) > self.max_value_bytes:
                truncated = truncated[:-1]
            return truncated + "...[truncated]"

        return value
