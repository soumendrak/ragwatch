"""Token-usage extraction from LLM response objects.

Extracted from ``decorators.py`` to keep span-lifecycle orchestration
separate from token-counting logic.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from ragwatch.instrumentation.attributes import safe_set_attribute


from ragwatch.instrumentation.semconv import (
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
)


# ---------------------------------------------------------------------------
# Protocol + Registry
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenExtractor(Protocol):
    """Contract for pluggable token-usage extractors.

    Implementors scan a result object and record token counts on the span.

    Token extractors receive the current ``InstrumentationContext``.

    .. code-block:: python

        class MyTokenExtractor:
            def extract(self, context: InstrumentationContext) -> None:
                usage = getattr(context.raw_result, "usage", None)
                if usage:
                    context.set_attribute("llm.token_count.prompt", usage["input"])

    """

    def extract(self, context: Any) -> None:
        """Extract token usage from an instrumentation context."""
        ...


_CUSTOM_TOKEN_EXTRACTORS: List[TokenExtractor] = []


def register_token_extractor(extractor: TokenExtractor) -> None:
    """Register a custom token extractor that runs before the built-in one."""
    _CUSTOM_TOKEN_EXTRACTORS.append(extractor)


def get_token_extractors() -> List[TokenExtractor]:
    return list(_CUSTOM_TOKEN_EXTRACTORS)


def clear_token_extractors() -> None:
    """Remove all custom token extractors.  Intended for testing only."""
    _CUSTOM_TOKEN_EXTRACTORS.clear()


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_token_usage(span: Any, result: Any, *, context: Any = None) -> None:
    """Run custom token extractors first, then the built-in scanner."""
    if not span.is_recording():
        return
    for ext in _CUSTOM_TOKEN_EXTRACTORS:
        if context is None:
            raise ValueError("custom token extractors require context")
        ext.extract(context)
    _builtin_extract_token_usage(span, result)


def _builtin_extract_token_usage(span: Any, result: Any) -> None:
    """Scan *result* for ``usage_metadata`` dicts and record token counts."""
    if not span.is_recording():
        return
    candidates: list = []
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                candidates.extend(v)
            else:
                candidates.append(v)
    elif isinstance(result, list):
        candidates = result
    else:
        candidates = [result]

    prompt = completion = total = 0
    found = False
    for obj in candidates:
        usage = getattr(obj, "usage_metadata", None)
        if isinstance(usage, dict):
            found = True
            prompt += usage.get("input_tokens", 0)
            completion += usage.get("output_tokens", 0)
            total += usage.get("total_tokens", 0)
    if found:
        safe_set_attribute(span, LLM_TOKEN_COUNT_PROMPT, prompt)
        safe_set_attribute(span, LLM_TOKEN_COUNT_COMPLETION, completion)
        safe_set_attribute(span, LLM_TOKEN_COUNT_TOTAL, total)
