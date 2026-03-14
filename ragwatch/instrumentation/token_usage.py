"""Token-usage extraction from LLM response objects.

Extracted from ``decorators.py`` to keep span-lifecycle orchestration
separate from token-counting logic.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional, Protocol, runtime_checkable

from ragwatch.instrumentation.attributes import safe_set_attribute


def _accepts_context_token(method: Any) -> bool:
    """Return True if *method* accepts a single ``context`` positional arg."""
    try:
        sig = inspect.signature(method)
        params = [
            p for p in sig.parameters.values()
            if p.name != "self" and p.default is inspect.Parameter.empty
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(params) == 1 and params[0].name == "context"
    except (ValueError, TypeError):
        return False

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

    Two signatures are supported — the dispatch auto-detects which one
    your extractor implements.

    **Canonical (context-first — recommended):**

    .. code-block:: python

        class MyTokenExtractor:
            def extract(self, context: InstrumentationContext) -> None:
                usage = getattr(context.raw_result, "usage", None)
                if usage:
                    context.set_attribute("llm.token_count.prompt", usage["input"])

    **Legacy (still supported):**

    .. code-block:: python

        class MyTokenExtractor:
            def extract(self, span, result) -> None:
                from ragwatch.instrumentation.attributes import safe_set_attribute
                usage = getattr(result, "usage", None)
                if usage:
                    safe_set_attribute(span, "llm.token_count.prompt", usage["input"])
    """

    def extract(self, *args: Any, **kwargs: Any) -> None:
        """Extract token usage — see class docstring for supported signatures."""
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
        if context is not None and _accepts_context_token(ext.extract):
            ext.extract(context)
        else:
            ext.extract(span, result)
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
            prompt     += usage.get("input_tokens", 0)
            completion += usage.get("output_tokens", 0)
            total      += usage.get("total_tokens", 0)
    if found:
        safe_set_attribute(span, LLM_TOKEN_COUNT_PROMPT, prompt)
        safe_set_attribute(span, LLM_TOKEN_COUNT_COMPLETION, completion)
        safe_set_attribute(span, LLM_TOKEN_COUNT_TOTAL, total)
