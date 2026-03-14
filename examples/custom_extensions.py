"""RAGWatch Custom Extensions — attribute contracts & extension points.

This example demonstrates all v0.3 extension points:

1. Custom TelemetryExtractor — pluggable telemetry extraction
2. Custom SpanHook — lifecycle enrichment (on_start, on_end, on_error)
3. Custom ResultTransformer — pluggable result transformation per span kind
4. Custom TokenExtractor — pluggable token usage extraction
5. AttributePolicy — truncation and redaction
6. InstrumentationContext — rich context in hooks
7. RAGWatchRuntime — scoped access to SDK state
"""

from __future__ import annotations

import ragwatch
from ragwatch import (
    AttributePolicy,
    RAGWatchConfig,
    RAGWatchRuntime,
    SpanKind,
    configure,
    get_default_registry,
    safe_set_attribute,
    trace,
)
from ragwatch.instrumentation.context_model import InstrumentationContext
from ragwatch.instrumentation.result_transformers import (
    get_default_transformer_registry,
)
from ragwatch.instrumentation.token_usage import register_token_extractor

# ── 1. Custom TelemetryExtractor ──────────────────────────────────────────

class LatencyExtractor:
    """Extract latency from result dicts and record as a span attribute."""
    name = "latency"

    def extract(self, span, span_name, args, result, state):
        if isinstance(result, dict) and "latency_ms" in result:
            safe_set_attribute(span, "custom.latency_ms", result["latency_ms"])


# ── 2. Custom SpanHook (with context + on_error) ─────────────────────────

class AuditHook:
    """Records the function name and catches errors via on_error."""

    def on_start(self, span, args, kwargs, *, context: InstrumentationContext = None):
        if context is not None:
            safe_set_attribute(span, "audit.func_name", context.func_name)
            safe_set_attribute(span, "audit.span_kind", context.span_kind.value)

    def on_end(self, span, result, *, context: InstrumentationContext = None):
        if context is not None and context.raw_result is not None:
            safe_set_attribute(span, "audit.has_result", True)

    def on_error(self, span, exception, *, context: InstrumentationContext = None):
        safe_set_attribute(span, "audit.error_type", type(exception).__name__)


# ── 3. Custom ResultTransformer ───────────────────────────────────────────

class JsonToolTransformer:
    """Transform TOOL results to JSON strings instead of the built-in format."""

    @property
    def span_kind(self):
        return SpanKind.TOOL

    def transform(self, span, args, kwargs, result, result_formatter):
        import json
        if result_formatter is not None:
            return result_formatter(result)
        if isinstance(result, (dict, list)):
            return json.dumps(result, default=str)
        return result


# ── 4. Custom TokenExtractor ─────────────────────────────────────────────

class AnthropicTokenExtractor:
    """Extract token usage from Anthropic-style response objects."""

    def extract(self, span, result):
        usage = getattr(result, "usage", None)
        if isinstance(usage, dict):
            safe_set_attribute(span, "llm.token_count.prompt",
                               usage.get("input_tokens", 0))
            safe_set_attribute(span, "llm.token_count.completion",
                               usage.get("output_tokens", 0))


# ── 5. Wire everything together ──────────────────────────────────────────

def main():
    # Configure with policy and hooks
    configure(RAGWatchConfig(
        service_name="custom-extensions-demo",
        attribute_policy=AttributePolicy(
            max_value_bytes=1024,
            redact_keys=["password", "secret", "api_key"],
            redact_patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        ),
        global_span_hooks=[AuditHook()],
        strict_mode=False,         # swallow extension errors in prod
        global_auto_track_io=True, # auto-capture I/O
    ))

    # Register custom extractor
    get_default_registry().register(LatencyExtractor())

    # Register custom result transformer
    get_default_transformer_registry().register(JsonToolTransformer())

    # Register custom token extractor
    register_token_extractor(AnthropicTokenExtractor())

    # ── 6. Use RAGWatchRuntime ────────────────────────────────────────────
    rt = RAGWatchRuntime.current()
    print(f"Runtime: {rt}")
    print(f"  strict_mode:  {rt.strict_mode}")
    print(f"  auto_track_io: {rt.auto_track_io}")
    print(f"  policy:        {rt.attribute_policy}")

    # ── 7. Traced functions ───────────────────────────────────────────────

    @trace("demo-tool", span_kind=SpanKind.TOOL)
    def fetch_data(query: str):
        return {"parent_id": "p1", "content": f"Result for {query}"}

    @trace("demo-chain", telemetry=["latency"])
    def pipeline(query: str):
        result = fetch_data(query)
        return {"answer": result, "latency_ms": 42.5}

    output = pipeline("What is RAGWatch?")
    print(f"\nPipeline output: {output}")


if __name__ == "__main__":
    main()
