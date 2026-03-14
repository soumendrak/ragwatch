"""RAGWatch Custom Extensions — context-first extension points.

This example demonstrates all extension points using the **canonical
context-first** pattern.  Legacy signatures are still supported but
context-first is recommended for new code.

Extension points shown:

1. Custom TelemetryExtractor (context-first)
2. Custom SpanHook (context-first with on_error)
3. Custom ResultTransformer (context-first)
4. Custom TokenExtractor (context-first)
5. AttributePolicy — truncation, redaction, I/O privacy
6. InstrumentationContext — the unified extension parameter
7. RAGWatchRuntime — scoped access to SDK state
"""

from __future__ import annotations

from ragwatch import (
    AttributePolicy,
    InstrumentationContext,
    RAGWatchConfig,
    RAGWatchRuntime,
    SpanKind,
    configure,
    get_default_registry,
    trace,
)
from ragwatch.instrumentation.result_transformers import (
    get_default_transformer_registry,
)
from ragwatch.instrumentation.token_usage import register_token_extractor

# ── 1. Custom TelemetryExtractor (context-first) ─────────────────────────
#
# Context-first extractors receive a single InstrumentationContext argument.
# Use context.set_attribute() for policy-enforced writes.

class LatencyExtractor:
    """Extract latency from result dicts and record as a span attribute."""
    name = "latency"

    def extract(self, context: InstrumentationContext) -> None:
        result = context.raw_result
        if isinstance(result, dict) and "latency_ms" in result:
            context.set_attribute("custom.latency_ms", result["latency_ms"])


# ── 2. Custom SpanHook (context-first with on_error) ─────────────────────
#
# Hooks accept an optional context= keyword argument.  When present it
# provides access to span, state, adapter, and the set_attribute writer.

class AuditHook:
    """Records the function name and catches errors via on_error."""

    def on_start(self, span, args, kwargs, *, context: InstrumentationContext = None):
        if context is not None:
            context.set_attribute("audit.func_name", context.func_name)
            context.set_attribute("audit.span_kind", context.span_kind.value)

    def on_end(self, span, result, *, context: InstrumentationContext = None):
        if context is not None and context.raw_result is not None:
            context.set_attribute("audit.has_result", True)

    def on_error(self, span, exception, *, context: InstrumentationContext = None):
        if context is not None:
            context.set_attribute("audit.error_type", type(exception).__name__)


# ── 3. Custom ResultTransformer (context-first) ──────────────────────────
#
# Context-first transformers receive InstrumentationContext and return the
# transformed result.  The raw result is in context.raw_result.

class JsonToolTransformer:
    """Transform TOOL results to JSON strings instead of the built-in format."""

    @property
    def span_kind(self):
        return SpanKind.TOOL

    def transform(self, context: InstrumentationContext):
        import json
        raw = context.raw_result
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, default=str)
        return raw


# ── 4. Custom TokenExtractor (context-first) ─────────────────────────────
#
# Context-first token extractors receive InstrumentationContext.

class AnthropicTokenExtractor:
    """Extract token usage from Anthropic-style response objects."""

    def extract(self, context: InstrumentationContext) -> None:
        usage = getattr(context.raw_result, "usage", None)
        if isinstance(usage, dict):
            context.set_attribute("llm.token_count.prompt",
                                  usage.get("input_tokens", 0))
            context.set_attribute("llm.token_count.completion",
                                  usage.get("output_tokens", 0))


# ── 5. Wire everything together ──────────────────────────────────────────

def main():
    # Configure with policy and hooks
    configure(RAGWatchConfig(
        service_name="custom-extensions-demo",
        attribute_policy=AttributePolicy(
            max_value_bytes=1024,
            max_indexed_attributes=20,
            redact_keys=["password", "secret", "api_key"],
            redact_patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            # I/O privacy: these keys are scrubbed from captured input/output
            redact_io_keys=["password", "secret", "api_key", "token", "authorization"],
        ),
        global_span_hooks=[AuditHook()],
        custom_transformers=[JsonToolTransformer()],
        custom_token_extractors=[AnthropicTokenExtractor()],
        strict_mode=False,         # swallow extension errors in prod
        global_auto_track_io=True, # auto-capture I/O
    ))

    # Register custom extractor
    get_default_registry().register(LatencyExtractor())

    # ── 6. Use RAGWatchRuntime ────────────────────────────────────────────
    rt = RAGWatchRuntime.current()
    print(f"Runtime: {rt}")
    print(f"  strict_mode:   {rt.strict_mode}")
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
