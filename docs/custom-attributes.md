# Custom Attributes

RAGWatch provides multiple ways to attach custom attributes to spans — from simple inline writes to reusable lifecycle hooks and extractors.

---

## 1. `safe_set_attribute` — Inline, Policy-Enforced

The simplest approach. Call `safe_set_attribute()` inside any traced function to set custom span attributes. All writes go through the active `AttributePolicy` for truncation, redaction, and naming validation.

```python
from ragwatch import trace, SpanKind, safe_set_attribute
from opentelemetry import trace as otel_trace


@trace("my-node", span_kind=SpanKind.AGENT)
def process(state):
    span = otel_trace.get_current_span()

    # String, numeric, and boolean attributes
    safe_set_attribute(span, "custom.model_name", "gpt-4")
    safe_set_attribute(span, "custom.temperature", 0.7)
    safe_set_attribute(span, "custom.tokens_used", 150)
    safe_set_attribute(span, "custom.is_cached", False)

    # List attributes (capped by AttributePolicy.max_list_length)
    safe_set_attribute(span, "custom.sources", ["doc1.pdf", "doc2.pdf"])

    return {"result": "done"}
```

### When to Use

- Quick, one-off attributes inside a specific function
- Metrics that are computed during function execution
- Framework-specific metadata not covered by built-in extractors

### Naming Convention

Attribute names must follow dot-separated lowercase convention:

```
✅  custom.latency_ms
✅  ragwatch.custom.model_version
✅  retrieval.chunk.0.score
❌  CustomLatency          (uppercase)
❌  custom-latency         (hyphens)
❌  latency                (no namespace)
```

Names are validated by `validate_attribute_name()` — invalid names are skipped with a warning (or raise in strict mode).

---

## 2. SpanHooks — Lifecycle-Based Attributes

Hooks run at span lifecycle events (`on_start`, `on_end`, `on_error`) without modifying function bodies. Ideal for cross-cutting concerns like auditing, metrics, or error enrichment.

### Context-First Hook (Recommended)

```python
from ragwatch import (
    configure, RAGWatchConfig, SpanKind, trace,
    InstrumentationContext,
)
from opentelemetry.sdk.trace.export import ConsoleSpanExporter


class MetricsHook:
    """Add custom attributes at span start and end."""

    def on_start(self, span, args, kwargs, *, context: InstrumentationContext = None):
        if context:
            context.set_attribute("audit.func_name", context.func_name)
            context.set_attribute("audit.span_kind", context.span_kind.value)

    def on_end(self, span, result, *, context: InstrumentationContext = None):
        if context and isinstance(context.raw_result, dict):
            context.set_attribute("audit.result_keys", list(context.raw_result.keys()))

    def on_error(self, span, exception, *, context: InstrumentationContext = None):
        if context:
            context.set_attribute("audit.error_type", type(exception).__name__)
```

### Hook Registration

```python
# Global — runs on every span
ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    exporter=ConsoleSpanExporter(),
    global_span_hooks=[MetricsHook()],
))

# Per-decorator — runs only on this span
@trace("critical-node", span_kind=SpanKind.AGENT, span_hooks=[MetricsHook()])
def critical_function(state):
    return {"answer": "done"}
```

### InstrumentationContext Fields

The `context` parameter gives hooks access to the full span lifecycle:

| Field | Type | Available in | Description |
|-------|------|-------------|-------------|
| `span` | `Span` | All | The OTel span |
| `span_name` | `str` | All | Span name |
| `span_kind` | `SpanKind` | All | CHAIN, AGENT, TOOL, etc. |
| `func_name` | `str` | All | Decorated function name |
| `args` | `tuple` | All | Function positional args |
| `kwargs` | `dict` | All | Function keyword args |
| `adapter` | `FrameworkAdapter` | All | Active adapter (or `None`) |
| `state` | `dict` | All | Adapter-extracted state (or `None`) |
| `raw_result` | `Any` | `on_end` | Return value before transformation |
| `result` | `Any` | `on_end` | Return value after transformation |
| `normalized` | `dict` | `on_end` | Adapter-normalized semantic keys (or `None`) |
| `exception` | `BaseException` | `on_error` | The caught exception |

### Legacy Hook (Still Supported)

```python
from ragwatch import safe_set_attribute

class SimpleHook:
    def on_start(self, span, args, kwargs):
        safe_set_attribute(span, "hook.started", True)

    def on_end(self, span, result):
        pass
```

### When to Use Hooks

- Cross-cutting concerns (audit trails, SLA tracking, error classification)
- Attributes that depend on span lifecycle timing
- Enrichment that should apply to many spans without changing function code

---

## 3. Custom TelemetryExtractors — Reusable Named Extractors

Extractors are named, reusable components that extract attributes from function results. They are activated per-decorator via the `telemetry=[...]` parameter.

### Context-First Extractor

```python
from ragwatch import (
    configure, RAGWatchConfig, SpanKind, trace,
    InstrumentationContext,
)
from opentelemetry.sdk.trace.export import ConsoleSpanExporter


class LatencyExtractor:
    name = "latency"

    def extract(self, context: InstrumentationContext) -> None:
        result = context.raw_result
        if isinstance(result, dict) and "latency_ms" in result:
            context.set_attribute("custom.latency_ms", result["latency_ms"])


class CostExtractor:
    name = "cost"

    def extract(self, context: InstrumentationContext) -> None:
        result = context.raw_result
        if isinstance(result, dict) and "cost_usd" in result:
            context.set_attribute("custom.cost_usd", result["cost_usd"])
            context.set_attribute("custom.cost_model", result.get("model", "unknown"))
```

### Registration and Activation

```python
# Register at configure time
ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    exporter=ConsoleSpanExporter(),
    custom_extractors=[LatencyExtractor(), CostExtractor()],
))

# Activate by name on specific decorators
@trace("my-pipeline", telemetry=["latency", "cost"])
def pipeline(query: str):
    return {"answer": "result", "latency_ms": 42.5, "cost_usd": 0.003, "model": "gpt-4"}

# Or register manually
from ragwatch import get_default_registry
get_default_registry().register(LatencyExtractor())
```

### When to Use Extractors

- Reusable extraction logic shared across multiple decorators
- Domain-specific metrics (cost tracking, latency, token counts)
- Clean separation of telemetry logic from business logic

---

## 4. `context.set_attribute` vs `safe_set_attribute`

Both are policy-enforced. Choose based on where you're writing:

| Method | Where | When |
|--------|-------|------|
| `context.set_attribute(key, val)` | Inside hooks, extractors, transformers | You have an `InstrumentationContext` |
| `safe_set_attribute(span, key, val)` | Inside traced functions | You have the span directly |

Both go through `AttributePolicy` truncation, redaction, and validation. **Never use raw `span.set_attribute()`** — it bypasses policy enforcement.

---

## 5. AttributePolicy — Controlling What Gets Written

Policy controls are applied automatically to all `safe_set_attribute` and `context.set_attribute` calls:

```python
from ragwatch import configure, RAGWatchConfig, AttributePolicy

ragwatch.configure(RAGWatchConfig(
    service_name="secure-app",
    attribute_policy=AttributePolicy(
        max_value_bytes=4096,           # Truncate long strings
        max_list_length=128,            # Cap list/tuple attributes
        max_indexed_attributes=50,      # Limit chunk.N.field families
        redact_keys=["password"],       # Redact by attribute name substring
        redact_patterns=[r"\d{3}-\d{2}-\d{4}"],  # Redact SSNs by regex
        redact_io_keys=["password", "secret", "api_key", "token"],
    ),
))
```

### Policy Fields

| Field | Default | Description |
|-------|---------|-------------|
| `max_value_bytes` | 4096 | Maximum byte length for string values |
| `max_list_length` | 128 | Maximum items in list/tuple values |
| `max_indexed_attributes` | 50 | Maximum index for `chunk.N.field` families |
| `redact_patterns` | `[]` | Regex patterns — matching values become `[REDACTED]` |
| `redact_keys` | `[]` | Name substrings — matching attribute names become `[REDACTED]` |
| `redact_io_keys` | `["password", "secret", "api_key", "token", "authorization"]` | Keys scrubbed from auto-captured I/O payloads |

### Examples

```python
# String truncation
safe_set_attribute(span, "custom.long_text", "x" * 10000)
# → Truncated to ~4096 bytes + "...[truncated]"

# Key-based redaction
safe_set_attribute(span, "user.password", "secret123")
# → "[REDACTED]"

# Pattern-based redaction
safe_set_attribute(span, "user.ssn", "123-45-6789")
# → "[REDACTED]"

# List truncation
safe_set_attribute(span, "custom.items", list(range(500)))
# → First 128 items only

# I/O payload scrubbing (auto-captured input/output)
@trace("login")
def login(username: str, password: str):
    return {"token": "abc123"}
# input.value will have password=[REDACTED]
# output.value will have token=[REDACTED]
```

See [Configuration](configuration.md) for the full `AttributePolicy` reference.

---

## Next Steps

- [Telemetry Extraction](telemetry-extraction.md) — built-in extractors and telemetry helpers
- [Configuration](configuration.md) — all `RAGWatchConfig` options
- [Extension Guide](EXTENSION_GUIDE.md) — writing adapters, transformers, token extractors
