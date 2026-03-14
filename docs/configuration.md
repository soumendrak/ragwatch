# Configuration

Complete reference for `RAGWatchConfig`, `AttributePolicy`, and all SDK configuration options.

---

## RAGWatchConfig

The central configuration dataclass. Pass it to `ragwatch.configure()` to initialize the SDK.

```python
import ragwatch
from ragwatch import RAGWatchConfig, AttributePolicy
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

ragwatch.configure(RAGWatchConfig(
    service_name="my-rag-app",
    exporter=ConsoleSpanExporter(),
))
```

### All Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `service_name` | `str` | `"ragwatch-service"` | Service name in OTel resource attributes |
| `exporters` | `list[SpanExporter]` | `[]` | List of span exporters (one processor per exporter) |
| `exporter` | `SpanExporter` | `None` | Single exporter (legacy API, ignored when `exporters` is non-empty) |
| `max_embedding_dims` | `int` | `512` | Max embedding dimensions stored in context |
| `custom_extractors` | `list[TelemetryExtractor]` | `[]` | Additional extractors registered at configure time |
| `global_span_hooks` | `list[SpanHook]` | `[]` | Hooks that run on every traced span |
| `adapters` | `list[FrameworkAdapter]` | `[]` | Framework adapters to register |
| `custom_transformers` | `list[ResultTransformer]` | `[]` | Custom result transformers |
| `custom_token_extractors` | `list[TokenExtractor]` | `[]` | Custom token usage extractors |
| `attribute_policy` | `AttributePolicy` | `None` | Truncation, redaction, and cardinality controls |
| `strict_mode` | `bool` | `False` | Re-raise extension errors instead of swallowing |
| `global_auto_track_io` | `bool` | `True` | Auto-capture function I/O as span attributes |

---

## Multiple Backends

Send traces to multiple OTel backends simultaneously. Each exporter gets its own `BatchSpanProcessor` with independent buffering and retry:

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

ragwatch.configure(RAGWatchConfig(
    service_name="my-rag-app",
    exporters=[
        OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"),  # Jaeger
        OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"),  # Phoenix
    ],
))
```

---

## AttributePolicy

Controls what gets written to spans. Applied automatically to all `safe_set_attribute()` and `context.set_attribute()` calls.

```python
from ragwatch import AttributePolicy

policy = AttributePolicy(
    max_value_bytes=4096,
    max_list_length=128,
    max_indexed_attributes=50,
    redact_keys=["password", "secret"],
    redact_patterns=[r"\d{3}-\d{2}-\d{4}"],
    redact_io_keys=["password", "secret", "api_key", "token", "authorization"],
)

ragwatch.configure(RAGWatchConfig(
    service_name="secure-app",
    attribute_policy=policy,
))
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_value_bytes` | `int` | `4096` | String values exceeding this are truncated with `...[truncated]` |
| `max_list_length` | `int` | `128` | List/tuple values are capped at this many items |
| `max_indexed_attributes` | `int` | `50` | Max index for `chunk.N.field` families; writes beyond are skipped |
| `redact_patterns` | `list[str]` | `[]` | Regex patterns — matching string **values** become `[REDACTED]` |
| `redact_keys` | `list[str]` | `[]` | Name substrings — matching attribute **names** become `[REDACTED]` |
| `redact_io_keys` | `list[str]` | `["password", "secret", "api_key", "token", "authorization"]` | Keys scrubbed from auto-captured I/O payloads |

### String Truncation

```python
policy = AttributePolicy(max_value_bytes=1024)

# A 10KB string will be truncated to ~1024 bytes + "...[truncated]"
safe_set_attribute(span, "custom.long_text", "x" * 10000)
```

### Key-Based Redaction

Any attribute whose name contains a `redact_keys` substring is replaced:

```python
policy = AttributePolicy(redact_keys=["password", "secret"])

safe_set_attribute(span, "user.password", "hunter2")
# → "[REDACTED]"

safe_set_attribute(span, "api.secret_key", "sk-abc123")
# → "[REDACTED]"

safe_set_attribute(span, "user.name", "Alice")
# → "Alice" (no match)
```

### Pattern-Based Redaction

String values matching any regex pattern are replaced:

```python
policy = AttributePolicy(
    redact_patterns=[
        r"\d{3}-\d{2}-\d{4}",                    # SSN
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b",  # Email
    ]
)

safe_set_attribute(span, "user.ssn", "123-45-6789")
# → "[REDACTED]"

safe_set_attribute(span, "user.email", "alice@example.com")
# → "[REDACTED]"
```

### I/O Payload Scrubbing

When auto I/O tracking is enabled, dict keys matching `redact_io_keys` are scrubbed before serialization:

```python
policy = AttributePolicy(
    redact_io_keys=["password", "token", "api_key"],
)

@trace("login")
def login(username: str, password: str):
    return {"token": "abc123", "user": "alice"}

# input.value  → {"username": "alice", "password": "[REDACTED]"}
# output.value → {"token": "[REDACTED]", "user": "alice"}
```

Set `redact_io_keys=[]` to disable I/O scrubbing entirely.

### Collection Size Limits

```python
policy = AttributePolicy(max_list_length=10)

safe_set_attribute(span, "custom.items", list(range(500)))
# → [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  (first 10 only)
```

---

## Strict Mode

In production, extension errors (hooks, extractors, transformers, normalizers) are caught, logged, and recorded as span events — they never crash your application.

Enable `strict_mode` during development and testing to surface errors immediately:

```python
ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    strict_mode=True,  # Re-raise extension errors
))
```

### Failure Isolation (Production)

When `strict_mode=False` (default), extension failures follow a 3-step pattern:

1. **Warning log** — `_logger.warning("ragwatch: hook/extractor/transformer error: ...")`
2. **Span event** — e.g. `ragwatch.hook_error`, `ragwatch.extractor_error`, `ragwatch.normalize_error`
3. **Continue execution** — the decorated function result is unaffected

### Error Types Recorded as Span Events

| Span Event | Source |
|-----------|--------|
| `ragwatch.hook_error` | SpanHook `on_start`/`on_end`/`on_error` failure |
| `ragwatch.extractor_error` | TelemetryExtractor `extract()` failure |
| `ragwatch.transform_error` | ResultTransformer `transform()` failure |
| `ragwatch.normalize_error` | Adapter `normalize_result()` failure |
| `ragwatch.token_extractor_error` | TokenExtractor `extract()` failure |

---

## Auto I/O Tracking

By default, all decorators capture function arguments as `input.value` and return values as `output.value` (JSON-serialized, 4KB truncation).

### Disable Per-Decorator

```python
@trace("sensitive-operation", auto_track_io=False)
def process_pii(data):
    return {"ssn": "123-45-6789"}  # Not captured
```

### Disable Globally

```python
ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    global_auto_track_io=False,  # No auto I/O on any span
))
```

When `global_auto_track_io=False`, per-decorator `auto_track_io=True` is overridden — nothing is auto-captured.

---

## Idempotent Configuration

`configure()` is safe to call multiple times. Each call resets all global registries (hooks, extractors, adapters, transformers, token extractors) before re-registering:

```python
# First configuration
ragwatch.configure(RAGWatchConfig(service_name="v1"))

# Re-configure — previous hooks/extractors are cleared
ragwatch.configure(RAGWatchConfig(
    service_name="v2",
    global_span_hooks=[MyNewHook()],
))
```

---

## RAGWatchRuntime

A read-only accessor for the active SDK state. Useful for introspection in extensions:

```python
from ragwatch import RAGWatchRuntime

rt = RAGWatchRuntime.current()
print(rt.config)              # RAGWatchConfig
print(rt.strict_mode)         # bool
print(rt.attribute_policy)    # AttributePolicy | None
print(rt.auto_track_io)       # bool
print(rt.extractor_registry)  # ExtractorRegistry
print(rt.adapter_registry)    # dict[str, FrameworkAdapter]
print(rt.transformer_registry)  # ResultTransformerRegistry
print(rt.token_extractor_registry)  # list[TokenExtractor]
```

---

## Next Steps

- [Custom Attributes](custom-attributes.md) — hooks, extractors, and policy in action
- [Extension Guide](EXTENSION_GUIDE.md) — full protocol reference
- [API Reference](api-reference.md) — complete public API
