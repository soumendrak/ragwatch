# RAGWatch Extension Guide

Build custom telemetry extractors, span hooks, result transformers, token extractors, and framework adapters for RAGWatch.

> **InstrumentationContext is the canonical extension model.** All extension points accept an `InstrumentationContext` parameter that provides the span, state, adapter, raw result, and a policy-enforced `set_attribute()` writer.

---

## InstrumentationContext

Every extension point receives an `InstrumentationContext`:

```python
from ragwatch import InstrumentationContext

# Available fields:
ctx.span          # The active OTel span
ctx.span_name     # Resolved span name
ctx.span_kind     # SpanKind enum
ctx.func_name     # Decorated function's __qualname__
ctx.args          # Positional args to the decorated function
ctx.kwargs        # Keyword args to the decorated function
ctx.adapter       # The FrameworkAdapter (or None)
ctx.state         # adapter.extract_state() result (or first dict arg)
ctx.normalized    # adapter.normalize_result() output (or None)
ctx.raw_result    # Return value before transformation
ctx.result        # Return value after transformation
ctx.exception     # Exception if the function raised

ctx.set_attribute(key, value)  # Policy-enforced attribute writer
```

Use `ctx.set_attribute()` instead of `span.set_attribute()` — it automatically applies truncation, redaction, and naming validation from the active `AttributePolicy`.

---

## 1. Custom TelemetryExtractor

Extractors run after the decorated function returns.  They read the result and write span attributes.

### Implementation

```python
from ragwatch import InstrumentationContext

class LatencyExtractor:
    name = "latency"  # referenced via telemetry=["latency"]

    def extract(self, context: InstrumentationContext) -> None:
        result = context.raw_result
        if isinstance(result, dict) and "latency_ms" in result:
            context.set_attribute("custom.latency_ms", result["latency_ms"])
```

### Registration

```python
from ragwatch import get_default_registry, configure, RAGWatchConfig

# Option A: register manually
get_default_registry().register(LatencyExtractor())

# Option B: via configure()
configure(RAGWatchConfig(custom_extractors=[LatencyExtractor()]))
```

### Usage

```python
@trace("my-node", telemetry=["latency"])
def my_function(state):
    return {"latency_ms": 42.5, "data": "..."}
```

---

## 2. Custom SpanHook

Hooks run at span lifecycle events: `on_start`, `on_end`, `on_error`.

### Implementation

```python
from ragwatch import InstrumentationContext

class AuditHook:
    def on_start(self, span, args, kwargs, *, context: InstrumentationContext = None):
        if context:
            context.set_attribute("audit.func_name", context.func_name)

    def on_end(self, span, result, *, context: InstrumentationContext = None):
        if context:
            context.set_attribute("audit.has_result", result is not None)

    def on_error(self, span, exception, *, context: InstrumentationContext = None):
        if context:
            context.set_attribute("audit.error_type", type(exception).__name__)
```

### Registration

```python
# Global (runs on every span)
configure(RAGWatchConfig(global_span_hooks=[AuditHook()]))

# Per-decorator
@trace("my-span", span_hooks=[AuditHook()])
def my_function(): ...
```

---

## 3. Custom ResultTransformer

Transformers convert raw results per `SpanKind`.  Only one transformer per kind is active.

### Implementation

```python
from ragwatch import InstrumentationContext, SpanKind

class JsonToolTransformer:
    @property
    def span_kind(self):
        return SpanKind.TOOL

    def transform(self, context: InstrumentationContext):
        import json
        raw = context.raw_result
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, default=str)
        return raw
```

### Registration

```python
configure(RAGWatchConfig(custom_transformers=[JsonToolTransformer()]))

# Or manually:
from ragwatch.instrumentation.result_transformers import get_default_transformer_registry
get_default_transformer_registry().register(JsonToolTransformer())
```

---

## 4. Custom TokenExtractor

Token extractors scan results for LLM usage metadata and record token counts.

### Context-first (recommended)

```python
from ragwatch import InstrumentationContext

class AnthropicTokenExtractor:
    def extract(self, context: InstrumentationContext) -> None:
        usage = getattr(context.raw_result, "usage", None)
        if isinstance(usage, dict):
            context.set_attribute("llm.token_count.prompt", usage.get("input_tokens", 0))
            context.set_attribute("llm.token_count.completion", usage.get("output_tokens", 0))
```

### Registration

```python
configure(RAGWatchConfig(custom_token_extractors=[AnthropicTokenExtractor()]))
```

---

## 5. Custom FrameworkAdapter

Adapters tell RAGWatch how to extract state from framework-specific function arguments and optionally normalize results into semantic keys.

### Mandatory vs Optional Methods

| Method | Required? | Purpose |
|--------|-----------|---------|
| `name: str` | **Mandatory** | Unique adapter identifier (e.g. `"langgraph"`) |
| `extract_state(args, kwargs)` | **Mandatory** | Pull the framework's state dict from function arguments |
| `default_extractors()` | **Mandatory** | Extractors registered when the adapter is activated |
| `capabilities()` | Optional | Declare supported telemetry features (e.g. `{"routing"}`) |
| `normalize_result(raw_result, state)` | Optional | Translate results into semantic keys for `ctx.normalized` |

Optional methods are detected via `getattr` at runtime and are **not** part of the `FrameworkAdapter` Protocol to preserve `@runtime_checkable` compatibility with minimal adapters.

When an adapter declares capabilities, RAGWatch records a
`ragwatch.unsupported_telemetry` span event if a decorator requests telemetry
outside that set. Extraction still runs so custom extractors and partial
framework support continue to work.

### Example

```python
from ragwatch.adapters.base import FrameworkAdapter

class MyFrameworkAdapter:
    name = "myframework"

    def extract_state(self, args, kwargs):
        """Return the framework's state dict from function arguments."""
        return args[0] if args and isinstance(args[0], dict) else None

    def default_extractors(self):
        """Return extractors relevant to this framework."""
        return [LatencyExtractor()]

    # Optional: declare capabilities
    def capabilities(self):
        return {"routing", "tool_calls"}

    # Optional: normalize results into semantic keys
    def normalize_result(self, raw_result, state):
        """Translate framework-specific shapes into well-known keys.

        Return a dict with any of these semantic keys (all optional):
        - tool_calls: list[dict] — LLM tool-call decisions
        - routing_target: str — destination node name
        - routing_reason: str — human-readable reason
        - agent_answer: str — final answer string
        - is_fallback: bool — whether the answer is a fallback
        - rewritten_questions: list[str] — decomposed queries
        - is_clear: bool — query clarity flag
        - original_query: str — original user query
        - compression_tokens_before: int
        - compression_tokens_after: int
        - context_summary: str
        - queries_run: list[str] — search queries executed
        - parents_retrieved: list[str] — parent document IDs retrieved
        """
        if not isinstance(raw_result, dict):
            return None
        norm = {}
        if "latency" in raw_result:
            norm["custom_latency"] = raw_result["latency"]
        return norm or None
```

### Registration

```python
configure(RAGWatchConfig(adapters=[MyFrameworkAdapter()]))

# Use in decorators:
@trace("my-node", adapter="myframework")
def process(state): ...
```

---

## Failure Isolation

Extension errors (hooks, extractors, transformers, token extractors, normalization) are:

- **Caught** and logged as warnings
- **Recorded** as span events (e.g. `ragwatch.hook_error`, `ragwatch.normalize_error`)
- **Never** crash the decorated function

Enable `strict_mode=True` in `RAGWatchConfig` to **re-raise** extension errors during development/testing.

---

## AttributePolicy

Control truncation, redaction, and cardinality:

```python
from ragwatch import AttributePolicy

policy = AttributePolicy(
    max_value_bytes=4096,          # String truncation limit
    max_list_length=128,           # Max items in list/tuple attributes
    max_indexed_attributes=50,     # Max index for chunk.N.field families
    redact_keys=["password"],      # Redact by attribute name substring
    redact_patterns=[r"\d{3}-\d{2}-\d{4}"],  # Redact by value regex (SSN)
    redact_io_keys=["password", "secret", "api_key", "token", "authorization"],
)
```

---

## API Stability

| Surface | Status |
|---------|--------|
| `configure()`, `trace()`, `record_feedback()` | **Stable** |
| `InstrumentationContext` fields | **Stable** |
| `TelemetryExtractor`, `SpanHook`, `ResultTransformer`, `TokenExtractor` protocols | **Stable** (`InstrumentationContext`) |
| `FrameworkAdapter` protocol (`extract_state`, `default_extractors`) | **Stable** |
| `AttributePolicy` fields | **Stable** |
| `normalize_result()` on adapters | **Experimental** — semantic keys may evolve |
| `capabilities()` on adapters | **Experimental** |

---

## See Also

- `examples/custom_extensions.py` — complete working example
- `examples/minimal_rag.py` — basic RAG pipeline
- `examples/langgraph_agent.py` — LangGraph integration
