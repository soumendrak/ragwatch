# API Reference

Complete reference for the RAGWatch public API surface.

---

## Core Functions

### `configure(config=None, **kwargs)`

Initialize the RAGWatch SDK and return the active `RAGWatchRuntime`. Idempotent — safe to call multiple times (resets all registries).

```python
import ragwatch
from ragwatch import RAGWatchConfig

ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    exporter=ConsoleSpanExporter(),
))

runtime = ragwatch.configure(service_name="my-app")
runtime.trace("my-operation")

# Or with kwargs
ragwatch.configure(service_name="my-app")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `RAGWatchConfig \| None` | Configuration instance. If `None`, created from `**kwargs`. |
| `**kwargs` | | Forwarded to `RAGWatchConfig` when `config` is `None`. |

**Returns:** `RAGWatchRuntime` — read-only accessor for active SDK state.

---

### `trace(span_name, span_kind, ...)`

Decorator for tracing sync and async functions. Creates an OpenTelemetry span around the decorated function.

```python
from ragwatch import trace, SpanKind

@trace("my-operation", span_kind=SpanKind.AGENT)
def my_function(state):
    return {"result": "done"}

@trace("async-op", span_kind=SpanKind.CHAIN)
async def async_function(data):
    return await process(data)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `span_name` | `str \| Callable` | Function name | Span name (or the function itself when used without args) |
| `span_kind` | `SpanKind` | `CHAIN` | OpenInference span kind |
| `auto_track_io` | `bool` | `True` | Capture input/output as span attributes |
| `telemetry` | `list[str]` | `None` | Extractor names to activate |
| `result_formatter` | `Callable` | `None` | Custom result formatting function |
| `span_hooks` | `list[SpanHook]` | `None` | Per-decorator lifecycle hooks |
| `adapter` | `str` | `None` | Adapter name to use (e.g. `"langgraph"`, `"crewai"`) |

**Usage patterns:**

```python
# With explicit name
@trace("my-span")
def func(): ...

# Without parentheses (uses function name)
@trace
def func(): ...

# Full options
@trace("my-span", span_kind=SpanKind.AGENT, telemetry=["tool_calls"], auto_track_io=False)
def func(state): ...
```

---

### `record_feedback(trace_id, score)`

Record user feedback as a separate span linked to a trace.

```python
from ragwatch import record_feedback

record_feedback(trace_id="abc123", score=0.85)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trace_id` | `str` | Trace ID of the request being rated |
| `score` | `float` | Feedback score (typically 0.0 – 1.0) |

---

### `chunk_relevance_score(chunk_embeddings, query_embedding=None)`

Compute cosine similarity between query and chunk embeddings.

```python
from ragwatch.instrumentation.evaluators import chunk_relevance_score

scores = chunk_relevance_score(chunk_embeddings)
# Or with explicit query embedding:
scores = chunk_relevance_score(chunk_embeddings, query_embedding=[0.5, 0.3, 0.2])
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_embeddings` | `list[list[float]]` | required | Chunk embedding vectors |
| `query_embedding` | `list[float] \| None` | `None` | Explicit query embedding (falls back to context-stored) |

**Returns:** `list[float]` — cosine similarity scores, one per chunk.

**Raises:** `ValueError` if no query embedding is available.

---

### `get_active_config()`

Return the config passed to the last `configure()` call, or `None`.

```python
from ragwatch import get_active_config

config = get_active_config()
```

---

## Attribute Functions

### `safe_set_attribute(span, key, value)`

Policy-enforced span attribute writer. Applies `AttributePolicy` truncation, redaction, and validation.

```python
from ragwatch import safe_set_attribute
from opentelemetry import trace as otel_trace

span = otel_trace.get_current_span()
safe_set_attribute(span, "custom.latency_ms", 42.5)
```

### `safe_set_attributes(span, attributes)`

Set multiple attributes at once:

```python
from ragwatch import safe_set_attributes

safe_set_attributes(span, {
    "custom.model": "gpt-4",
    "custom.tokens": 150,
})
```

### `validate_attribute_name(name)`

Check if an attribute name follows dot-separated lowercase convention. Returns `bool`.

```python
from ragwatch import validate_attribute_name

validate_attribute_name("custom.latency_ms")  # True
validate_attribute_name("InvalidName")         # False
```

---

## Telemetry Helpers

All helpers are exported from the top-level `ragwatch` package. See [Telemetry Extraction](telemetry-extraction.md) for full documentation.

| Function | Description |
|----------|-------------|
| `record_chunks(results, query, span, max_content_chars)` | Per-chunk retrieval telemetry |
| `record_tool_calls(tool_calls, span)` | LLM tool-call decisions |
| `record_routing(from_node, to_node, reason, span)` | Routing/edge decisions |
| `record_agent_completion(status, ...)` | Agent task completion metadata |
| `record_context_compression(tokens_before, tokens_after, ...)` | Compression statistics |
| `record_query_rewrite(original_query, rewritten_questions, is_clear, span)` | Query decomposition |

---

## Configuration Classes

### `RAGWatchConfig`

See [Configuration](configuration.md) for full field reference.

```python
from ragwatch import RAGWatchConfig

config = RAGWatchConfig(
    service_name="my-app",
    exporter=ConsoleSpanExporter(),
    strict_mode=False,
    global_auto_track_io=True,
)
```

### `AttributePolicy`

See [Configuration](configuration.md#attributepolicy) for full field reference.

```python
from ragwatch import AttributePolicy

policy = AttributePolicy(
    max_value_bytes=4096,
    max_list_length=128,
    redact_keys=["password"],
)
```

### `SpanKind`

OpenInference span kind enum:

| Value | Use for |
|-------|---------|
| `SpanKind.CHAIN` | Root spans, orchestration, pipelines |
| `SpanKind.AGENT` | Nodes, pipeline steps, agent tasks |
| `SpanKind.TOOL` | Tool implementations |
| `SpanKind.RETRIEVER` | Vector search, retrieval |
| `SpanKind.EMBEDDING` | Embedding generation |

### `InstrumentationContext`

Context object passed to all extension points. See [Custom Attributes](custom-attributes.md#instrumentationcontext-fields) for field reference.

```python
from ragwatch import InstrumentationContext
```

### `RAGWatchRuntime`

Read-only accessor for active SDK state:

```python
from ragwatch import RAGWatchRuntime

rt = RAGWatchRuntime.current()
rt.config                 # RAGWatchConfig | None
rt.strict_mode            # bool
rt.attribute_policy       # AttributePolicy | None
rt.auto_track_io          # bool
rt.extractor_registry     # ExtractorRegistry
rt.adapter_registry       # dict[str, FrameworkAdapter]
rt.transformer_registry   # ResultTransformerRegistry
rt.token_extractor_registry  # list[TokenExtractor]
```

---

## Extension Protocols

### `TelemetryExtractor`

Pluggable telemetry extraction. Must have a `name` attribute and an `extract()` method.

```python
class MyExtractor:
    name = "my_extractor"

    def extract(self, context: InstrumentationContext) -> None:
        context.set_attribute("custom.key", "value")
```

**Registry:** `get_default_registry()` returns the `ExtractorRegistry`.

### `SpanHook`

Span lifecycle hook with `on_start`, `on_end`, and optional `on_error`.

```python
class MyHook:
    def on_start(self, span, args, kwargs, *, context: InstrumentationContext = None): ...
    def on_end(self, span, result, *, context: InstrumentationContext = None): ...
    def on_error(self, span, exception, *, context: InstrumentationContext = None): ...
```

**Registration:** `register_global_hook(hook)` or `RAGWatchConfig(global_span_hooks=[...])`.

### `FrameworkAdapter`

Protocol for framework integrations. See [Extension Guide](EXTENSION_GUIDE.md) for full contract.

**Mandatory methods:** `name`, `extract_state(args, kwargs)`, `default_extractors()`

**Optional methods:** `capabilities()`, `normalize_result(raw_result, state)`

**Registration:** `register_adapter(adapter)` or `RAGWatchConfig(adapters=[...])`.

### `ResultTransformer`

Pluggable result transformation per `SpanKind`.

```python
class MyTransformer:
    @property
    def span_kind(self):
        return SpanKind.TOOL

    # Context-first
    def transform(self, context: InstrumentationContext):
        return str(context.raw_result)

    # Legacy
    def transform(self, span, args, kwargs, result, result_formatter):
        return str(result)
```

### `TokenExtractor`

Pluggable token-usage extraction.

```python
class MyTokenExtractor:
    # Context-first
    def extract(self, context: InstrumentationContext) -> None:
        usage = getattr(context.raw_result, "usage", None)
        if usage:
            context.set_attribute("llm.token_count.prompt", usage["input_tokens"])
```

---

## Framework Adapters

### LangGraph

```python
from ragwatch.adapters.langgraph import node, workflow, tool
from ragwatch.adapters.langgraph import LangGraphAdapter
```

| Decorator | SpanKind | Description |
|-----------|----------|-------------|
| `@node("name")` | `AGENT` | Graph nodes |
| `@workflow("name")` | `CHAIN` | Workflow orchestrators |
| `@tool("name")` | `TOOL` | Tool implementations |

### CrewAI

```python
from ragwatch.adapters.crewai import node, endpoint
from ragwatch.adapters.crewai import CrewAIAdapter
```

| Decorator | SpanKind | Description |
|-----------|----------|-------------|
| `@node("name")` | `AGENT` | Individual agents/tasks |
| `@endpoint("name")` | `CHAIN` | Crew orchestration endpoints |

---

## Full `__all__` Export List

```python
from ragwatch import (
    # Core
    configure, trace, RAGWatchConfig, SpanKind,
    InstrumentationContext, RAGWatchRuntime, get_active_config,

    # Quality scores
    record_feedback, chunk_relevance_score,

    # Extension protocols
    TelemetryExtractor, ExtractorRegistry, get_default_registry,
    SpanHook, register_global_hook,
    FrameworkAdapter, register_adapter,
    ResultTransformer, ResultTransformerRegistry, get_default_transformer_registry,
    TokenExtractor, register_token_extractor,

    # Attribute policy & writer
    AttributePolicy, validate_attribute_name,
    safe_set_attribute, safe_set_attributes,

    # Telemetry helpers
    record_chunks, record_agent_completion, record_routing,
    record_tool_calls, record_context_compression, record_query_rewrite,
)
```
