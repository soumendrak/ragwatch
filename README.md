# RAGWatch

> Quality scores in your RAG traces — computed, not just recorded.

**RAGWatch** is an OpenTelemetry-native Python SDK that adds semantic quality scores to your RAG traces. Unlike generic tracing tools, RAGWatch computes `chunk_relevance_score` inline via cosine similarity — zero LLM calls, ~1-5 ms overhead.

## Installation

```bash
pip install ragwatch                  # Core SDK
pip install ragwatch[langgraph]       # + LangGraph adapter
pip install ragwatch[crewai]          # + CrewAI adapter
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add ragwatch                       # Core SDK
uv add ragwatch --extra langgraph     # + LangGraph adapter
uv add ragwatch --extra crewai        # + CrewAI adapter
```

---

## Quickstart — Minimal RAG Pipeline

```python
import ragwatch
from ragwatch import RAGWatchConfig, SpanKind, trace
from ragwatch.instrumentation.evaluators import chunk_relevance_score
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

ragwatch.configure(RAGWatchConfig(
    service_name="my-rag-app",
    exporter=ConsoleSpanExporter(),
))

@trace("embedding.generate", span_kind=SpanKind.EMBEDDING)
def embed_query(text: str) -> list[float]:
    return [0.5, 0.3, 0.2]  # Replace with your embedding API

@trace("retrieval.search", span_kind=SpanKind.RETRIEVER)
def retrieve_chunks(query: str) -> list[dict]:
    chunk_embeddings = [[0.5, 0.3, 0.2], [0.1, 0.9, 0.0]]
    scores = chunk_relevance_score(chunk_embeddings)
    return [{"text": "chunk", "score": s} for s in scores]

@trace("response.generate", span_kind=SpanKind.CHAIN)
def generate_response(chunks: list[dict]) -> str:
    return "Generated response based on retrieved chunks"

# Run the pipeline
embedding = embed_query("What is RAG?")
chunks = retrieve_chunks("What is RAG?")
response = generate_response(chunks)
```

Each decorated function creates an OpenTelemetry span with:
- **`input.value`** / **`output.value`** — auto-captured (4KB truncation)
- **`chunk.relevance_score`** — computed inline via cosine similarity
- **`openinference.span.kind`** — `EMBEDDING`, `RETRIEVER`, or `CHAIN`

---

## Instrumenting a LangGraph Application

RAGWatch provides `@node`, `@workflow`, and `@tool` decorators that map directly to LangGraph concepts. Each decorator automatically sets the correct `SpanKind` and wires up the LangGraph adapter for rich telemetry extraction.

```python
from typing import TypedDict
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from langgraph.graph import StateGraph, END

import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.langgraph import node, workflow, tool
from ragwatch.instrumentation.evaluators import chunk_relevance_score


ragwatch.configure(RAGWatchConfig(
    service_name="langgraph-rag-app",
    exporter=ConsoleSpanExporter(),
))


class RAGState(TypedDict):
    query: str
    embedding: list[float]
    docs: list[dict]
    response: str


@node("embed-node")
def embed_node(state: RAGState) -> RAGState:
    """Generate query embedding."""
    embedding = [0.5, 0.3, 0.2]  # Replace with your embedding API
    return {**state, "embedding": embedding}


@node("retrieve-node", telemetry=["agent_completion"])
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve chunks with relevance scores."""
    chunk_embeddings = [[0.5, 0.3, 0.2], [0.1, 0.9, 0.0]]
    scores = chunk_relevance_score(chunk_embeddings)
    docs = [
        {"text": "RAG combines retrieval with generation.", "score": scores[0]},
        {"text": "Transformers use self-attention.", "score": scores[1]},
    ]
    return {**state, "docs": docs}


@node("generate-node")
def generate_node(state: RAGState) -> RAGState:
    """Generate response from top chunk."""
    top_doc = max(state["docs"], key=lambda d: d["score"])
    return {**state, "response": f"Based on: {top_doc['text']}"}


@workflow("rag-pipeline")
def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("embed", embed_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.set_entry_point("embed")
    return graph.compile()


rag_graph = build_graph()
result = rag_graph.invoke({
    "query": "What is RAG?", "embedding": [], "docs": [], "response": ""
})
```

### LangGraph Decorator Reference

| Decorator | SpanKind | Use for |
|-----------|----------|---------|
| `@node("name")` | `AGENT` | Graph nodes |
| `@workflow("name")` | `CHAIN` | Workflow orchestrators |
| `@tool("name")` | `TOOL` | Tool implementations |

### Built-in Telemetry Extraction

Pass `telemetry=[...]` to any LangGraph decorator to automatically extract structured telemetry from the node's return value — no code changes inside the function:

```python
@node("orchestrator", telemetry=["tool_calls", "routing"])
def orchestrator(state):
    # RAGWatch inspects the return value and extracts tool_calls, routing info
    return {"messages": [ai_message_with_tool_calls]}

@node("answer-node", telemetry=["agent_completion"])
def collect_answer(state):
    return {"final_answer": "The answer is 42", "agent_answers": [...]}

@node("rewrite-node", telemetry=["query_rewrite"])
def rewrite_query(state):
    return {"rewrittenQuestions": ["sub-q1", "sub-q2"], "questionIsClear": False}
```

Available strategies: `tool_calls`, `routing`, `agent_completion`, `query_rewrite`, `compression`.

---

## Instrumenting a CrewAI Application

RAGWatch provides `@node` and `@endpoint` decorators for CrewAI workflows.

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.crewai import node, endpoint


ragwatch.configure(RAGWatchConfig(
    service_name="crewai-app",
    exporter=ConsoleSpanExporter(),
))


@node("researcher")
def researcher(task: str) -> dict:
    """Simulate a research agent."""
    return {
        "task_output": f"Research results for: {task}",
        "tools_used": ["web_search", "pdf_reader"],
        "status": "success",
    }


@node("writer")
def writer(findings: dict) -> dict:
    """Simulate a writer agent."""
    return {
        "task_output": f"Article based on: {findings['task_output']}",
        "status": "success",
    }


@endpoint("research-crew")
def run_crew(topic: str) -> dict:
    """Orchestrate the crew."""
    findings = researcher(topic)
    article = writer(findings)
    return article


result = run_crew("OpenTelemetry for LLM applications")
```

### CrewAI Decorator Reference

| Decorator | SpanKind | Use for |
|-----------|----------|---------|
| `@node("name")` | `AGENT` | Individual agents/tasks |
| `@endpoint("name")` | `CHAIN` | Crew orchestration endpoints |

The CrewAI adapter automatically normalizes `task_output`/`output` → `agent_answer`, `tools_used` → `tool_calls`, and `status` → `is_fallback` for consistent telemetry across frameworks.

---

## Custom Attributes

### Using `safe_set_attribute` (policy-enforced)

The recommended way to set custom span attributes. Respects your `AttributePolicy` for truncation, redaction, and validation:

```python
from ragwatch import trace, SpanKind, safe_set_attribute
from opentelemetry import trace as otel_trace

@trace("my-node", span_kind=SpanKind.AGENT)
def process(state):
    span = otel_trace.get_current_span()
    safe_set_attribute(span, "custom.model_name", "gpt-4")
    safe_set_attribute(span, "custom.temperature", 0.7)
    safe_set_attribute(span, "custom.tokens_used", 150)
    return {"result": "done"}
```

### Using SpanHooks (lifecycle-based)

Hooks run at `on_start`, `on_end`, and `on_error` — attach custom attributes without modifying function bodies:

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
            context.set_attribute("custom.func_name", context.func_name)
            context.set_attribute("custom.span_kind", context.span_kind.value)

    def on_end(self, span, result, *, context: InstrumentationContext = None):
        if context and isinstance(context.raw_result, dict):
            context.set_attribute("custom.result_keys", list(context.raw_result.keys()))

    def on_error(self, span, exception, *, context: InstrumentationContext = None):
        if context:
            context.set_attribute("custom.error_type", type(exception).__name__)


ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    exporter=ConsoleSpanExporter(),
    global_span_hooks=[MetricsHook()],  # Runs on every span
))

# Or attach to specific decorators:
@trace("critical-node", span_kind=SpanKind.AGENT, span_hooks=[MetricsHook()])
def critical_function(state):
    return {"answer": "done"}
```

### Using Custom TelemetryExtractors

For reusable attribute extraction logic tied to specific decorator invocations:

```python
from ragwatch import (
    configure, RAGWatchConfig, SpanKind, trace,
    InstrumentationContext, get_default_registry,
)
from opentelemetry.sdk.trace.export import ConsoleSpanExporter


class LatencyExtractor:
    name = "latency"

    def extract(self, context: InstrumentationContext) -> None:
        result = context.raw_result
        if isinstance(result, dict) and "latency_ms" in result:
            context.set_attribute("custom.latency_ms", result["latency_ms"])


ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    exporter=ConsoleSpanExporter(),
    custom_extractors=[LatencyExtractor()],
))

@trace("my-pipeline", telemetry=["latency"])  # Activate by name
def pipeline(query: str):
    return {"answer": "result", "latency_ms": 42.5}

pipeline("What is RAG?")
# Span will have: custom.latency_ms = 42.5
```

### AttributePolicy — Truncation, Redaction, and Limits

Control what gets written to spans at the policy level:

```python
from ragwatch import configure, RAGWatchConfig, AttributePolicy
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

ragwatch.configure(RAGWatchConfig(
    service_name="secure-app",
    exporter=ConsoleSpanExporter(),
    attribute_policy=AttributePolicy(
        max_value_bytes=4096,           # Truncate long strings
        max_list_length=128,            # Cap list/tuple attributes
        max_indexed_attributes=50,      # Limit chunk.N.field families
        redact_keys=["password"],       # Redact by attribute name
        redact_patterns=[r"\d{3}-\d{2}-\d{4}"],  # Redact SSNs by regex
        redact_io_keys=["password", "secret", "api_key", "token"],
    ),
))
```

---

## User Feedback

```python
from ragwatch import record_feedback

record_feedback(trace_id="abc123", score=0.85)
```

---

## Auto I/O Tracking

All decorators automatically capture function arguments as `input.value` and return values as `output.value` (4KB truncation). Disable per-decorator or globally:

```python
# Disable per-decorator
@trace("my-span", auto_track_io=False)
def my_func():
    ...

# Disable globally
ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    global_auto_track_io=False,
))
```

---

## Use with OpenLLMetry

RAGWatch complements OpenLLMetry — use both together for full-stack LLM observability:

```python
# OpenLLMetry: auto-trace LLM calls (OpenAI, Anthropic, etc.)
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()

# RAGWatch: add quality scores + structured telemetry to RAG stages
import ragwatch
ragwatch.configure(RAGWatchConfig(service_name="my-app"))
```

---

## How It Works

1. **Embedding stage** — `@trace` with `SpanKind.EMBEDDING` stores the query embedding in OTel context
2. **Retrieval stage** — `chunk_relevance_score()` reads the stored embedding and computes cosine similarity against each chunk
3. **Scores on spans** — `chunk.relevance_score` (average) and `chunk.relevance_scores` (per-chunk) are set as span attributes
4. **Adapter normalization** — framework adapters translate results into semantic keys (`agent_answer`, `tool_calls`, `routing_target`, etc.) so extractors work across frameworks

---

## API Reference

| Export | Description |
|--------|-------------|
| `configure(config)` | Initialize RAGWatch with a `RAGWatchConfig` |
| `trace(span_name, span_kind, ...)` | Decorator for tracing functions |
| `record_feedback(trace_id, score)` | Record user feedback score |
| `chunk_relevance_score(embeddings)` | Compute cosine-similarity relevance scores |
| `safe_set_attribute(span, key, val)` | Policy-enforced attribute writer |
| `InstrumentationContext` | Context object passed to extensions |
| `RAGWatchConfig` | Configuration dataclass |
| `SpanKind` | OpenInference span kind enum (`CHAIN`, `AGENT`, `TOOL`, `RETRIEVER`, `EMBEDDING`) |
| `AttributePolicy` | Truncation, redaction, and cardinality controls |
| `SpanHook` | Span lifecycle hook protocol |
| `TelemetryExtractor` | Pluggable telemetry extraction protocol |
| `FrameworkAdapter` | Framework adapter protocol |

See [`docs/EXTENSION_GUIDE.md`](docs/EXTENSION_GUIDE.md) for the full extension authoring guide and [`docs/roadmap.md`](docs/roadmap.md) for upcoming work.

---

## Development

```bash
uv sync                              # Install dependencies
uv run pytest -v                     # Run all tests
uv run pytest tests/test_tracer.py   # Run specific test file
```

## Requirements

- Python 3.11
- opentelemetry-sdk 1.24.0
- opentelemetry-api 1.24.0
- openinference-semantic-conventions 0.1.9

## License

MIT
