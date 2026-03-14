# Quickstart

Get RAGWatch running in under 5 minutes.

---

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

## Configure RAGWatch

RAGWatch needs an OpenTelemetry span exporter. You bring the backend — Jaeger, Phoenix, Grafana Tempo, or any OTel-compatible collector.

```python
import ragwatch
from ragwatch import RAGWatchConfig
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

ragwatch.configure(RAGWatchConfig(
    service_name="my-rag-app",
    exporter=ConsoleSpanExporter(),  # Replace with your backend exporter
))
```

### Multiple Backends

Send traces to multiple backends simultaneously:

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

## Your First Traced Pipeline

A minimal 3-stage RAG pipeline with computed relevance scores:

```python
import ragwatch
from ragwatch import RAGWatchConfig, SpanKind, trace
from ragwatch.instrumentation.evaluators import chunk_relevance_score
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

ragwatch.configure(RAGWatchConfig(
    service_name="my-rag-app",
    exporter=ConsoleSpanExporter(),
))


# Stage 1: Generate embedding
@trace("embedding.generate", span_kind=SpanKind.EMBEDDING)
def embed_query(text: str) -> list[float]:
    # Replace with your embedding API (OpenAI, Cohere, etc.)
    return [0.5, 0.3, 0.2]


# Stage 2: Retrieve chunks with relevance scores
@trace("retrieval.search", span_kind=SpanKind.RETRIEVER)
def retrieve_chunks(query: str) -> list[dict]:
    # Replace with your vector DB search
    chunk_embeddings = [[0.5, 0.3, 0.2], [0.1, 0.9, 0.0], [0.4, 0.4, 0.1]]

    # RAGWatch computes cosine similarity automatically
    scores = chunk_relevance_score(chunk_embeddings)

    return [
        {"text": "RAG combines retrieval with generation.", "score": scores[0]},
        {"text": "Transformers use self-attention.", "score": scores[1]},
        {"text": "Vector databases store embeddings.", "score": scores[2]},
    ]


# Stage 3: Generate response
@trace("response.generate", span_kind=SpanKind.CHAIN)
def generate_response(chunks: list[dict]) -> str:
    top_chunk = max(chunks, key=lambda c: c["score"])
    return f"Based on: {top_chunk['text']}"


# Run the pipeline
embedding = embed_query("What is RAG?")
chunks = retrieve_chunks("What is RAG?")
response = generate_response(chunks)

print(response)
```

### What Gets Traced

Each `@trace` decorator creates an OpenTelemetry span with:

| Attribute | Description |
|-----------|-------------|
| `openinference.span.kind` | `EMBEDDING`, `RETRIEVER`, or `CHAIN` |
| `input.value` | Function arguments (auto-captured, 4KB truncation) |
| `output.value` | Return value (auto-captured, 4KB truncation) |
| `chunk.relevance_score` | Average cosine similarity (on retriever spans) |
| `chunk.relevance_scores` | Per-chunk similarity scores (on retriever spans) |

---

## Record User Feedback

Link user satisfaction scores back to traces:

```python
from ragwatch import record_feedback

# After user rates the response
record_feedback(trace_id="abc123", score=0.85)
```

This creates a `ragwatch.feedback` span with `user.feedback_score` and `user.feedback_trace_id` attributes.

---

## Disable Auto I/O Tracking

By default, RAGWatch captures function arguments and return values. Disable when handling sensitive data:

```python
# Per-decorator
@trace("my-span", auto_track_io=False)
def sensitive_function(data):
    ...

# Globally
ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    global_auto_track_io=False,
))
```

---

## Use with OpenLLMetry

RAGWatch handles RAG-specific instrumentation. For LLM call tracing, pair it with [OpenLLMetry](https://github.com/traceloop/openllmetry):

```python
# OpenLLMetry: auto-trace LLM calls (OpenAI, Anthropic, etc.)
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()

# RAGWatch: add quality scores + structured telemetry to RAG stages
import ragwatch
from ragwatch import RAGWatchConfig
ragwatch.configure(RAGWatchConfig(service_name="my-app"))
```

Both emit spans to the same OTel backend, giving you full-stack observability.

---

## Next Steps

- [LangGraph Integration](langgraph.md) — instrument LangGraph applications
- [CrewAI Integration](crewai.md) — instrument CrewAI workflows
- [Quality Scores](quality-scores.md) — deep dive into relevance scoring
- [Custom Attributes](custom-attributes.md) — add your own span attributes
- [Configuration](configuration.md) — `AttributePolicy`, strict mode, and more
