# RAGWatch

> Quality scores in your RAG traces — computed, not just recorded.

**RAGWatch** is an OpenTelemetry-native Python SDK that adds semantic quality scores to your RAG traces. Unlike generic tracing tools, RAGWatch computes `chunk_relevance_score` inline via cosine similarity — zero LLM calls, ~1-5 ms overhead.

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv add ragwatch                    # Core SDK
uv add ragwatch --extra langgraph  # + LangGraph adapter
uv add ragwatch --extra crewai     # + CrewAI adapter
```

## Quickstart

```python
import ragwatch
from ragwatch import RAGWatchConfig, SpanKind, trace
from ragwatch.instrumentation.evaluators import chunk_relevance_score

# Configure with your OTel exporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

ragwatch.configure(RAGWatchConfig(
    service_name="my-rag-app",
    exporter=ConsoleSpanExporter(),
))

@trace("ragwatch.embedding.generate", span_kind=SpanKind.EMBEDDING)
def embed_query(text: str) -> list[float]:
    # Your embedding API call here
    return [0.5, 0.3, 0.2]

@trace("ragwatch.retrieval.search", span_kind=SpanKind.RETRIEVER)
def retrieve_chunks(query: str) -> list[dict]:
    chunk_embeddings = [[0.5, 0.3, 0.2], [0.1, 0.9, 0.0]]
    scores = chunk_relevance_score(chunk_embeddings)
    return [{"text": "chunk", "score": s} for s in scores]

@trace("ragwatch.response.emit", span_kind=SpanKind.CHAIN)
def generate_response(chunks: list[dict]) -> str:
    return "Generated response"

# Run your pipeline
embedding = embed_query("What is RAG?")
chunks = retrieve_chunks("What is RAG?")
response = generate_response(chunks)
```

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -v

# Run specific test
uv run pytest tests/test_tracer.py -v
```

## How It Works

1. **Embedding stage**: `@trace` with `SpanKind.EMBEDDING` stores the query embedding in OTel context
2. **Retrieval stage**: `chunk_relevance_score()` reads the stored embedding and computes cosine similarity against each chunk
3. **Scores appear on spans**: `chunk.relevance_score` (average) and `chunk.relevance_scores` (per-chunk) are set as span attributes

## Framework Adapters

### LangGraph

```python
from ragwatch.adapters.langgraph import node, workflow

@node("retrieve-node")
def retrieve_node(state):
    return {**state, "docs": ["doc1"]}

@workflow("rag-pipeline")
def run_pipeline(input_data):
    return retrieve_node(input_data)
```

### CrewAI

```python
from ragwatch.adapters.crewai import node, endpoint

@node("researcher")
def researcher(task):
    return {"findings": "data"}

@endpoint("research-crew")
def run_crew(topic):
    return researcher(topic)
```

## User Feedback

```python
from ragwatch import record_feedback

record_feedback(trace_id="abc123", score=0.85)
```

## Auto I/O Tracking

All decorators automatically capture function arguments as `input.value` and return values as `output.value` (4KB truncation). Disable per-decorator:

```python
@trace("my-span", auto_track_io=False)
def my_func():
    ...
```

## Use with OpenLLMetry

RAGWatch complements OpenLLMetry — use both together:

```python
# OpenLLMetry: auto-trace LLM calls
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()

# RAGWatch: add quality scores to RAG stages
import ragwatch
ragwatch.configure(RAGWatchConfig(service_name="my-app"))
```

## API Reference

| Export | Description |
|--------|-------------|
| `configure(config)` | Initialize RAGWatch with a `RAGWatchConfig` |
| `trace(span_name, span_kind, auto_track_io)` | Decorator for tracing functions |
| `record_feedback(trace_id, score)` | Record user feedback score |
| `chunk_relevance_score(chunk_embeddings)` | Compute relevance scores |
| `RAGWatchConfig` | Configuration dataclass |
| `SpanKind` | OpenInference span kind enum |

## Requirements

- Python 3.11+
- opentelemetry-sdk 1.24.0
- opentelemetry-api 1.24.0

## License

MIT
