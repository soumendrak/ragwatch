# RAGWatch — AGENTS.md

> **Version**: v0.1 (slimmed scope)
> **Tagline**: "Quality scores in your RAG traces — computed, not just recorded"

## Project Overview

- **Name**: RAGWatch
- **Purpose**: OpenTelemetry-native RAG observability Python SDK with semantic quality scores
- **Key Differentiator**: `chunk_relevance_score` + `user.feedback_score` — **computed** semantic quality
- **Instrumentation**: Explicit decorators only (use OpenLLMetry for auto-instrumentation)
- **RAG Types**: Linear RAG (v0.1), multi-stage via composed decorators
- **Efficiency**: ~1-5 ms overhead per request
- **Scope**: SDK only. Users bring their own OTel backend.
- **Install**: `uv add ragwatch` (recommended) or `pip install ragwatch` (core); `uv add ragwatch --extra langgraph` or `pip install ragwatch[langgraph]`

## Architecture

### Instrumentation Strategy

**Explicit decorators only** — no auto-instrumentation in v0.1.

| Layer | Instrumentation | Reason |
|-------|-----------------|--------|
| LLM calls | **None** (use OpenLLMetry) | They do it better; compose via OTel |
| Embedding generation | **Explicit** | Query embedding stored for relevance score |
| Vector search / retrieval | **Explicit** | `chunk_relevance_score` computed here |
| User feedback | **Explicit** | `record_feedback()` is user-initiated |
| Agent nodes / workflows | **Explicit** | User decides span boundaries |

### Linear RAG Stages

| Stage | Span Name | Key Attributes |
|-------|-----------|----------------|
| Embedding | `ragwatch.embedding.generate` | model.name, embedding.dimensions, duration_ms |
| Vector Search | `ragwatch.retrieval.search` | top_k, chunks.returned, **chunk.relevance_score** |
| Response | `ragwatch.response.emit` | response.length, **user.feedback_score** |

### `chunk_relevance_score` Plumbing (single-process only)

1. `embedding.py` computes query embedding → stores in OTel `Context` (not baggage)
2. `context.py` manages thread-local storage for query embedding
3. `retrieval.py` reads context → computes cosine similarity → sets `chunk.relevance_score`
4. **Supported dimensions**: up to 512-dim

### Span Kinds (OpenInference standard)

- **CHAIN**: Root spans, orchestration
- **AGENT**: Nodes, pipeline steps
- **TOOL**: Shared functions
- **RETRIEVER**: Vector search
- **EMBEDDING**: Embedding generation

## Repository Structure

```
ragwatch/
├── __init__.py              # Public API: configure, trace, record_feedback, RAGWatchConfig, SpanKind
├── core/
│   ├── __init__.py
│   ├── config.py            # RAGWatchConfig dataclass
│   ├── tracer.py            # TracerProvider setup + singleton
│   ├── span_kinds.py        # SpanKind enum (OpenInference standard)
│   └── context.py           # Thread-local context for query embedding
├── instrumentation/
│   ├── __init__.py
│   ├── decorators.py        # @trace (sync + async, any function)
│   ├── io_tracker.py        # Auto input.value/output.value with 4KB truncation
│   ├── semconv.py           # GenAI SemConv strings. SEMCONV_VERSION = "v1.40"
│   ├── embedding.py         # SpanKind.EMBEDDING + context set
│   ├── retrieval.py         # SpanKind.RETRIEVER + chunk_relevance_score
│   └── evaluators.py        # chunk_relevance_score() + record_feedback()
└── adapters/
    ├── __init__.py
    ├── langgraph/
    │   ├── __init__.py      # from ragwatch.adapters.langgraph import node, workflow
    │   └── decorators.py    # @node (AGENT), @workflow (CHAIN)
    └── crewai/
        ├── __init__.py      # from ragwatch.adapters.crewai import node, endpoint
        └── decorators.py    # @node (AGENT), @endpoint (CHAIN)

examples/
├── minimal_rag.py
├── langgraph_agent.py
├── crewai_agent.py
└── README.md

tests/
├── conftest.py
├── test_tracer.py
├── test_evaluators.py
├── test_context.py
├── test_io_tracker.py
├── test_decorators.py
├── test_adapters_langgraph.py
├── test_adapters_crewai.py
├── test_integration.py
└── test_semconv.py
```

## Public API Surface

```python
# ragwatch/__init__.py
from ragwatch import configure, trace, record_feedback, RAGWatchConfig, SpanKind

# ragwatch/adapters/langgraph/
from ragwatch.adapters.langgraph import node, workflow

# ragwatch/adapters/crewai/
from ragwatch.adapters.crewai import node, endpoint
```

### Core Exports

- `configure(config: RAGWatchConfig)` — TracerProvider setup; users bring their own exporter
- `trace(span_name, span_kind=SpanKind.CHAIN, auto_track_io=True)` — sync + async decorator
- `record_feedback(trace_id, score)` — `user.feedback_score` on response span
- `RAGWatchConfig` — configuration dataclass
- `SpanKind` — OpenInference enum (CHAIN, AGENT, TOOL, RETRIEVER, EMBEDDING)

### Auto I/O Tracking

Default ON. Captures args → `input.value`, return → `output.value`. 4KB truncation. Disable: `@trace(..., auto_track_io=False)`.

All decorators work on both `async def` and `def`.

## Tech Stack

| Layer | Choice | Version |
|-------|--------|---------|
| Language | Python | 3.11+ |
| OTel SDK | opentelemetry-sdk | **pinned 1.24.0** |
| OTel API | opentelemetry-api | **pinned 1.24.0** |
| Packaging | pyproject.toml | — |

## Key Design Decisions

1. **Explicit decorators only** — No auto-instrumentation; use OpenLLMetry for that
2. **semconv.py as single source of truth** — AST-based CI guard
3. **Quality layer built-in** — `chunk_relevance_score` via Context — core differentiator
4. **Users provide exporter** — SDK doesn't bundle OTel backend
5. **Span links for non-linear flows** — Agent loops/branches use links
6. **2 adapters in v0.1** — LangGraph, CrewAI
7. **OTel-native positioning** — vs Langfuse/LangSmith: zero vendor lock-in
8. **OpenInference span kinds** — open standard
9. **Auto I/O tracking** — default ON; 4KB cap; disable per-decorator
10. **Semantic quality scores** — computed, not just recorded
