# RAGWatch — AGENTS.md

> **Version**: v0.3 (enterprise hardening + runtime wiring)
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
├── __init__.py              # Public API: configure, trace, record_feedback, RAGWatchConfig, SpanKind,
│                            #   TelemetryExtractor, ExtractorRegistry, SpanHook, FrameworkAdapter,
│                            #   AttributePolicy, validate_attribute_name
├── core/
│   ├── __init__.py
│   ├── config.py            # RAGWatchConfig dataclass (incl. strict_mode, global_auto_track_io)
│   ├── tracer.py            # TracerProvider setup + singleton
│   ├── runtime.py           # RAGWatchRuntime scoped accessor
│   ├── span_kinds.py        # SpanKind enum (OpenInference standard)
│   └── context.py           # Thread-local context for query embedding
├── instrumentation/
│   ├── __init__.py
│   ├── decorators.py        # @trace (sync + async, orchestration + failure isolation)
│   ├── extractors.py        # TelemetryExtractor Protocol + ExtractorRegistry + 5 built-in extractors
│   ├── span_hooks.py        # SpanHook Protocol + global/local hook lifecycle + on_error
│   ├── context_model.py     # InstrumentationContext dataclass (passed to hooks)
│   ├── attributes.py        # safe_set_attribute() centralized writer with policy enforcement
│   ├── result_transformers.py # ResultTransformer Protocol + Registry + built-in RETRIEVER/TOOL logic
│   ├── token_usage.py       # TokenExtractor Protocol + registry + built-in usage_metadata scanner
│   ├── attribute_policy.py  # Attribute naming validation, truncation, redaction
│   ├── io_tracker.py        # Auto input.value/output.value with 4KB truncation (policy-driven)
│   ├── semconv.py           # GenAI SemConv strings. SEMCONV_VERSION = "v1.40"
│   ├── helpers.py           # record_chunks, record_routing, etc.
│   ├── embedding.py         # SpanKind.EMBEDDING + context set
│   └── evaluators.py        # chunk_relevance_score() + record_feedback()
└── adapters/
    ├── __init__.py
    ├── base.py              # FrameworkAdapter Protocol + adapter registry
    ├── langgraph/
    │   ├── __init__.py      # from ragwatch.adapters.langgraph import node, workflow, LangGraphAdapter
    │   ├── adapter.py       # LangGraphAdapter (FrameworkAdapter impl)
    │   └── decorators.py    # @node (AGENT), @workflow (CHAIN), @tool (TOOL)
    └── crewai/
        ├── __init__.py      # from ragwatch.adapters.crewai import node, endpoint, CrewAIAdapter
        ├── adapter.py       # CrewAIAdapter (FrameworkAdapter impl)
        └── decorators.py    # @node (AGENT), @endpoint (CHAIN)

examples/
├── minimal_rag.py
├── langgraph_agent.py
├── crewai_agent.py
├── custom_extensions.py     # All v0.3 extension points demo
└── README.md

tests/
├── conftest.py
├── test_tracer.py
├── test_evaluators.py
├── test_context.py
├── test_io_tracker.py
├── test_decorators.py
├── test_extractors.py       # Plugin registry + built-in extractors
├── test_span_hooks.py       # SpanHook lifecycle (local + global)
├── test_adapter_contracts.py # FrameworkAdapter Protocol compliance + runtime wiring
├── test_attribute_policy.py  # Schema validation + redaction + truncation + safe_set_attribute
├── test_failure_isolation.py # Failure isolation for hooks/extractors/transformers + strict_mode
├── test_context_model.py    # InstrumentationContext + on_error hooks
├── test_pluggable_transforms.py # ResultTransformer + TokenExtractor + global_auto_track_io
├── test_runtime.py          # RAGWatchRuntime scoped accessor
├── test_adapters_langgraph.py
├── test_adapters_crewai.py
├── test_integration.py
└── test_semconv.py
```

## Public API Surface

```python
# ragwatch/__init__.py
from ragwatch import (
    configure, trace, record_feedback, RAGWatchConfig, SpanKind,
    # Plugin system
    TelemetryExtractor, ExtractorRegistry, get_default_registry,
    # Span hooks
    SpanHook, register_global_hook,
    # Adapter contracts
    FrameworkAdapter, register_adapter,
    # Attribute policy + centralized writer
    AttributePolicy, validate_attribute_name,
    safe_set_attribute, safe_set_attributes, get_active_config,
    # Runtime
    RAGWatchRuntime,
)

# ragwatch/adapters/langgraph/
from ragwatch.adapters.langgraph import node, workflow, LangGraphAdapter

# ragwatch/adapters/crewai/
from ragwatch.adapters.crewai import node, endpoint, CrewAIAdapter
```

### Core Exports

- `configure(config: RAGWatchConfig)` — TracerProvider setup; users bring their own exporter
- `trace(span_name, span_kind, auto_track_io, telemetry, result_formatter, span_hooks, adapter)` — sync + async decorator
- `record_feedback(trace_id, score)` — `user.feedback_score` on response span
- `RAGWatchConfig` — configuration dataclass (incl. `custom_extractors`, `global_span_hooks`, `adapters`, `attribute_policy`, `strict_mode`, `global_auto_track_io`)
- `SpanKind` — OpenInference enum (CHAIN, AGENT, TOOL, RETRIEVER, EMBEDDING)

### Extensibility (v0.2 → v0.3)

- **`TelemetryExtractor`** — Protocol for pluggable telemetry extraction. Register via `get_default_registry().register()` or `configure(custom_extractors=[...])`.
- **`SpanHook`** — Protocol with `on_start(span, args, kwargs)` / `on_end(span, result)` / `on_error(span, exception)`. Hooks may accept an optional `context=` kwarg to receive `InstrumentationContext`. Per-decorator via `@trace(span_hooks=[...])` or global via `configure(global_span_hooks=[...])`.
- **`FrameworkAdapter`** — Protocol with `extract_state(args, kwargs)` / `default_extractors()`. Register via `configure(adapters=[...])`. Wired at runtime via `@trace(adapter="name")`.
- **`AttributePolicy`** — Truncation (`max_value_bytes`), pattern-based redaction (`redact_patterns`), key-based redaction (`redact_keys`).
- **`validate_attribute_name(name)`** — Enforces dot-separated lowercase namespace convention, max 128 chars.
- **`safe_set_attribute()` / `safe_set_attributes()`** — Centralized attribute writer; enforces active `AttributePolicy` on every write.
- **`InstrumentationContext`** — Dataclass passed to hooks with span, args, kwargs, adapter, raw_result, result, exception.
- **`ResultTransformer`** — Protocol for pluggable result transformation per `SpanKind`. Register via `get_default_transformer_registry().register()`.
- **`TokenExtractor`** — Protocol for pluggable token-usage extraction. Register via `register_token_extractor()`.
- **`RAGWatchRuntime`** — Scoped read-only accessor for active config, registries, and policy. `RAGWatchRuntime.current()`.

### Auto I/O Tracking

Default ON. Captures args → `input.value`, return → `output.value`. 4KB truncation. Disable per-decorator: `@trace(..., auto_track_io=False)`. Disable globally: `RAGWatchConfig(global_auto_track_io=False)`.

### Failure Isolation

Extension errors (hooks, extractors, transformers) are caught, logged, and recorded as span events. Enable `strict_mode=True` in config to re-raise instead (for dev/test).

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
6. **2 adapters in v0.1** — LangGraph, CrewAI; Protocol-based contracts in v0.2
7. **OTel-native positioning** — vs Langfuse/LangSmith: zero vendor lock-in
8. **OpenInference span kinds** — open standard
9. **Auto I/O tracking** — default ON; 4KB cap; disable per-decorator
10. **Semantic quality scores** — computed, not just recorded
11. **Plugin architecture (v0.2)** — `TelemetryExtractor` registry replaces hardcoded switchboard
12. **Span lifecycle hooks (v0.2)** — `SpanHook` protocol for custom attribute enrichment
13. **Adapter contracts (v0.2)** — `FrameworkAdapter` protocol formalises framework integrations
14. **Attribute policy (v0.2)** — Validation, truncation, and redaction for enterprise safety
15. **Adapter runtime wiring (v0.3)** — `@trace(adapter=...)` drives `extract_state()` at call time
16. **Centralized attribute writer (v0.3)** — `safe_set_attribute()` enforces `AttributePolicy` on every write
17. **Failure isolation (v0.3)** — Extension errors caught, logged, span-evented; `strict_mode` for dev
18. **InstrumentationContext (v0.3)** — Unified context object threaded through all extension points
19. **on_error hooks (v0.3)** — `SpanHook.on_error(span, exception)` called on decorated-function exceptions
20. **Pluggable result transformation (v0.3)** — `ResultTransformer` protocol + per-SpanKind registry
21. **Pluggable token extraction (v0.3)** — `TokenExtractor` protocol + registry
22. **Policy-driven I/O (v0.3)** — `global_auto_track_io` config toggle; all writes via `safe_set_attribute`
23. **RAGWatchRuntime (v0.3)** — Scoped accessor reducing reliance on module-level globals
