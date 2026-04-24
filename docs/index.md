# RAGWatch Documentation

> Quality scores in your RAG traces — computed, not just recorded.

**RAGWatch** is an OpenTelemetry-native Python SDK that adds semantic quality scores to your RAG traces. Unlike generic tracing tools, RAGWatch computes `chunk_relevance_score` inline via cosine similarity — zero LLM calls, ~1-5 ms overhead.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Quickstart](quickstart.md) | Installation, configuration, and your first traced RAG pipeline |
| [LangGraph Integration](langgraph.md) | Instrument LangGraph applications with `@node`, `@workflow`, `@tool` |
| [CrewAI Integration](crewai.md) | Instrument CrewAI workflows with `@node`, `@endpoint` |
| [Quality Scores](quality-scores.md) | `chunk_relevance_score`, user feedback, and how scoring works |
| [Custom Attributes](custom-attributes.md) | `safe_set_attribute`, SpanHooks, and custom TelemetryExtractors |
| [Telemetry Extraction](telemetry-extraction.md) | Built-in extractors, telemetry helpers, and writing your own |
| [Configuration](configuration.md) | `RAGWatchConfig`, `AttributePolicy`, strict mode, auto I/O tracking |
| [Extension Guide](EXTENSION_GUIDE.md) | Full guide to all extension protocols and adapter contracts |
| [API Reference](api-reference.md) | Complete public API surface |
| [Roadmap](roadmap.md) | Planned runtime, schema, and adapter work |
| [Production Cookbook](production-cookbook.md) | Production LangGraph, CrewAI, and feedback patterns |

---

## Architecture at a Glance

```
Your Application
    │
    ├── @trace / @node / @workflow / @endpoint
    │       │
    │       ├── Auto I/O tracking (input.value, output.value)
    │       ├── SpanHooks (on_start, on_end, on_error)
    │       ├── Adapter normalization (framework → semantic keys)
    │       ├── TelemetryExtractors (tool_calls, routing, etc.)
    │       └── ResultTransformers + TokenExtractors
    │
    ├── chunk_relevance_score()  →  cosine similarity on spans
    ├── record_feedback()        →  user.feedback_score
    │
    └── OpenTelemetry SDK  →  Your Backend (Jaeger, Phoenix, etc.)
```

### Key Concepts

- **Explicit decorators only** — RAGWatch uses decorators (`@trace`, `@node`, etc.) for instrumentation. For LLM call auto-instrumentation, use [OpenLLMetry](https://github.com/traceloop/openllmetry).
- **Users bring their own OTel backend** — RAGWatch is an SDK, not a platform. Export spans to Jaeger, Phoenix, Grafana Tempo, or any OTel-compatible backend.
- **Policy-enforced attributes** — All attribute writes go through `safe_set_attribute()`, which applies truncation, redaction, and naming validation via `AttributePolicy`.
- **Framework adapters** — LangGraph and CrewAI adapters normalize framework-specific results into semantic keys (`agent_answer`, `tool_calls`, `routing_target`, etc.) so extractors work across frameworks.

### Span Kinds (OpenInference Standard)

| SpanKind | Use for |
|----------|---------|
| `CHAIN` | Root spans, orchestration, pipelines |
| `AGENT` | Nodes, pipeline steps, agent tasks |
| `TOOL` | Tool implementations, shared functions |
| `RETRIEVER` | Vector search, retrieval |
| `EMBEDDING` | Embedding generation |

---

## Requirements

- Python 3.11
- opentelemetry-sdk 1.24.0
- opentelemetry-api 1.24.0
- openinference-semantic-conventions 0.1.9

## License

MIT
