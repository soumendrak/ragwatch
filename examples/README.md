# RAGWatch Examples

## Prerequisites

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Examples

### Minimal Linear RAG

Demonstrates the core embedding → retrieval → response pipeline with `chunk_relevance_score`.

```bash
uv run python examples/minimal_rag.py
```

### LangGraph Agent

Shows how to use `@node` and `@workflow` decorators for LangGraph-style agents.

```bash
uv add ragwatch --extra langgraph
uv run python examples/langgraph_agent.py
```

### CrewAI Agent

Shows how to use `@node` and `@endpoint` decorators for CrewAI-style agents.

```bash
uv add ragwatch --extra crewai
uv run python examples/crewai_agent.py
```

## Viewing Traces

All examples use `ConsoleSpanExporter` by default. To send traces to Jaeger:

```python
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    exporter=JaegerExporter(agent_host_name="localhost", agent_port=6831),
))
```

Then open Jaeger UI at `http://localhost:16686`.
