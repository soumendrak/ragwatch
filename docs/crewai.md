# CrewAI Integration

Instrument your CrewAI workflows with RAGWatch for structured telemetry and full trace visibility.

---

## Installation

```bash
pip install ragwatch[crewai]
# or
uv add ragwatch --extra crewai
```

---

## Decorators

RAGWatch provides two decorators for CrewAI workflows:

| Decorator | SpanKind | Use for |
|-----------|----------|---------|
| `@node("name")` | `AGENT` | Individual agents or tasks |
| `@endpoint("name")` | `CHAIN` | Crew orchestration endpoints |

Import from `ragwatch.adapters.crewai`:

```python
from ragwatch.adapters.crewai import node, endpoint
```

Each decorator automatically:
- Sets the correct `SpanKind`
- Wires up the CrewAI adapter (`adapter="crewai"`)
- Captures `input.value` and `output.value` (auto I/O tracking)
- Normalizes CrewAI-specific result shapes into semantic keys

---

## Full Example

A multi-agent CrewAI workflow with RAGWatch instrumentation:

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.crewai import node, endpoint


ragwatch.configure(RAGWatchConfig(
    service_name="crewai-research-crew",
    exporter=ConsoleSpanExporter(),
))


@node("researcher")
def researcher(task: str) -> dict:
    """Research agent that gathers information."""
    return {
        "task_output": f"Research findings for: {task}",
        "tools_used": ["web_search", "pdf_reader"],
        "sources": ["paper1.pdf", "paper2.pdf"],
        "status": "success",
    }


@node("analyst")
def analyst(findings: dict) -> dict:
    """Analysis agent that processes research findings."""
    return {
        "task_output": f"Analysis of: {findings['task_output']}",
        "tools_used": ["calculator", "chart_generator"],
        "status": "success",
    }


@node("writer")
def writer(analysis: dict) -> dict:
    """Writer agent that produces the final output."""
    return {
        "task_output": f"Report based on: {analysis['task_output']}",
        "status": "success",
    }


@endpoint("research-crew")
def run_crew(topic: str) -> dict:
    """Orchestrate the research crew."""
    findings = researcher(topic)
    analysis = analyst(findings)
    report = writer(analysis)
    return report


result = run_crew("OpenTelemetry for LLM observability")
print(f"Report: {result['task_output']}")
```

---

## Automatic Result Normalization

The CrewAI adapter automatically translates CrewAI-specific result shapes into RAGWatch's semantic keys:

| CrewAI Result Key | Semantic Key | Description |
|-------------------|-------------|-------------|
| `task_output` or `output` | `agent_answer` | The agent's answer text |
| `tools_used` | `tool_calls` | List of tools the agent invoked |
| `status` | `is_fallback` | `True` if status indicates failure/error |

This normalization means extractors work identically across LangGraph and CrewAI — the framework-specific differences are handled by the adapter.

### Dict Results

```python
@node("researcher")
def researcher(task: str) -> dict:
    return {
        "task_output": "Research findings here",
        "tools_used": ["web_search"],
        "status": "success",
    }
    # Normalized to: agent_answer="Research findings here",
    #                tool_calls=["web_search"], is_fallback=False
```

### TaskOutput-like Objects

The adapter also handles CrewAI's `TaskOutput` objects (any object with `.raw` or `.output` attributes):

```python
@node("researcher")
def researcher(task: str):
    result = crew_task.execute()  # Returns TaskOutput
    return result
    # Adapter reads result.raw or result.output for agent_answer
```

### Fallback Detection

The adapter detects failure states from the `status` field:

```python
# These statuses trigger is_fallback=True:
{"status": "error"}
{"status": "failed"}
{"status": "timeout"}

# These do not:
{"status": "success"}
{"status": "completed"}
```

---

## Decorator Parameters

Both decorators accept these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `span_name` | `str` | Function name | Custom span name |
| `auto_track_io` | `bool` | `True` | Capture input/output as span attributes |
| `span_hooks` | `list[SpanHook]` | `None` | Per-decorator span lifecycle hooks |

### Usage Patterns

```python
# With explicit name
@node("researcher-agent")
def researcher(task): ...

# Without parentheses (uses function name as span name)
@node
def researcher(task): ...

# With options
@node("researcher", auto_track_io=False)
def researcher(task): ...

# Endpoint patterns work the same way
@endpoint("research-crew")
def run_crew(topic): ...

@endpoint
def run_crew(topic): ...
```

---

## Adding Custom Attributes

Use `SpanHook` or `safe_set_attribute` to add crew-specific metadata:

```python
from ragwatch import safe_set_attribute, InstrumentationContext
from opentelemetry import trace as otel_trace


class CrewMetricsHook:
    def on_end(self, span, result, *, context: InstrumentationContext = None):
        if context and isinstance(context.raw_result, dict):
            sources = context.raw_result.get("sources", [])
            context.set_attribute("crew.source_count", len(sources))
            context.set_attribute("crew.sources", sources)


ragwatch.configure(RAGWatchConfig(
    service_name="crewai-app",
    exporter=ConsoleSpanExporter(),
    global_span_hooks=[CrewMetricsHook()],
))
```

Or inline within a node:

```python
@node("researcher")
def researcher(task: str) -> dict:
    span = otel_trace.get_current_span()
    safe_set_attribute(span, "crew.task_type", "research")
    safe_set_attribute(span, "crew.priority", "high")
    return {"task_output": "findings", "status": "success"}
```

---

## Next Steps

- [Custom Attributes](custom-attributes.md) — more ways to enrich your spans
- [Configuration](configuration.md) — `AttributePolicy` for redaction and truncation
- [Extension Guide](EXTENSION_GUIDE.md) — writing your own adapter
