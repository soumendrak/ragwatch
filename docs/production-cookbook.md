# Production Cookbook

Practical patterns for using RAGWatch in production RAG and agent workflows.

## LangGraph RAG Workflow

Use the built-in decorators directly. The LangGraph adapter is resolved
automatically; users do not need to pass `adapters=[LangGraphAdapter()]` for the
built-in integration.

```python
import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.langgraph import node, workflow, tool


ragwatch.configure(RAGWatchConfig(service_name="rag-api"))


@node("rewrite-query", telemetry=["query_rewrite"])
def rewrite_query(state: dict) -> dict:
    return {
        **state,
        "originalQuery": state["query"],
        "rewrittenQuestions": [state["query"]],
        "questionIsClear": True,
    }


@node("retrieve", telemetry=["compression"])
def retrieve(state: dict) -> dict:
    return {**state, "context_summary": "short retrieval context"}


@tool("load-parent")
def load_parent(parent_id: str) -> dict:
    return {
        "parent_id": parent_id,
        "content": "Parent document text",
        "metadata": {"source": "docs.md"},
    }


@workflow("rag-workflow")
def run_workflow(query: str) -> dict:
    state = rewrite_query({"query": query, "messages": []})
    return retrieve(state)
```

## CrewAI Endpoint

CrewAI normalization maps common result keys to RAGWatch semantic telemetry:
`task_output` or `output` becomes `agent_answer`, `status` contributes fallback
detection, and `tools_used` becomes `tool_calls`.

```python
import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.crewai import endpoint, node


ragwatch.configure(RAGWatchConfig(service_name="research-crew"))


@node("researcher")
def researcher(topic: str) -> dict:
    return {
        "task_output": f"Research for {topic}",
        "tools_used": ["web_search"],
        "status": "success",
    }


@endpoint("crew-run")
def run_crew(topic: str) -> dict:
    return researcher(topic)
```

## Feedback Correlation

For attribute-only correlation, pass `trace_id`. For a real OpenTelemetry span
link, pass both `trace_id` and `span_id` as hex identifiers.

```python
from ragwatch import record_feedback


record_feedback(
    trace_id="00000000000000000000000000000001",
    span_id="0000000000000002",
    score=0.92,
)
```

## Production Defaults

- Set `AttributePolicy.redact_io_keys` for credentials and user identifiers.
- Disable `global_auto_track_io` for highly sensitive workloads.
- Use `strict_mode=True` in development and test environments.
- Keep custom extensions context-first: `extract(context)`, `transform(context)`,
  and hooks with `context=`.
