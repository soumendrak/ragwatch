# LangGraph Integration

Instrument your LangGraph applications with RAGWatch for structured telemetry, relevance scoring, and full trace visibility.

---

## Installation

```bash
pip install ragwatch[langgraph]
# or
uv add ragwatch --extra langgraph
```

---

## Decorators

RAGWatch provides three decorators that map directly to LangGraph concepts:

| Decorator | SpanKind | Use for |
|-----------|----------|---------|
| `@node("name")` | `AGENT` | Graph nodes (retrieval, generation, routing, etc.) |
| `@workflow("name")` | `CHAIN` | Workflow orchestrators, graph builders |
| `@tool("name")` | `TOOL` | Tool implementations called by agents |

All decorators are imported from `ragwatch.adapters.langgraph`:

```python
from ragwatch.adapters.langgraph import node, workflow, tool
```

Each decorator automatically:
- Sets the correct `SpanKind`
- Wires up the LangGraph adapter (`adapter="langgraph"`)
- Captures `input.value` and `output.value` (auto I/O tracking)
- Supports `telemetry=[...]` for structured extraction

---

## Full Example

A complete LangGraph RAG pipeline with RAGWatch instrumentation:

```python
from typing import TypedDict
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from langgraph.graph import StateGraph, END

import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.langgraph import node, workflow
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


# --- Embedding Node ---
@node("embed-node")
def embed_node(state: RAGState) -> RAGState:
    """Generate query embedding."""
    embedding = [0.5, 0.3, 0.2]  # Replace with your embedding API
    return {**state, "embedding": embedding}


# --- Retrieval Node ---
@node("retrieve-node")
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve chunks with computed relevance scores."""
    chunk_embeddings = [
        [0.5, 0.3, 0.2],  # Very similar
        [0.1, 0.9, 0.0],  # Less similar
        [0.4, 0.4, 0.1],  # Somewhat similar
    ]
    scores = chunk_relevance_score(chunk_embeddings)
    docs = [
        {"text": "RAG combines retrieval with generation.", "score": scores[0]},
        {"text": "Transformers use self-attention.", "score": scores[1]},
        {"text": "Vector databases store embeddings.", "score": scores[2]},
    ]
    return {**state, "docs": docs}


# --- Generation Node ---
@node("generate-node")
def generate_node(state: RAGState) -> RAGState:
    """Generate response from the top-scoring chunk."""
    top_doc = max(state["docs"], key=lambda d: d["score"])
    return {**state, "response": f"Based on: {top_doc['text']}"}


# --- Build and Run the Graph ---
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

print(f"Response: {result['response']}")
```

---

## Telemetry Extraction

Pass `telemetry=[...]` to any LangGraph decorator to automatically extract structured telemetry from the node's return value. No code changes inside the function body — RAGWatch inspects the return dict.

### Available Strategies

| Strategy | What it extracts | Return value keys inspected |
|----------|-----------------|----------------------------|
| `tool_calls` | LLM tool-call decisions | `messages` (looks for `tool_calls` on AI messages) |
| `routing` | Routing/edge decisions | `routing_target`, `routing_reason`, or `Command` objects |
| `agent_completion` | Final answer, fallback detection | `final_answer`, `agent_answers` |
| `query_rewrite` | Query decomposition | `rewrittenQuestions`, `questionIsClear`, `originalQuery` |
| `compression` | Context compression stats | `context_summary` + state's `messages`, `retrieval_keys` |

### Examples

#### Tool Calls

```python
@node("orchestrator", telemetry=["tool_calls"])
def orchestrator(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
    # RAGWatch extracts: tool names, args, IDs from response.tool_calls
```

#### Agent Completion

```python
@node("answer-node", telemetry=["agent_completion"])
def collect_answer(state):
    return {
        "final_answer": "The answer is 42",
        "agent_answers": [{"answer": "42", "question": "What is the meaning?"}],
    }
    # RAGWatch extracts: agent_answer, is_fallback
```

#### Query Rewrite

```python
@node("rewrite-node", telemetry=["query_rewrite"])
def rewrite_query(state):
    return {
        "rewrittenQuestions": ["What is total revenue?", "What is profit margin?"],
        "questionIsClear": False,
        "originalQuery": "Tell me about the financials",
    }
    # RAGWatch extracts: rewritten_questions, is_clear, original_query
```

#### Routing

```python
@node("router-node", telemetry=["routing"])
def route_decision(state):
    if needs_more_context(state):
        return Command(goto="retrieve", update=state)
    return Command(goto="generate", update=state)
    # RAGWatch extracts: routing_target, routing_reason
```

#### Context Compression

```python
@node("compress-node", telemetry=["compression"])
def compress_context(state):
    summary = summarize(state["messages"])
    return {**state, "context_summary": summary}
    # RAGWatch extracts: compression_tokens_before/after, queries_run, parents_retrieved
```

### Combining Strategies

```python
@node("orchestrator", telemetry=["tool_calls", "routing", "agent_completion"])
def orchestrator(state):
    # All three extractors run on the same return value
    ...
```

---

## Using `@tool`

Wrap tool implementations that are called by LangGraph agents:

```python
from ragwatch.adapters.langgraph import tool

@tool("search-documents")
def search_documents(query: str, limit: int = 5) -> list[dict]:
    results = vector_db.search(query, top_k=limit)
    return [{"text": r.text, "score": r.score} for r in results]

@tool("fetch-parent-chunk", result_formatter=format_chunk)
def fetch_parent_chunk(parent_id: str) -> dict:
    return db.get_chunk(parent_id)
```

---

## Decorator Parameters

All LangGraph decorators accept these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `span_name` | `str` | Function name | Custom span name |
| `auto_track_io` | `bool` | `True` | Capture input/output as span attributes |
| `telemetry` | `list[str]` | `None` | Extraction strategies to activate |
| `result_formatter` | `Callable` | `None` | Custom result formatting function |
| `span_hooks` | `list[SpanHook]` | `None` | Per-decorator span lifecycle hooks |

---

## Telemetry Helpers

For cases where you want explicit control over what gets recorded (instead of automatic extraction), use the telemetry helper functions inside your node:

```python
from ragwatch import record_tool_calls, record_routing, record_agent_completion

@node("orchestrator")
def orchestrator(state):
    response = llm.invoke(state["messages"])

    # Explicitly record tool calls
    record_tool_calls(response.tool_calls)

    # Explicitly record routing decision
    record_routing(
        from_node="orchestrator",
        to_node="retrieve" if response.tool_calls else "answer",
        reason=f"Tool calls: {len(response.tool_calls)}",
    )

    return {"messages": [response]}
```

Available helpers: `record_chunks`, `record_tool_calls`, `record_routing`, `record_agent_completion`, `record_context_compression`, `record_query_rewrite`.

See [Telemetry Extraction](telemetry-extraction.md) for full documentation of each helper.

---

## Next Steps

- [Quality Scores](quality-scores.md) — how `chunk_relevance_score` works
- [Custom Attributes](custom-attributes.md) — add your own span attributes
- [Telemetry Extraction](telemetry-extraction.md) — all built-in extractors and helpers
