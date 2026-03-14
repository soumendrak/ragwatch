# Telemetry Extraction

RAGWatch provides two approaches for recording structured telemetry on spans: **automatic extractors** (activated via `telemetry=[...]`) and **explicit helper functions** (called inside your code).

---

## Built-in Extractors

These are registered automatically when you configure a framework adapter (LangGraph or CrewAI). Activate them by name on any decorator.

| Extractor | Name | What It Extracts |
|-----------|------|-----------------|
| `ToolCallsExtractor` | `tool_calls` | LLM tool-call decisions from AI messages |
| `RoutingExtractor` | `routing` | Edge/routing decisions (target node, reason) |
| `AgentCompletionExtractor` | `agent_completion` | Final answer, fallback detection |
| `QueryRewriteExtractor` | `query_rewrite` | Query decomposition results |
| `CompressionExtractor` | `compression` | Context compression statistics |

### Activation

```python
from ragwatch.adapters.langgraph import node

# Single extractor
@node("orchestrator", telemetry=["tool_calls"])
def orchestrator(state): ...

# Multiple extractors
@node("orchestrator", telemetry=["tool_calls", "routing", "agent_completion"])
def orchestrator(state): ...
```

### How Extractors Work

1. Your decorated function returns a result
2. The adapter's `normalize_result()` translates framework-specific shapes into semantic keys
3. Each active extractor reads from the normalized result (or falls back to raw result)
4. Attributes are written to the span via `safe_set_attribute()`

### Semantic Keys

Adapters normalize results into these well-known keys (all optional):

| Semantic Key | Type | Used By |
|-------------|------|---------|
| `tool_calls` | `list[dict]` | `ToolCallsExtractor` |
| `routing_target` | `str` | `RoutingExtractor` |
| `routing_reason` | `str` | `RoutingExtractor` |
| `agent_answer` | `str` | `AgentCompletionExtractor` |
| `is_fallback` | `bool` | `AgentCompletionExtractor` |
| `rewritten_questions` | `list[str]` | `QueryRewriteExtractor` |
| `is_clear` | `bool` | `QueryRewriteExtractor` |
| `original_query` | `str` | `QueryRewriteExtractor` |
| `compression_tokens_before` | `int` | `CompressionExtractor` |
| `compression_tokens_after` | `int` | `CompressionExtractor` |
| `context_summary` | `str` | `CompressionExtractor` |
| `queries_run` | `list[str]` | `CompressionExtractor` |
| `parents_retrieved` | `list[str]` | `CompressionExtractor` |

---

## Telemetry Helper Functions

For explicit control over what gets recorded, call these functions inside your traced code. All are exported from the top-level `ragwatch` package.

### record_chunks

Record per-chunk retrieval telemetry with content, scores, and source metadata:

```python
from ragwatch import record_chunks

@trace("retrieval.search", span_kind=SpanKind.RETRIEVER)
def retrieve(query: str):
    results = vector_db.similarity_search_with_relevance_scores(query, k=5)
    record_chunks(results, query=query)
    return results
```

**Attributes set:**

```
retrieval.chunk_count           = 5
retrieval.chunk.avg_score       = 0.87
retrieval.chunk.min_score       = 0.72
retrieval.chunk.max_score       = 0.93
retrieval.query                 = "what is RAG?"
retrieval.chunk.0.content       = "RAG combines retrieval with..."
retrieval.chunk.0.score         = 0.93
retrieval.chunk.0.source        = "doc_name.md"
retrieval.chunk.0.parent_id     = "abc123"
retrieval.chunk.0.char_count    = 487
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `Sequence[(Document, float)]` | required | `(doc, score)` tuples |
| `query` | `str` | `""` | The search query |
| `span` | `Span` | current span | OTel span to record on |
| `max_content_chars` | `int` | 600 | Truncate chunk content |

---

### record_tool_calls

Record LLM tool-call decisions on the orchestrator span:

```python
from ragwatch import record_tool_calls

@node("orchestrator")
def orchestrator(state):
    response = llm_with_tools.invoke(state["messages"])
    record_tool_calls(response.tool_calls)
    return {"messages": [response]}
```

**Attributes set:**

```
llm.tool_calls.count    = 2
llm.tool_calls.names    = ["search_docs", "calculate"]
llm.tool_call.0.name    = "search_docs"
llm.tool_call.0.args    = "{'query': 'RAG latency', 'limit': 5}"
llm.tool_call.0.id      = "call_abc123"
```

Also emits a `llm.tool_calls_decided` span event on the timeline.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_calls` | `list[dict]` | required | Tool-call dicts with `name`, `args`, `id` |
| `span` | `Span` | current span | OTel span to record on |

---

### record_routing

Record a routing/edge decision:

```python
from ragwatch import record_routing

@node("router")
def route_decision(state):
    target = "retrieve" if needs_context(state) else "answer"
    record_routing(
        from_node="router",
        to_node=target,
        reason=f"Context sufficient: {not needs_context(state)}",
    )
    return Command(goto=target, update=state)
```

**Attributes set:**

```
routing.from_node = "router"
routing.to_node   = "retrieve"
routing.reason    = "Context sufficient: False"
```

Also emits a `routing.decision` span event.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_node` | `str` | required | Source node name |
| `to_node` | `str` | required | Destination node name |
| `reason` | `str` | `""` | Human-readable reason |
| `span` | `Span` | current span | OTel span to record on |

---

### record_agent_completion

Record agent task completion metadata:

```python
from ragwatch import record_agent_completion

@node("collect-answer")
def collect_answer(state):
    answer = state.get("final_answer", "")
    is_fallback = not bool(answer.strip())

    record_agent_completion(
        status="success" if not is_fallback else "fallback",
        iteration_count=state.get("iteration_count", 0),
        tool_call_count=state.get("tool_call_count", 0),
        question=state.get("question", ""),
        answer_length=len(answer),
        is_fallback=is_fallback,
    )

    return state
```

**Attributes set:**

```
agent.completion_status   = "success"
agent.iteration_count     = 3
agent.tool_call_count     = 5
agent.question            = "What is the revenue for Q3?"
agent.question_index      = 1
agent.answer_length       = 342
agent.is_fallback         = false
```

Also emits an `agent.task_completed` span event.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `str` | required | `"success"`, `"fallback"`, `"max_iterations"`, `"error"` |
| `iteration_count` | `int` | 0 | LLM→tool loop iterations |
| `tool_call_count` | `int` | 0 | Total tool invocations |
| `question` | `str` | `""` | The sub-question being answered |
| `question_index` | `int` | 0 | Position in decomposed list |
| `answer_length` | `int` | 0 | Character length of final answer |
| `is_fallback` | `bool` | `False` | Used the fallback response path |
| `span` | `Span` | current span | OTel span to record on |

---

### record_context_compression

Record context compression statistics:

```python
from ragwatch import record_context_compression

@node("compress-context")
def compress_context(state):
    old_tokens = estimate_tokens(state["messages"])
    summary = summarize(state["messages"])
    new_tokens = estimate_tokens(summary)

    record_context_compression(
        tokens_before=old_tokens,
        tokens_after=new_tokens,
        queries_run=list(state.get("queries_run", [])),
        parents_retrieved=list(state.get("parents_retrieved", [])),
    )

    return {**state, "context_summary": summary}
```

**Attributes set:**

```
context.compression.tokens_before       = 4200
context.compression.tokens_after        = 1100
context.compression.ratio               = 0.262
context.compression.queries_already_run = ["what is RAG?", ...]
context.compression.unique_queries       = 2
context.compression.parents_retrieved    = ["abc", "def"]
context.compression.unique_parents       = 2
```

Also emits a `context.compressed` span event with reduction percentage.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokens_before` | `int` | required | Token count before compression |
| `tokens_after` | `int` | 0 | Token count after compression |
| `queries_run` | `list[str]` | `None` | Search queries already executed |
| `parents_retrieved` | `list[str]` | `None` | Parent chunk IDs already fetched |
| `span` | `Span` | current span | OTel span to record on |

---

### record_query_rewrite

Record query decomposition telemetry:

```python
from ragwatch import record_query_rewrite

@node("rewrite-query")
def rewrite_query(state):
    result = llm.invoke(rewrite_prompt.format(query=state["query"]))
    questions = result.get("rewrittenQuestions", [])

    record_query_rewrite(
        original_query=state["query"],
        rewritten_questions=questions,
        is_clear=result.get("questionIsClear", True),
    )

    return {**state, "rewrittenQuestions": questions}
```

**Attributes set:**

```
query.original            = "What is the total revenue and profit margin?"
query.rewritten_count     = 2
query.rewritten_questions = ["What is the total revenue?", "What is the profit margin?"]
query.is_clear            = true
```

Also emits a `query.rewritten` span event.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `original_query` | `str` | required | The raw user message |
| `rewritten_questions` | `list[str]` | required | Decomposed question list |
| `is_clear` | `bool` | required | Whether the query was deemed answerable |
| `span` | `Span` | current span | OTel span to record on |

---

## Writing Custom Extractors

See [Custom Attributes](custom-attributes.md#3-custom-telemetryextractors--reusable-named-extractors) for writing your own extractors, or [Extension Guide](EXTENSION_GUIDE.md) for the full protocol reference.

---

## Next Steps

- [Custom Attributes](custom-attributes.md) — hooks, extractors, and policy
- [Configuration](configuration.md) — all `RAGWatchConfig` options
- [API Reference](api-reference.md) — complete public API
