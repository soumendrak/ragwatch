# LlamaIndex Integration

Instrument LlamaIndex query, chat and ingestion pipelines with RAGWatch.

---

## Installation

```bash
pip install ragwatch[llamaindex]
# or
uv add ragwatch --extra llamaindex
```

The adapter never imports LlamaIndex at module load, so `llama-index-core` stays optional.

---

## Callback handler (recommended)

Register `RagWatchCallbackHandler` on LlamaIndex's `CallbackManager` once; every LlamaIndex event becomes a RAGWatch span, with no decorators needed:

```python
from llama_index.core import Settings, VectorStoreIndex
from ragwatch import RAGWatchConfig, configure
from ragwatch.adapters.llamaindex import RagWatchCallbackHandler

configure(RAGWatchConfig(service_name="llamaindex-app"))
Settings.callback_manager.add_handler(RagWatchCallbackHandler())

index = VectorStoreIndex.from_documents(docs)
response = index.as_query_engine().query("What is RAGWatch?")
```

| LlamaIndex event | Span name | `openinference.span.kind` | Attributes |
|------------------|-----------|---------------------------|------------|
| `query` | `llama_index.query` | `CHAIN` | `input.value`, `output.value`, `response.length` |
| `retrieve`, `reranking` | `llama_index.retrieve` / `llama_index.reranking` | `RETRIEVER` | `retrieval.chunks.returned`, `chunk.relevance_scores`, `retrieval.chunk.{i}.*` |
| `synthesize` | `llama_index.synthesize` | `CHAIN` | `output.value`, `response.length` |
| `embedding` | `llama_index.embedding` | `EMBEDDING` | `embedding.model.name`, `embedding.dimensions` |
| `function_call` | `llama_index.function_call` | `TOOL` | `tool.name`, `input.value`, `output.value` |
| `llm` | — (skipped) | — | Leave LLM spans to OpenLLMetry; children are re-parented to the nearest traced ancestor |
| any other | `llama_index.<event>` | `CHAIN` | — |

- Spans are parented through LlamaIndex's own event `parent_id`, so concurrent asyncio queries never share spans. Each span is the current OTel span for the duration of its event, so OpenLLMetry provider spans nest beneath it.
- Spans left open when a trace ends (e.g. after a start-only `exception` event) are ended in `end_trace()`; the exception is recorded on the nearest traced span.
- If several query embeddings are emitted under one query (multiple `custom_embedding_strs`), no fallback score is computed. Scores that the retriever or reranker already set are never overwritten.
- Retrieved nodes use the retriever's scores. If a node has no score, RAGWatch falls back to cosine similarity between the node embedding and the query embedding captured from the `embedding` event under the query/retrieve.
- `input.value` / `output.value` respect `RAGWatchConfig(global_auto_track_io=False)`.
- Streaming responses (`StreamingResponse`, `AsyncStreamingResponse`) are never read, so the caller still gets the full stream; their spans carry no `output.value`.
- The fallback query embedding is scoped to the LlamaIndex query/retrieve event it was computed under, so a later retrieval never reuses a previous query's embedding.
- Exceptions reported by LlamaIndex set the span status to `ERROR`, record the exception and set `error.type`.
- Errors inside the handler are logged and swallowed, unless strict mode is on.

---

Events started outside a `start_trace`/`end_trace` pair are ended only by
their own end callback. To bound memory if an end never arrives, at most
`max_open_events` such events (default 10,000) are kept open; beyond that the
oldest is ended early and marked `ragwatch.callback.evicted=true`. Its
context is detached once the events attached after it have ended, so the
ended span does not stay current. Events inside a trace are never evicted. Set it with
`RagWatchCallbackHandler(max_open_events=...)`.

Embedding events are not made the current OpenTelemetry context, because
async embedding batches start all their events before awaiting any of them.
Provider spans created while embedding nest under the enclosing event instead.

## Decorators

For your own functions around LlamaIndex objects:

| Decorator | SpanKind | LlamaIndex concept |
|-----------|----------|--------------------|
| `@query_engine("name")` | `CHAIN` | `QueryEngine.query()` |
| `@retriever("name")` | `RETRIEVER` | `BaseRetriever.retrieve()` |
| `@chat_engine("name")` | `AGENT` | `ChatEngine.chat()` |
| `@ingestion("name")` | `EMBEDDING` | `IngestionPipeline.run()` |

```python
from ragwatch.adapters.llamaindex import chat_engine, query_engine

@query_engine("docs-qa")
def ask(question: str):
    return engine.query(question)

@chat_engine
async def chat(message: str):
    return await chat_engine_obj.achat(message)
```

Each decorator works with or without arguments, supports sync and async functions, and sets `adapter="llamaindex"`. The `LlamaIndexAdapter` is registered automatically on first use. Output tracking is streaming-safe: a `StreamingResponse` is returned untouched and its span has no `output.value`.

---

## Adapter

`LlamaIndexAdapter` implements the `FrameworkAdapter` protocol and normalizes LlamaIndex results into semantic keys:

- `Response.response` / `AgentChatResponse.response` / `CompletionResponse.text` → `agent_answer` (empty answer → `is_fallback=True`)
- `AgentChatResponse.sources` (`ToolOutput` list) → `tool_calls`

```python
from ragwatch import RAGWatchConfig, configure
from ragwatch.adapters.llamaindex import LlamaIndexAdapter

configure(RAGWatchConfig(service_name="app", adapters=[LlamaIndexAdapter()]))
```
