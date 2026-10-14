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

- Spans are parented through LlamaIndex's own event `parent_id`, so concurrent asyncio queries never share spans.
- Retrieved nodes use the retriever's scores. If a node has no score, RAGWatch falls back to cosine similarity between the node embedding and the query embedding captured from the `embedding` event under the query/retrieve.
- `input.value` / `output.value` respect `RAGWatchConfig(global_auto_track_io=False)`.
- Streaming responses (`StreamingResponse`, `AsyncStreamingResponse`) are never read, so the caller still gets the full stream; their spans carry no `output.value`.
- The fallback query embedding is scoped to the LlamaIndex query/retrieve event it was computed under, so a later retrieval never reuses a previous query's embedding.
- Exceptions reported by LlamaIndex set the span status to `ERROR`, record the exception and set `error.type`.
- Errors inside the handler are logged and swallowed, unless strict mode is on.

---

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

Each decorator works with or without arguments, supports sync and async functions, and sets `adapter="llamaindex"`. The `LlamaIndexAdapter` is registered automatically on first use.

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
