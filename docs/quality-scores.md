# Quality Scores

RAGWatch's core differentiator: **computed** semantic quality scores on your RAG traces — not just recorded metadata.

---

## chunk_relevance_score

Computes cosine similarity between the query embedding and each chunk embedding, then records the scores as span attributes.

### How It Works

1. **Embedding stage** — `@trace` with `SpanKind.EMBEDDING` stores the query embedding in OTel context
2. **Retrieval stage** — `chunk_relevance_score()` reads the stored embedding and computes cosine similarity against each chunk
3. **Scores on spans** — `chunk.relevance_score` (average) and `chunk.relevance_scores` (per-chunk) appear as span attributes

### Basic Usage

```python
from ragwatch import trace, SpanKind
from ragwatch.instrumentation.evaluators import chunk_relevance_score


@trace("embedding.generate", span_kind=SpanKind.EMBEDDING)
def embed_query(text: str) -> list[float]:
    # Your embedding API — the return value is stored in OTel context
    return openai.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding


@trace("retrieval.search", span_kind=SpanKind.RETRIEVER)
def retrieve_chunks(query: str) -> list[dict]:
    results = vector_db.search(query, top_k=5)

    # Extract chunk embeddings from your vector DB results
    chunk_embeddings = [r.embedding for r in results]

    # RAGWatch computes cosine similarity automatically
    # Uses the query embedding stored by the EMBEDDING span above
    scores = chunk_relevance_score(chunk_embeddings)

    return [
        {"text": r.text, "score": score}
        for r, score in zip(results, scores)
    ]
```

### Span Attributes Set

| Attribute | Type | Description |
|-----------|------|-------------|
| `chunk.relevance_score` | `float` | Average cosine similarity across all chunks |
| `chunk.relevance_scores` | `list[float]` | Per-chunk cosine similarity scores |

### Explicit Query Embedding

If you don't use `SpanKind.EMBEDDING` (or need to override the stored embedding), pass it explicitly:

```python
scores = chunk_relevance_score(
    chunk_embeddings,
    query_embedding=[0.5, 0.3, 0.2],  # Explicit override
)
```

### Dimension Limit

Query embeddings stored in context are capped at `max_embedding_dims` (default 512) to keep context overhead small:

```python
ragwatch.configure(RAGWatchConfig(
    service_name="my-app",
    max_embedding_dims=1024,  # Increase for larger models
))
```

---

## Per-Chunk Telemetry with record_chunks

For richer per-chunk attributes (content, source, parent ID), use the `record_chunks` helper:

```python
from ragwatch import record_chunks


@trace("retrieval.search", span_kind=SpanKind.RETRIEVER)
def retrieve(query: str):
    results = vector_db.similarity_search_with_relevance_scores(query, k=5)

    # Record detailed per-chunk telemetry
    record_chunks(results, query=query)

    return results
```

### Attributes Set by record_chunks

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
retrieval.chunk.1.*             = ...
```

The `results` parameter expects `(Document, score)` tuples where documents have `.page_content` and `.metadata` attributes (LangChain-compatible).

### Content Truncation

Chunk content is truncated to 600 characters by default:

```python
record_chunks(results, query=query, max_content_chars=1000)
```

---

## User Feedback

Link user satisfaction scores back to specific traces:

```python
from ragwatch import record_feedback

# After the user rates a response
record_feedback(trace_id="abc123def456", score=0.85)
```

### How It Works

`record_feedback` creates a new `ragwatch.feedback` span with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `user.feedback_score` | `float` | The user's rating (typically 0.0 – 1.0) |
| `user.feedback_trace_id` | `str` | The trace ID of the request being rated |

This separate span allows you to correlate user satisfaction with retrieval quality scores in your OTel backend.

### Example: Feedback in a Web API

```python
from fastapi import FastAPI
from ragwatch import record_feedback

app = FastAPI()

@app.post("/feedback")
def submit_feedback(trace_id: str, score: float):
    record_feedback(trace_id=trace_id, score=score)
    return {"status": "recorded"}
```

---

## Interpreting Scores

### Relevance Score Ranges

| Score Range | Interpretation |
|-------------|---------------|
| 0.90 – 1.00 | Excellent match — chunk is highly relevant |
| 0.75 – 0.89 | Good match — chunk is relevant |
| 0.50 – 0.74 | Moderate — chunk may be tangentially relevant |
| 0.00 – 0.49 | Poor — chunk is likely irrelevant |

### Alerting on Low Scores

Use your OTel backend to set alerts when average relevance drops:

```
# Example PromQL (Grafana/Prometheus)
avg(chunk_relevance_score) < 0.7
```

---

## Next Steps

- [Telemetry Extraction](telemetry-extraction.md) — built-in extractors for tool calls, routing, etc.
- [Custom Attributes](custom-attributes.md) — add your own metrics alongside quality scores
- [Configuration](configuration.md) — embedding dimension limits and other settings
