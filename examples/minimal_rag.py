"""Minimal linear RAG example using RAGWatch.

Run::

    uv add ragwatch  # recommended
    # or: pip install ragwatch
    python examples/minimal_rag.py

Traces appear in your configured OTel backend (e.g. Jaeger at http://localhost:16686).
"""

from opentelemetry.sdk.trace.export import ConsoleSpanExporter

import ragwatch
from ragwatch import RAGWatchConfig, SpanKind, trace
from ragwatch.instrumentation.evaluators import chunk_relevance_score

# --- Configure RAGWatch with a console exporter for demo purposes ---
ragwatch.configure(
    RAGWatchConfig(
        service_name="minimal-rag-example",
        exporter=ConsoleSpanExporter(),
    )
)


# --- Stage 1: Embedding ---
@trace("ragwatch.embedding.generate", span_kind=SpanKind.EMBEDDING)
def embed_query(text: str) -> list[float]:
    """Simulate embedding generation. In production, call your embedding API."""
    # Fake 3-dim embedding
    return [0.5, 0.3, 0.2]


# --- Stage 2: Retrieval ---
@trace("ragwatch.retrieval.search", span_kind=SpanKind.RETRIEVER)
def retrieve_chunks(query: str) -> list[dict]:
    """Simulate vector search. In production, call your vector DB."""
    # Fake chunk embeddings
    chunk_embeddings = [
        [0.5, 0.3, 0.2],  # Very similar to query
        [0.1, 0.9, 0.0],  # Less similar
        [0.4, 0.4, 0.1],  # Somewhat similar
    ]

    # RAGWatch computes relevance scores automatically
    scores = chunk_relevance_score(chunk_embeddings)

    chunks = [
        {"text": "RAG combines retrieval with generation.", "score": scores[0]},
        {"text": "Transformers use self-attention.", "score": scores[1]},
        {"text": "Vector databases store embeddings.", "score": scores[2]},
    ]
    return chunks


# --- Stage 3: Response ---
@trace("ragwatch.response.emit", span_kind=SpanKind.CHAIN)
def generate_response(chunks: list[dict]) -> str:
    """Simulate LLM response generation."""
    top_chunk = max(chunks, key=lambda c: c["score"])
    return f"Based on: {top_chunk['text']}"


# --- Run the pipeline ---
if __name__ == "__main__":
    query = "What is RAG?"

    print(f"Query: {query}\n")

    embedding = embed_query(query)
    print(f"Embedding: {embedding}")

    chunks = retrieve_chunks(query)
    for c in chunks:
        print(f"  Chunk: {c['text']} (score: {c['score']:.3f})")

    response = generate_response(chunks)
    print(f"\nResponse: {response}")
