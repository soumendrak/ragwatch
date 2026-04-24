"""LangGraph agent example using actual LangGraph with RAGWatch tracing.

Run::

    uv add ragwatch langgraph  # recommended
    # or: pip install ragwatch langgraph
    python examples/langgraph_agent.py

This example demonstrates how to use RAGWatch decorators with
actual LangGraph StateGraph for a simple RAG pipeline.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.langgraph import node, workflow
from ragwatch.instrumentation.evaluators import chunk_relevance_score

# Configure RAGWatch
ragwatch.configure(
    RAGWatchConfig(
        service_name="langgraph-rag-example",
        exporter=ConsoleSpanExporter(),
    )
)


# Define the state schema
class RAGState(TypedDict):
    query: str
    embedding: list[float]
    docs: list[dict]
    response: str


# --- Stage 1: Embedding ---
@node("embed-node")
def embed_node(state: RAGState) -> RAGState:
    """Generate query embedding."""
    query = state["query"]
    # Fake embedding - replace with actual embedding API
    embedding = [0.5, 0.3, 0.2] if "RAG" in query else [0.2, 0.8, 0.1]
    return {"query": query, "embedding": embedding, "docs": [], "response": ""}


# --- Stage 2: Retrieval ---
@node("retrieve-node")
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks with computed relevance scores."""
    # Fake chunk embeddings - replace with actual vector DB search
    chunk_embeddings = [
        [0.5, 0.3, 0.2],  # Very similar
        [0.1, 0.9, 0.0],  # Less similar
        [0.4, 0.4, 0.1],  # Somewhat similar
    ]

    # RAGWatch computes relevance scores automatically
    scores = chunk_relevance_score(chunk_embeddings)

    docs = [
        {"text": "RAG combines retrieval with generation.", "score": scores[0]},
        {"text": "Transformers use self-attention.", "score": scores[1]},
        {"text": "Vector databases store embeddings.", "score": scores[2]},
    ]

    return {
        "query": state["query"],
        "embedding": state["embedding"],
        "docs": docs,
        "response": "",
    }


# --- Stage 3: Generation ---
@node("generate-node")
def generate_node(state: RAGState) -> RAGState:
    """Generate response from retrieved docs."""
    docs = state["docs"]
    # Sort by relevance score and pick top doc
    top_doc = max(docs, key=lambda x: x["score"])
    response = f"Based on: {top_doc['text']}"

    return {
        "query": state["query"],
        "embedding": state["embedding"],
        "docs": docs,
        "response": response,
    }


@workflow("rag-pipeline")
def build_rag_graph() -> StateGraph:
    """Build and compile the RAG LangGraph."""
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("embed", embed_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Add edges - linear flow: embed -> retrieve -> generate -> END
    workflow.add_edge("embed", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Set entry point
    workflow.set_entry_point("embed")

    return workflow.compile()


if __name__ == "__main__":
    # Build the graph
    rag_graph = build_rag_graph()

    # Run the pipeline
    result = rag_graph.invoke(
        {"query": "What is RAG?", "embedding": [], "docs": [], "response": ""}
    )

    print(f"\nQuery: {result['query']}")
    print(f"\nRetrieved {len(result['docs'])} chunks:")
    for doc in result["docs"]:
        print(f"  - {doc['text']} (score: {doc['score']:.3f})")
    print(f"\nResponse: {result['response']}")
