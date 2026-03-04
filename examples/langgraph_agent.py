"""LangGraph agent example using RAGWatch adapters.

Run::

    pip install ragwatch[langgraph]
    python examples/langgraph_agent.py

This example demonstrates how to decorate LangGraph-style nodes and
workflows with RAGWatch tracing.
"""

from opentelemetry.sdk.trace.export import ConsoleSpanExporter

import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.langgraph import node, workflow

ragwatch.configure(
    RAGWatchConfig(
        service_name="langgraph-agent-example",
        exporter=ConsoleSpanExporter(),
    )
)


@node("retrieve-node")
def retrieve_node(state: dict) -> dict:
    """Simulate a retrieval node in a LangGraph graph."""
    query = state.get("query", "")
    docs = [
        {"text": f"Document about {query}", "id": 1},
        {"text": f"Another document about {query}", "id": 2},
    ]
    return {**state, "docs": docs}


@node("generate-node")
def generate_node(state: dict) -> dict:
    """Simulate a generation node."""
    docs = state.get("docs", [])
    response = f"Generated answer from {len(docs)} documents"
    return {**state, "response": response}


@workflow("rag-agent-workflow")
def run_agent(input_data: dict) -> dict:
    """Orchestrate the agent workflow."""
    state = retrieve_node(input_data)
    state = generate_node(state)
    return state


if __name__ == "__main__":
    result = run_agent({"query": "What is observability?"})
    print(f"Query: {result['query']}")
    print(f"Docs: {len(result['docs'])}")
    print(f"Response: {result['response']}")
