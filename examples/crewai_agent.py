"""CrewAI agent example using RAGWatch adapters.

Run::

    uv add ragwatch --extra crewai  # recommended
    # or: pip install ragwatch[crewai]
    python examples/crewai_agent.py

This example demonstrates how to decorate CrewAI-style agents and
endpoints with RAGWatch tracing.
"""

from opentelemetry.sdk.trace.export import ConsoleSpanExporter

import ragwatch
from ragwatch import RAGWatchConfig
from ragwatch.adapters.crewai import endpoint, node

ragwatch.configure(
    RAGWatchConfig(
        service_name="crewai-agent-example",
        exporter=ConsoleSpanExporter(),
    )
)


@node("researcher")
def researcher(task: str) -> dict:
    """Simulate a research agent."""
    return {
        "findings": f"Research results for: {task}",
        "sources": ["source1.pdf", "source2.pdf"],
    }


@node("writer")
def writer(findings: dict) -> dict:
    """Simulate a writer agent."""
    return {
        "article": f"Article based on: {findings['findings']}",
        "word_count": 500,
    }


@endpoint("research-crew")
def run_crew(topic: str) -> dict:
    """Orchestrate the crew."""
    findings = researcher(topic)
    article = writer(findings)
    return article


if __name__ == "__main__":
    result = run_crew("OpenTelemetry for LLM applications")
    print(f"Article: {result['article']}")
    print(f"Word count: {result['word_count']}")
