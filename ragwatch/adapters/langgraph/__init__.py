"""LangGraph adapter for RAGWatch.

Usage::

    from ragwatch.adapters.langgraph import node, workflow, tool
"""

from ragwatch.adapters.langgraph.decorators import node, tool, workflow

__all__ = ["node", "workflow", "tool"]
