"""CrewAI adapter for RAGWatch.

Usage::

    from ragwatch.adapters.crewai import node, endpoint
    from ragwatch.adapters.crewai import CrewAIAdapter
"""

from ragwatch.adapters.crewai.adapter import CrewAIAdapter
from ragwatch.adapters.crewai.decorators import endpoint, node

__all__ = ["node", "endpoint", "CrewAIAdapter"]
