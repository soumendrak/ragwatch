"""Async-safe context for storing the query embedding between stages.

The query embedding is stored in a ``contextvars.ContextVar``. This keeps
large vectors out of OTel baggage while preserving request isolation for
async workloads and concurrent tasks.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

_query_embedding: ContextVar[Optional[list[float]]] = ContextVar(
    "ragwatch_query_embedding",
    default=None,
)


def set_query_embedding(embedding: list[float]) -> None:
    """Store *embedding* in thread-local storage.

    Args:
        embedding: Query embedding vector (up to 512-dim).
    """
    _query_embedding.set(embedding)


def get_query_embedding() -> Optional[list[float]]:
    """Retrieve the query embedding from thread-local storage.

    Returns:
        The stored embedding list, or ``None`` if not set.
    """
    return _query_embedding.get()


def clear_query_embedding() -> None:
    """Remove the stored query embedding from thread-local storage."""
    _query_embedding.set(None)
