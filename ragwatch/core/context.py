"""Thread-local context for storing the query embedding between stages.

The query embedding is stored in a Python ``threading.local()`` object
(thread-local, **not** propagated across process boundaries).  This keeps
large vectors out of OTel baggage while allowing the retrieval stage to
compute ``chunk_relevance_score`` without re-embedding the query.

We use plain thread-local storage instead of OTel ``Context`` because OTel
context is scoped to spans — values set inside a span's context manager
are lost when the span exits.  Thread-local storage persists across
sequential function calls on the same thread, which is exactly what the
linear RAG pipeline needs.
"""

from __future__ import annotations

import threading
from typing import Optional

_thread_local = threading.local()


def set_query_embedding(embedding: list[float]) -> None:
    """Store *embedding* in thread-local storage.

    Args:
        embedding: Query embedding vector (up to 512-dim).
    """
    _thread_local.query_embedding = embedding


def get_query_embedding() -> Optional[list[float]]:
    """Retrieve the query embedding from thread-local storage.

    Returns:
        The stored embedding list, or ``None`` if not set.
    """
    return getattr(_thread_local, "query_embedding", None)


def clear_query_embedding() -> None:
    """Remove the stored query embedding from thread-local storage."""
    _thread_local.query_embedding = None
