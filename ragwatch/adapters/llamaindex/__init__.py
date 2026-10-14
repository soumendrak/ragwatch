"""LlamaIndex adapter for RAGWatch.

Usage::

    from ragwatch.adapters.llamaindex import RagWatchCallbackHandler
    from ragwatch.adapters.llamaindex import query_engine, retriever
    from ragwatch.adapters.llamaindex import LlamaIndexAdapter
"""

from ragwatch.adapters.llamaindex.adapter import LlamaIndexAdapter
from ragwatch.adapters.llamaindex.decorators import (
    chat_engine,
    ingestion,
    query_engine,
    retriever,
)
from ragwatch.adapters.llamaindex.hooks import RagWatchCallbackHandler, record_nodes

__all__ = [
    "LlamaIndexAdapter",
    "RagWatchCallbackHandler",
    "chat_engine",
    "ingestion",
    "query_engine",
    "record_nodes",
    "retriever",
]
