"""OpenInference-compatible span kind enum for RAGWatch."""

from __future__ import annotations

from enum import Enum


class SpanKind(str, Enum):
    """Span kinds following the OpenInference standard.

    These map to semantic span types used in RAG and agentic workflows.
    """

    CHAIN = "CHAIN"
    AGENT = "AGENT"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
