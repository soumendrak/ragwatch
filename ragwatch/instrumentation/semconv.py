"""Centralized semantic convention strings for RAGWatch.

This module is the **single source of truth** for all attribute keys used
across the SDK.  An AST-based CI guard (``tests/test_semconv.py``) ensures
no hardcoded attribute strings appear outside this file.
"""

SEMCONV_VERSION = "v1.40"

# --- OpenInference span kind attribute -------
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

# --- Embedding stage -------------------------
EMBEDDING_MODEL_NAME = "embedding.model.name"
EMBEDDING_DIMENSIONS = "embedding.dimensions"
EMBEDDING_DURATION_MS = "embedding.duration_ms"

# --- Retrieval stage -------------------------
RETRIEVAL_TOP_K = "retrieval.top_k"
RETRIEVAL_CHUNKS_RETURNED = "retrieval.chunks.returned"
CHUNK_RELEVANCE_SCORE = "chunk.relevance_score"
CHUNK_RELEVANCE_SCORES = "chunk.relevance_scores"

# --- Response / feedback ---------------------
RESPONSE_LENGTH = "response.length"
USER_FEEDBACK_SCORE = "user.feedback_score"
USER_FEEDBACK_TRACE_ID = "user.feedback_trace_id"

# --- I/O tracking ----------------------------
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"

# --- RAGWatch metadata -----------------------
RAGWATCH_SDK_VERSION = "ragwatch.sdk.version"
