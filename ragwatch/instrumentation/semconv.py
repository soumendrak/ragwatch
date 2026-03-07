"""Centralized semantic convention strings for RAGWatch.

Standard OpenInference attributes are imported directly from
``openinference-semantic-conventions`` so RAGWatch stays in sync with the
official spec.  Only RAGWatch-specific attributes are defined here.
"""

from openinference.semconv.trace import SpanAttributes

SEMCONV_VERSION = "v1.40"

# --- OpenInference standard attributes (re-exported for convenience) ---------
OPENINFERENCE_SPAN_KIND  = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_TOKEN_COUNT_PROMPT     = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL      = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
INPUT_VALUE                = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE               = SpanAttributes.OUTPUT_VALUE

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

# --- RAGWatch metadata -----------------------
RAGWATCH_SDK_VERSION = "ragwatch.sdk.version"
