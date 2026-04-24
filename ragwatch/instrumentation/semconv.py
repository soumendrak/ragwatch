"""Centralized semantic convention strings for RAGWatch.

Standard OpenInference attributes are imported directly from
``openinference-semantic-conventions`` so RAGWatch stays in sync with the
official spec.  Only RAGWatch-specific attributes are defined here.

Stability tiers
~~~~~~~~~~~~~~~~
- **OpenInference standard** — upstream spec, never changes within a semconv
  version.
- **RAGWatch stable** — committed API; changes are backward-compatible and
  versioned.
- **RAGWatch experimental** — may change or be removed without notice; used
  for fast iteration on new telemetry concepts.
"""

from openinference.semconv.trace import SpanAttributes

SEMCONV_VERSION = "v1.40"

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — OpenInference standard attributes (re-exported for convenience)
# ═══════════════════════════════════════════════════════════════════════════════
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — RAGWatch stable attributes
#   Dashboard-safe; part of the public contract.
# ═══════════════════════════════════════════════════════════════════════════════

# --- Embedding stage -------------------------
EMBEDDING_MODEL_NAME = "embedding.model.name"
EMBEDDING_DIMENSIONS = "embedding.dimensions"
EMBEDDING_DURATION_MS = "embedding.duration_ms"

# --- Retrieval stage: aggregate -----------------------
RETRIEVAL_TOP_K = "retrieval.top_k"
RETRIEVAL_CHUNKS_RETURNED = "retrieval.chunks.returned"
CHUNK_RELEVANCE_SCORE = "chunk.relevance_score"  # avg score (legacy)
CHUNK_RELEVANCE_SCORES = "chunk.relevance_scores"  # list of scores (legacy)

# --- Retrieval stage: per-chunk (indexed) ----------------------
# Use helpers.record_chunks() to set these automatically.
# Attributes follow the pattern:  retrieval.chunk.{i}.{field}
CHUNK_COUNT = "retrieval.chunk_count"
CHUNK_PREFIX = "retrieval.chunk"  # → .0.content, .0.score, …
CHUNK_MIN_SCORE = "retrieval.chunk.min_score"
CHUNK_MAX_SCORE = "retrieval.chunk.max_score"
CHUNK_AVG_SCORE = "retrieval.chunk.avg_score"
CHUNK_QUERY = "retrieval.query"

# --- Agent execution -------------------------
# Set via helpers.record_agent_completion()
AGENT_COMPLETION_STATUS = (
    "agent.completion_status"  # success | fallback | max_iterations | error
)
AGENT_ITERATION_COUNT = "agent.iteration_count"  # total LLM ↔ tool loops
AGENT_TOOL_CALL_COUNT = "agent.tool_call_count"  # total tool invocations
AGENT_QUESTION = "agent.question"  # the sub-question handled
AGENT_QUESTION_INDEX = "agent.question_index"  # position in decomposed questions
AGENT_ANSWER_LENGTH = "agent.answer_length"  # chars in final answer
AGENT_IS_FALLBACK = "agent.is_fallback"  # bool: fell back to fallback_response?

# --- LLM tool-call decisions (from orchestrator) -----
# Set via helpers.record_tool_calls()
LLM_TOOL_CALLS_COUNT = "llm.tool_calls.count"
LLM_TOOL_CALLS_NAMES = "llm.tool_calls.names"
LLM_TOOL_CALL_PREFIX = "llm.tool_call"  # → .0.name, .0.args, .0.id

# --- Routing decisions -------------------------
# Set via helpers.record_routing()
ROUTING_FROM_NODE = "routing.from_node"
ROUTING_TO_NODE = "routing.to_node"
ROUTING_REASON = "routing.reason"

# --- Error recording -------------------------
# OTel semconv: https://opentelemetry.io/docs/specs/semconv/general/recording-errors/
ERROR_TYPE = "error.type"

# --- Response / feedback ---------------------
RESPONSE_LENGTH = "response.length"
USER_FEEDBACK_SCORE = "user.feedback_score"
USER_FEEDBACK_TRACE_ID = "user.feedback_trace_id"
USER_FEEDBACK_SPAN_ID = "user.feedback_span_id"

# --- RAGWatch metadata -----------------------
RAGWATCH_SDK_VERSION = "ragwatch.sdk.version"

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 — RAGWatch experimental attributes
#   May change or be removed without notice.
# ═══════════════════════════════════════════════════════════════════════════════

# --- Context compression ---------------------
# Set via helpers.record_context_compression()
COMPRESSION_TOKENS_BEFORE = "context.compression.tokens_before"
COMPRESSION_TOKENS_AFTER = "context.compression.tokens_after"
COMPRESSION_RATIO = "context.compression.ratio"
COMPRESSION_QUERIES_RUN = "context.compression.queries_already_run"
COMPRESSION_PARENTS_RETRIEVED = "context.compression.parents_retrieved"
COMPRESSION_UNIQUE_QUERIES = "context.compression.unique_queries"
COMPRESSION_UNIQUE_PARENTS = "context.compression.unique_parents"

# --- Query rewriting -------------------------
QUERY_ORIGINAL = "query.original"
QUERY_REWRITTEN_COUNT = "query.rewritten_count"
QUERY_REWRITTEN_QUESTIONS = "query.rewritten_questions"  # list[str]
QUERY_IS_CLEAR = "query.is_clear"


STANDARD_ATTRIBUTES = frozenset(
    {
        OPENINFERENCE_SPAN_KIND,
        LLM_TOKEN_COUNT_PROMPT,
        LLM_TOKEN_COUNT_COMPLETION,
        LLM_TOKEN_COUNT_TOTAL,
        INPUT_VALUE,
        OUTPUT_VALUE,
    }
)

STABLE_ATTRIBUTES = frozenset(
    {
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DIMENSIONS,
        EMBEDDING_DURATION_MS,
        RETRIEVAL_TOP_K,
        RETRIEVAL_CHUNKS_RETURNED,
        CHUNK_RELEVANCE_SCORE,
        CHUNK_RELEVANCE_SCORES,
        CHUNK_COUNT,
        CHUNK_PREFIX,
        CHUNK_MIN_SCORE,
        CHUNK_MAX_SCORE,
        CHUNK_AVG_SCORE,
        CHUNK_QUERY,
        AGENT_COMPLETION_STATUS,
        AGENT_ITERATION_COUNT,
        AGENT_TOOL_CALL_COUNT,
        AGENT_QUESTION,
        AGENT_QUESTION_INDEX,
        AGENT_ANSWER_LENGTH,
        AGENT_IS_FALLBACK,
        LLM_TOOL_CALLS_COUNT,
        LLM_TOOL_CALLS_NAMES,
        LLM_TOOL_CALL_PREFIX,
        ROUTING_FROM_NODE,
        ROUTING_TO_NODE,
        ROUTING_REASON,
        ERROR_TYPE,
        RESPONSE_LENGTH,
        USER_FEEDBACK_SCORE,
        USER_FEEDBACK_TRACE_ID,
        USER_FEEDBACK_SPAN_ID,
        RAGWATCH_SDK_VERSION,
    }
)

EXPERIMENTAL_ATTRIBUTES = frozenset(
    {
        COMPRESSION_TOKENS_BEFORE,
        COMPRESSION_TOKENS_AFTER,
        COMPRESSION_RATIO,
        COMPRESSION_QUERIES_RUN,
        COMPRESSION_PARENTS_RETRIEVED,
        COMPRESSION_UNIQUE_QUERIES,
        COMPRESSION_UNIQUE_PARENTS,
        QUERY_ORIGINAL,
        QUERY_REWRITTEN_COUNT,
        QUERY_REWRITTEN_QUESTIONS,
        QUERY_IS_CLEAR,
    }
)

ALL_ATTRIBUTES = STANDARD_ATTRIBUTES | STABLE_ATTRIBUTES | EXPERIMENTAL_ATTRIBUTES


def get_attribute_stability(attribute: str) -> str | None:
    """Return the stability tier for a semantic attribute.

    Args:
        attribute: Semantic attribute name.

    Returns:
        One of ``"standard"``, ``"stable"``, ``"experimental"``, or ``None``
        when the attribute is not part of the RAGWatch schema registry.
    """
    if attribute in STANDARD_ATTRIBUTES:
        return "standard"
    if attribute in STABLE_ATTRIBUTES:
        return "stable"
    if attribute in EXPERIMENTAL_ATTRIBUTES:
        return "experimental"
    return None
