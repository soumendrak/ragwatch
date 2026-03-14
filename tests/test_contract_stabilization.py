"""Tests for contract stabilization changes.

Covers:
- Protocol alignment (dual-signature for all 4 extension protocols)
- InstrumentationContext.normalized field
- Indexed attribute explosion guard
- I/O privacy scrubbing (redact_io_keys)
- normalize_result() on adapters
- Built-in extractors reading normalized keys
- InstrumentationContext exported from ragwatch top-level
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import ragwatch
from ragwatch import (
    AttributePolicy,
    InstrumentationContext,
    RAGWatchConfig,
    SpanKind,
    configure,
    get_default_registry,
    safe_set_attribute,
    trace,
)
from ragwatch.adapters.base import normalize_result
from ragwatch.adapters.langgraph.adapter import LangGraphAdapter
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.attribute_policy import AttributePolicy as AP
from ragwatch.instrumentation.attributes import _span_indexed_counts
from ragwatch.instrumentation.context_model import InstrumentationContext as CtxModel
from ragwatch.instrumentation.extractors import (
    AgentCompletionExtractor,
    CompressionExtractor,
    QueryRewriteExtractor,
    RoutingExtractor,
    ToolCallsExtractor,
)
from ragwatch.instrumentation.io_tracker import track_input, track_output
from tests.conftest import InMemorySpanExporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_config():
    """Reset global config after each test."""
    yield
    ragwatch._ACTIVE_CONFIG = None


@pytest.fixture()
def exporter():
    exp = InMemorySpanExporter()
    configure(RAGWatchConfig(service_name="test-contract-stab"))
    configure_tracer(
        RAGWatchConfig(service_name="test-contract-stab", exporter=exp),
        _force_flush=True,
    )
    yield exp
    reset_tracer_provider()


@pytest.fixture()
def mock_span():
    span = MagicMock()
    span.is_recording.return_value = True
    return span


# ---------------------------------------------------------------------------
# 1. InstrumentationContext exported from ragwatch
# ---------------------------------------------------------------------------

class TestInstrumentationContextExport:

    def test_importable_from_top_level(self):
        from ragwatch import InstrumentationContext
        assert InstrumentationContext is CtxModel

    def test_in_all(self):
        assert "InstrumentationContext" in ragwatch.__all__

    def test_normalized_field_exists(self):
        span = MagicMock()
        ctx = CtxModel(
            span=span, span_name="t", span_kind=SpanKind.CHAIN, func_name="f",
        )
        assert ctx.normalized is None
        ctx.normalized = {"tool_calls": []}
        assert ctx.normalized == {"tool_calls": []}


# ---------------------------------------------------------------------------
# 2. Indexed attribute explosion guard
# ---------------------------------------------------------------------------

class TestIndexedAttributeExplosion:

    def test_writes_under_limit_allowed(self, mock_span):
        policy = AP(max_indexed_attributes=3)
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        safe_set_attribute(mock_span, "retrieval.chunk.0.content", "a")
        safe_set_attribute(mock_span, "retrieval.chunk.1.content", "b")
        safe_set_attribute(mock_span, "retrieval.chunk.2.content", "c")
        assert mock_span.set_attribute.call_count == 3

    def test_writes_at_limit_skipped(self, mock_span):
        policy = AP(max_indexed_attributes=2)
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        safe_set_attribute(mock_span, "retrieval.chunk.0.content", "a")
        safe_set_attribute(mock_span, "retrieval.chunk.1.content", "b")
        safe_set_attribute(mock_span, "retrieval.chunk.2.content", "c")  # skipped
        assert mock_span.set_attribute.call_count == 2

    def test_span_event_recorded_on_limit(self, mock_span):
        _span_indexed_counts.clear()
        policy = AP(max_indexed_attributes=1)
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        safe_set_attribute(mock_span, "chunk.1.score", 0.5)  # skipped
        mock_span.add_event.assert_called_once()
        event_args = mock_span.add_event.call_args
        assert event_args[0][0] == "ragwatch.indexed_attr_limit"

    def test_event_recorded_only_once_per_prefix(self, mock_span):
        _span_indexed_counts.clear()
        policy = AP(max_indexed_attributes=1)
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        safe_set_attribute(mock_span, "chunk.1.score", 0.5)
        safe_set_attribute(mock_span, "chunk.2.score", 0.6)
        safe_set_attribute(mock_span, "chunk.3.score", 0.7)
        # Only one event per prefix
        assert mock_span.add_event.call_count == 1

    def test_different_prefixes_tracked_separately(self, mock_span):
        _span_indexed_counts.clear()
        policy = AP(max_indexed_attributes=1)
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        safe_set_attribute(mock_span, "chunk.1.score", 0.5)
        safe_set_attribute(mock_span, "doc.1.title", "t")
        assert mock_span.add_event.call_count == 2  # one per prefix

    def test_non_indexed_keys_unaffected(self, mock_span):
        policy = AP(max_indexed_attributes=1)
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        safe_set_attribute(mock_span, "my.regular.key", "val")
        mock_span.set_attribute.assert_called_once_with("my.regular.key", "val")

    def test_default_max_indexed_attributes(self):
        policy = AP()
        assert policy.max_indexed_attributes == 50


# ---------------------------------------------------------------------------
# 3. I/O privacy scrubbing
# ---------------------------------------------------------------------------

class TestIOPrivacyScrubbing:

    def test_scrub_io_payload_redacts_sensitive_keys(self):
        policy = AP(redact_io_keys=["password", "secret"])
        payload = {"username": "alice", "password": "s3cret", "data": "ok"}
        result = policy.scrub_io_payload(payload)
        assert result["username"] == "alice"
        assert result["password"] == "[REDACTED]"
        assert result["data"] == "ok"

    def test_scrub_io_payload_recursive(self):
        policy = AP(redact_io_keys=["token"])
        payload = {"outer": {"token": "abc123", "name": "test"}}
        result = policy.scrub_io_payload(payload)
        assert result["outer"]["token"] == "[REDACTED]"
        assert result["outer"]["name"] == "test"

    def test_scrub_io_payload_lists(self):
        policy = AP(redact_io_keys=["api_key"])
        payload = [{"api_key": "key1", "val": 1}, {"val": 2}]
        result = policy.scrub_io_payload(payload)
        assert result[0]["api_key"] == "[REDACTED]"
        assert result[0]["val"] == 1
        assert result[1]["val"] == 2

    def test_scrub_io_payload_case_insensitive(self):
        policy = AP(redact_io_keys=["Password"])
        payload = {"password": "secret"}
        result = policy.scrub_io_payload(payload)
        assert result["password"] == "[REDACTED]"

    def test_scrub_io_payload_empty_keys_no_op(self):
        policy = AP(redact_io_keys=[])
        payload = {"password": "secret"}
        result = policy.scrub_io_payload(payload)
        assert result["password"] == "secret"

    def test_default_redact_io_keys(self):
        policy = AP()
        assert "password" in policy.redact_io_keys
        assert "api_key" in policy.redact_io_keys
        assert "authorization" in policy.redact_io_keys

    def test_track_input_scrubs_sensitive_kwargs(self, mock_span):
        policy = AP(redact_io_keys=["password"])
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        track_input(mock_span, (), {"user": "alice", "password": "s3cret"})
        call_args = mock_span.set_attribute.call_args
        serialized = call_args[0][1]
        assert "[REDACTED]" in serialized
        assert "s3cret" not in serialized

    def test_track_output_scrubs_sensitive_result(self, mock_span):
        policy = AP(redact_io_keys=["token"])
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)

        track_output(mock_span, {"result": "ok", "token": "abc123"})
        call_args = mock_span.set_attribute.call_args
        serialized = call_args[0][1]
        assert "[REDACTED]" in serialized
        assert "abc123" not in serialized


# ---------------------------------------------------------------------------
# 4. normalize_result() on adapters
# ---------------------------------------------------------------------------

class TestNormalizeResult:

    def test_normalize_result_none_adapter(self):
        assert normalize_result(None, {"data": 1}, {}) is None

    def test_normalize_result_adapter_without_method(self):
        class _Plain:
            name = "plain"
            def extract_state(self, args, kwargs): return None
            def default_extractors(self): return []

        assert normalize_result(_Plain(), {"data": 1}, {}) is None

    def test_langgraph_normalize_tool_calls(self):
        adapter = LangGraphAdapter()
        msg = MagicMock()
        msg.tool_calls = [{"name": "search", "args": {"q": "x"}}]
        result = {"messages": [msg]}
        norm = adapter.normalize_result(result, {})
        assert norm is not None
        assert norm["tool_calls"] == [{"name": "search", "args": {"q": "x"}}]

    def test_langgraph_normalize_agent_completion(self):
        adapter = LangGraphAdapter()
        result = {"agent_answers": [{"answer": "42"}], "final_answer": "42"}
        norm = adapter.normalize_result(result, {})
        assert norm is not None
        assert norm["agent_answer"] == "42"
        assert norm["is_fallback"] is False

    def test_langgraph_normalize_fallback(self):
        adapter = LangGraphAdapter()
        result = {"agent_answers": [{"answer": "Unable to generate an answer."}]}
        norm = adapter.normalize_result(result, {})
        assert norm["is_fallback"] is True

    def test_langgraph_normalize_query_rewrite(self):
        adapter = LangGraphAdapter()
        result = {
            "rewrittenQuestions": ["q1", "q2"],
            "questionIsClear": True,
            "originalQuery": "original",
        }
        norm = adapter.normalize_result(result, {})
        assert norm["rewritten_questions"] == ["q1", "q2"]
        assert norm["is_clear"] is True
        assert norm["original_query"] == "original"

    def test_langgraph_normalize_command_routing(self):
        adapter = LangGraphAdapter()
        cmd = MagicMock()
        cmd.goto = "compress"
        cmd.update = {}
        norm = adapter.normalize_result(cmd, None)
        assert norm is not None
        assert norm["routing_target"] == "compress"

    def test_langgraph_normalize_non_dict_returns_none(self):
        adapter = LangGraphAdapter()
        assert adapter.normalize_result("plain string", {}) is None

    def test_langgraph_normalize_empty_dict_returns_none(self):
        adapter = LangGraphAdapter()
        assert adapter.normalize_result({}, {}) is None


# ---------------------------------------------------------------------------
# 5. Built-in extractors reading normalized keys (context-first path)
# ---------------------------------------------------------------------------

class TestExtractorsNormalized:

    def _make_ctx(self, raw_result, normalized=None, state=None):
        span = MagicMock()
        span.is_recording.return_value = True
        return CtxModel(
            span=span, span_name="test", span_kind=SpanKind.AGENT,
            func_name="test_fn", state=state or {},
            normalized=normalized, raw_result=raw_result,
        )

    def test_tool_calls_from_normalized(self):
        tc = [{"name": "search", "args": {"q": "x"}}]
        ctx = self._make_ctx(
            raw_result={"messages": []},
            normalized={"tool_calls": tc},
        )
        ext = ToolCallsExtractor()
        ext.extract(ctx)
        # Should have called set_attribute for tool calls on the span
        assert ctx.span.set_attribute.called or ctx.span.is_recording.called

    def test_routing_from_normalized(self):
        ctx = self._make_ctx(
            raw_result={},
            normalized={"routing_target": "tools", "routing_reason": "calling: [search]"},
        )
        ext = RoutingExtractor()
        ext.extract(ctx)
        # Verify routing attributes were written
        assert ctx.span.set_attribute.called

    def test_agent_completion_from_normalized(self):
        ctx = self._make_ctx(
            raw_result={},
            normalized={"agent_answer": "42", "is_fallback": False},
            state={"iteration_count": 2, "tool_call_count": 1,
                   "question": "what?", "question_index": 0},
        )
        ext = AgentCompletionExtractor()
        ext.extract(ctx)
        assert ctx.span.set_attribute.called

    def test_query_rewrite_from_normalized(self):
        ctx = self._make_ctx(
            raw_result={},
            normalized={
                "rewritten_questions": ["q1", "q2"],
                "is_clear": True,
                "original_query": "orig",
            },
        )
        ext = QueryRewriteExtractor()
        ext.extract(ctx)
        assert ctx.span.set_attribute.called

    def test_compression_from_normalized(self):
        ctx = self._make_ctx(
            raw_result={},
            normalized={
                "compression_tokens_before": 100,
                "compression_tokens_after": 50,
                "queries_run": ["q1"],
                "parents_retrieved": ["p1"],
            },
        )
        ext = CompressionExtractor()
        ext.extract(ctx)
        assert ctx.span.set_attribute.called

    def test_extractors_fallback_to_legacy_when_no_normalized(self):
        """When normalized is None, extractors fall back to raw result reading."""
        msg = MagicMock()
        msg.tool_calls = [{"name": "search", "args": {}}]
        ctx = self._make_ctx(
            raw_result={"messages": [msg]},
            normalized=None,
        )
        ext = ToolCallsExtractor()
        ext.extract(ctx)
        assert ctx.span.set_attribute.called


# ---------------------------------------------------------------------------
# 6. Protocol dual-signature — runtime_checkable still works
# ---------------------------------------------------------------------------

class TestProtocolDualSignature:

    def test_context_first_extractor_is_telemetry_extractor(self):
        from ragwatch import TelemetryExtractor

        class _CtxFirst:
            name = "ctx"
            def extract(self, context): pass

        assert isinstance(_CtxFirst(), TelemetryExtractor)

    def test_legacy_extractor_is_telemetry_extractor(self):
        from ragwatch import TelemetryExtractor

        class _Legacy:
            name = "leg"
            def extract(self, span, span_name, args, result, state): pass

        assert isinstance(_Legacy(), TelemetryExtractor)

    def test_context_first_hook_is_span_hook(self):
        from ragwatch import SpanHook

        class _CtxHook:
            def on_start(self, span, args, kwargs, *, context=None): pass
            def on_end(self, span, result, *, context=None): pass

        assert isinstance(_CtxHook(), SpanHook)

    def test_legacy_hook_is_span_hook(self):
        from ragwatch import SpanHook

        class _LegHook:
            def on_start(self, span, args, kwargs): pass
            def on_end(self, span, result): pass

        assert isinstance(_LegHook(), SpanHook)

    def test_context_first_transformer_is_result_transformer(self):
        from ragwatch import ResultTransformer

        class _CtxTrans:
            @property
            def span_kind(self): return SpanKind.TOOL
            def transform(self, context): return context.raw_result

        assert isinstance(_CtxTrans(), ResultTransformer)

    def test_legacy_transformer_is_result_transformer(self):
        from ragwatch import ResultTransformer

        class _LegTrans:
            @property
            def span_kind(self): return SpanKind.TOOL
            def transform(self, span, args, kwargs, result, formatter): return result

        assert isinstance(_LegTrans(), ResultTransformer)

    def test_context_first_token_extractor_is_token_extractor(self):
        from ragwatch import TokenExtractor

        class _CtxTok:
            def extract(self, context): pass

        assert isinstance(_CtxTok(), TokenExtractor)

    def test_legacy_token_extractor_is_token_extractor(self):
        from ragwatch import TokenExtractor

        class _LegTok:
            def extract(self, span, result): pass

        assert isinstance(_LegTok(), TokenExtractor)


# ---------------------------------------------------------------------------
# 7. End-to-end: context-first extractor via decorator
# ---------------------------------------------------------------------------

class TestContextFirstE2E:

    def test_context_first_extractor_receives_normalized_none_without_adapter(self, exporter):
        """Without an adapter, ctx.normalized is None."""
        captured = {}

        class _NormExtractor:
            name = "norm_check"
            def extract(self, context):
                captured["normalized"] = context.normalized
                captured["raw_result"] = context.raw_result

        get_default_registry().register(_NormExtractor())

        @trace("e2e-norm", span_kind=SpanKind.AGENT, telemetry=["norm_check"])
        def my_func(state):
            return {"data": True}

        my_func({"key": "val"})

        assert captured["raw_result"] == {"data": True}
        assert captured["normalized"] is None


# ---------------------------------------------------------------------------
# 8. Positive E2E: @trace(adapter="langgraph") → ctx.normalized
# ---------------------------------------------------------------------------

class TestNormalizedE2EWithAdapter:

    @pytest.fixture()
    def lg_exporter(self):
        """Exporter + LangGraphAdapter configured together."""
        from ragwatch.adapters.langgraph.adapter import LangGraphAdapter
        exp = InMemorySpanExporter()
        configure(RAGWatchConfig(
            service_name="e2e-norm-test",
            adapters=[LangGraphAdapter()],
        ))
        # Override tracer with force-flush so spans export immediately
        configure_tracer(
            RAGWatchConfig(service_name="e2e-norm-test", exporter=exp),
            _force_flush=True,
        )
        yield exp
        reset_tracer_provider()

    def test_agent_completion_normalized_e2e(self, lg_exporter):
        """Adapter normalizes agent_answers → ctx.normalized["agent_answer"]."""
        captured = {}

        class _CaptureExtractor:
            name = "capture_norm"
            def extract(self, context):
                captured["normalized"] = context.normalized
                captured["state"] = context.state

        get_default_registry().register(_CaptureExtractor())

        @trace("e2e-agent", span_kind=SpanKind.AGENT,
               telemetry=["capture_norm"], adapter="langgraph")
        def agent_node(state):
            return {"agent_answers": [{"answer": "The answer is 42"}],
                    "final_answer": "The answer is 42"}

        agent_node({"question": "What is the meaning?"})

        norm = captured["normalized"]
        assert norm is not None
        assert norm["agent_answer"] == "The answer is 42"
        assert norm["is_fallback"] is False

    def test_tool_calls_normalized_e2e(self, lg_exporter):
        """Adapter normalizes messages with tool_calls → ctx.normalized["tool_calls"]."""
        captured = {}

        class _CaptureExtractor:
            name = "capture_tc"
            def extract(self, context):
                captured["normalized"] = context.normalized

        get_default_registry().register(_CaptureExtractor())

        msg = MagicMock()
        msg.tool_calls = [{"name": "web_search", "args": {"q": "RAGWatch"}}]

        @trace("e2e-tools", span_kind=SpanKind.AGENT,
               telemetry=["capture_tc"], adapter="langgraph")
        def orchestrator(state):
            return {"messages": [msg]}

        orchestrator({"messages": []})

        norm = captured["normalized"]
        assert norm is not None
        assert norm["tool_calls"] == [{"name": "web_search", "args": {"q": "RAGWatch"}}]

    def test_query_rewrite_normalized_e2e(self, lg_exporter):
        """Adapter normalizes rewrittenQuestions → ctx.normalized."""
        captured = {}

        class _CaptureExtractor:
            name = "capture_qr"
            def extract(self, context):
                captured["normalized"] = context.normalized

        get_default_registry().register(_CaptureExtractor())

        @trace("e2e-qr", span_kind=SpanKind.AGENT,
               telemetry=["capture_qr"], adapter="langgraph")
        def rewriter(state):
            return {
                "rewrittenQuestions": ["sub-q1", "sub-q2"],
                "questionIsClear": True,
                "originalQuery": "main question",
            }

        rewriter({"messages": []})

        norm = captured["normalized"]
        assert norm is not None
        assert norm["rewritten_questions"] == ["sub-q1", "sub-q2"]
        assert norm["is_clear"] is True
        assert norm["original_query"] == "main question"

    def test_routing_via_command_normalized_e2e(self, lg_exporter):
        """Adapter normalizes Command(goto=...) → ctx.normalized["routing_target"]."""
        captured = {}

        class _CaptureExtractor:
            name = "capture_route"
            def extract(self, context):
                captured["normalized"] = context.normalized

        get_default_registry().register(_CaptureExtractor())

        cmd = MagicMock()
        cmd.goto = "compress"
        cmd.update = {}

        @trace("e2e-route", span_kind=SpanKind.AGENT,
               telemetry=["capture_route"], adapter="langgraph")
        def router(state):
            return cmd

        router({"messages": []})

        norm = captured["normalized"]
        assert norm is not None
        assert norm["routing_target"] == "compress"


# ---------------------------------------------------------------------------
# 9. Normalization failure handling (P1)
# ---------------------------------------------------------------------------

class TestNormalizationFailureHandling:

    @staticmethod
    def _make_exporter_and_configure(**kwargs):
        reset_tracer_provider()
        exp = InMemorySpanExporter()
        configure(RAGWatchConfig(**kwargs))
        # Override tracer with force-flush so spans export immediately
        configure_tracer(
            RAGWatchConfig(service_name=kwargs.get("service_name", "t"), exporter=exp),
            _force_flush=True,
        )
        return exp

    def test_normalize_error_logged_and_span_event(self):
        """Broken normalize_result() logs warning + records span event."""
        class _BrokenAdapter:
            name = "broken"
            def extract_state(self, args, kwargs): return None
            def default_extractors(self): return []
            def normalize_result(self, raw_result, state):
                raise ValueError("normalization boom")

        exp = self._make_exporter_and_configure(
            service_name="norm-fail-test",
            adapters=[_BrokenAdapter()],
        )

        @trace("norm-fail", span_kind=SpanKind.AGENT, adapter="broken")
        def my_func(state):
            return {"data": True}

        # Should NOT raise — failure is isolated
        result = my_func({"key": "val"})
        assert result == {"data": True}

        spans = exp.get_finished_spans()
        span = next(s for s in spans if s.name == "norm-fail")
        events = [e for e in span.events if e.name == "ragwatch.normalize_error"]
        assert len(events) == 1
        assert "normalization boom" in events[0].attributes["error"]

        reset_tracer_provider()

    def test_normalize_error_strict_mode_reraises(self):
        """In strict_mode, broken normalize_result() re-raises."""
        class _BrokenAdapter:
            name = "broken_strict"
            def extract_state(self, args, kwargs): return None
            def default_extractors(self): return []
            def normalize_result(self, raw_result, state):
                raise RuntimeError("strict boom")

        exp = self._make_exporter_and_configure(
            service_name="norm-strict-test",
            adapters=[_BrokenAdapter()],
            strict_mode=True,
        )

        @trace("norm-strict", span_kind=SpanKind.AGENT, adapter="broken_strict")
        def my_func(state):
            return {"ok": True}

        with pytest.raises(RuntimeError, match="strict boom"):
            my_func({"key": "val"})

        reset_tracer_provider()

    def test_normalize_graceful_when_no_adapter(self, exporter):
        """Without an adapter, normalization is silently None — no event."""
        @trace("no-adapter", span_kind=SpanKind.AGENT)
        def my_func(state):
            return {"data": True}

        result = my_func({"key": "val"})
        assert result == {"data": True}

        spans = exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "no-adapter")
        norm_events = [e for e in span.events if e.name == "ragwatch.normalize_error"]
        assert len(norm_events) == 0
