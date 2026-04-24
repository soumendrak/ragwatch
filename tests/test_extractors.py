"""Tests for ragwatch.instrumentation.extractors — plugin registry system."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch import SpanKind, trace
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.extractors import (
    ExtractorRegistry,
    get_default_registry,
    reset_default_registry,
)


@pytest.fixture(autouse=True)
def _setup():
    reset_default_registry()
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    yield exporter
    reset_tracer_provider()
    reset_default_registry()


# ---------------------------------------------------------------------------
# ExtractorRegistry basics
# ---------------------------------------------------------------------------


class _DummyExtractor:
    name = "dummy"

    def extract(self, context):
        context.span.set_attribute("custom.dummy", "hello")


def test_registry_register_and_get():
    reg = ExtractorRegistry()
    ext = _DummyExtractor()
    reg.register(ext)
    assert reg.get("dummy") is ext
    assert "dummy" in reg.names()


def test_registry_unregister():
    reg = ExtractorRegistry()
    reg.register(_DummyExtractor())
    reg.unregister("dummy")
    assert reg.get("dummy") is None


def test_registry_get_unknown_returns_none():
    reg = ExtractorRegistry()
    assert reg.get("nonexistent") is None


def test_registry_extract_all_skips_unknown():
    """extract_all silently skips names that aren't registered."""
    reg = ExtractorRegistry()
    reg.register(_DummyExtractor())
    span = MagicMock()
    context = MagicMock(span=span)
    # Should not raise even though "unknown" is not registered
    reg.extract_all(["dummy", "unknown"], span, "test", (), {}, context=context)
    span.set_attribute.assert_called_once_with("custom.dummy", "hello")


# ---------------------------------------------------------------------------
# Default registry has all 5 built-ins
# ---------------------------------------------------------------------------


def test_default_registry_has_builtins():
    reg = get_default_registry()
    expected = {
        "tool_calls",
        "routing",
        "agent_completion",
        "query_rewrite",
        "compression",
    }
    assert expected == set(reg.names())


# ---------------------------------------------------------------------------
# Custom extractor via @trace(telemetry=[...])
# ---------------------------------------------------------------------------


def test_custom_extractor_via_trace(_setup):
    exporter = _setup

    class LatencyExtractor:
        name = "latency"

        def extract(self, context):
            if (
                isinstance(context.raw_result, dict)
                and "latency_ms" in context.raw_result
            ):
                context.set_attribute(
                    "custom.latency_ms", context.raw_result["latency_ms"]
                )

    get_default_registry().register(LatencyExtractor())

    @trace("my-node", span_kind=SpanKind.AGENT, telemetry=["latency"])
    def my_node(state):
        return {"latency_ms": 42}

    my_node({})

    span = next(s for s in exporter.get_finished_spans() if s.name == "my-node")
    assert span.attributes["custom.latency_ms"] == 42


# ---------------------------------------------------------------------------
# Custom extractor via configure(custom_extractors=[...])
# ---------------------------------------------------------------------------


def test_custom_extractor_via_configure(_setup):
    exporter = _setup

    class CostExtractor:
        name = "cost"

        def extract(self, context):
            if (
                isinstance(context.raw_result, dict)
                and "cost_usd" in context.raw_result
            ):
                context.set_attribute("custom.cost_usd", context.raw_result["cost_usd"])

    # Simulate what configure(custom_extractors=[...]) does
    get_default_registry().register(CostExtractor())

    @trace("pricing", span_kind=SpanKind.AGENT, telemetry=["cost"])
    def pricing_node(state):
        return {"cost_usd": 0.03}

    pricing_node({})

    span = next(s for s in exporter.get_finished_spans() if s.name == "pricing")
    assert span.attributes["custom.cost_usd"] == 0.03


# ---------------------------------------------------------------------------
# Built-in extractor: ToolCallsExtractor
# ---------------------------------------------------------------------------


def test_tool_calls_extractor(_setup):
    exporter = _setup

    tc = MagicMock()
    tc.tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "c1"}]

    @trace("orch", span_kind=SpanKind.AGENT, telemetry=["tool_calls"])
    def orchestrator(state):
        return {"messages": [tc]}

    orchestrator({})

    span = next(s for s in exporter.get_finished_spans() if s.name == "orch")
    assert span.attributes["llm.tool_calls.count"] == 1
    assert span.attributes["llm.tool_calls.names"] == ("search",)


# ---------------------------------------------------------------------------
# Built-in extractor: RoutingExtractor
# ---------------------------------------------------------------------------


def test_routing_extractor_with_tool_calls(_setup):
    exporter = _setup

    tc = MagicMock()
    tc.tool_calls = [{"name": "search", "args": {}, "id": "c1"}]

    @trace("orch", span_kind=SpanKind.AGENT, telemetry=["routing"])
    def orchestrator(state):
        return {"messages": [tc]}

    orchestrator({})

    span = next(s for s in exporter.get_finished_spans() if s.name == "orch")
    assert span.attributes["routing.from_node"] == "orch"
    assert span.attributes["routing.to_node"] == "tools"


def test_routing_extractor_no_tool_calls(_setup):
    exporter = _setup

    @trace("orch", span_kind=SpanKind.AGENT, telemetry=["routing"])
    def orchestrator(state):
        return {"messages": []}

    orchestrator({})

    span = next(s for s in exporter.get_finished_spans() if s.name == "orch")
    assert span.attributes["routing.to_node"] == "collect_answer"


# ---------------------------------------------------------------------------
# Built-in extractor: AgentCompletionExtractor
# ---------------------------------------------------------------------------


def test_agent_completion_extractor_success(_setup):
    exporter = _setup

    @trace("collect", span_kind=SpanKind.AGENT, telemetry=["agent_completion"])
    def collect(state):
        return {"final_answer": "The answer is 42."}

    collect({"question": "What?", "iteration_count": 2, "tool_call_count": 3})

    span = next(s for s in exporter.get_finished_spans() if s.name == "collect")
    assert span.attributes["agent.completion_status"] == "success"
    assert span.attributes["agent.iteration_count"] == 2
    assert span.attributes["agent.is_fallback"] is False


def test_agent_completion_extractor_fallback(_setup):
    exporter = _setup

    @trace("collect", span_kind=SpanKind.AGENT, telemetry=["agent_completion"])
    def collect(state):
        return {"final_answer": "Unable to generate an answer."}

    collect({})

    span = next(s for s in exporter.get_finished_spans() if s.name == "collect")
    assert span.attributes["agent.completion_status"] == "fallback"
    assert span.attributes["agent.is_fallback"] is True


# ---------------------------------------------------------------------------
# Built-in extractor: QueryRewriteExtractor
# ---------------------------------------------------------------------------


def test_query_rewrite_extractor(_setup):
    exporter = _setup

    @trace("rewrite", span_kind=SpanKind.AGENT, telemetry=["query_rewrite"])
    def rewrite(state):
        return {
            "originalQuery": "revenue and profit",
            "rewrittenQuestions": ["What is revenue?", "What is profit?"],
            "questionIsClear": True,
        }

    rewrite({})

    span = next(s for s in exporter.get_finished_spans() if s.name == "rewrite")
    assert span.attributes["query.original"] == "revenue and profit"
    assert span.attributes["query.rewritten_count"] == 2
    assert span.attributes["query.is_clear"] is True


# ---------------------------------------------------------------------------
# Built-in extractor: CompressionExtractor
# ---------------------------------------------------------------------------


def test_compression_extractor(_setup):
    exporter = _setup

    @trace("compress", span_kind=SpanKind.AGENT, telemetry=["compression"])
    def compress(state):
        return {"context_summary": "short summary"}

    state = {
        "messages": [],
        "context_summary": "a" * 400,  # ~100 tokens
        "retrieval_keys": {"parent::abc", "search::q1"},
    }
    compress(state)

    span = next(s for s in exporter.get_finished_spans() if s.name == "compress")
    assert "context.compression.tokens_before" in span.attributes


# ---------------------------------------------------------------------------
# Extractor overwrite
# ---------------------------------------------------------------------------


def test_register_overwrites_existing():
    reg = ExtractorRegistry()
    ext1 = _DummyExtractor()
    ext2 = _DummyExtractor()
    reg.register(ext1)
    reg.register(ext2)
    assert reg.get("dummy") is ext2


# ---------------------------------------------------------------------------
# Context-first extractor contract
# ---------------------------------------------------------------------------


class _ContextFirstExtractor:
    """New-style extractor that receives InstrumentationContext."""

    name = "ctx_extractor"
    captured_ctx = None

    def extract(self, context):
        _ContextFirstExtractor.captured_ctx = context
        context.set_attribute("custom.ctx_extractor", "from_context")


def test_context_first_extractor_receives_context(_setup):
    """Extractors with extract(self, context) get the InstrumentationContext."""
    exporter = _setup
    _ContextFirstExtractor.captured_ctx = None
    get_default_registry().register(_ContextFirstExtractor())

    @trace("ctx-ext-test", span_kind=SpanKind.AGENT, telemetry=["ctx_extractor"])
    def my_node(state):
        return {"ok": True}

    my_node({"question": "test"})

    ctx = _ContextFirstExtractor.captured_ctx
    assert ctx is not None
    assert ctx.span_name == "ctx-ext-test"
    assert ctx.state == {"question": "test"}
    assert ctx.raw_result == {"ok": True}

    span = next(s for s in exporter.get_finished_spans() if s.name == "ctx-ext-test")
    assert span.attributes["custom.ctx_extractor"] == "from_context"


def test_custom_extractor_uses_context_writer(_setup):
    """Custom extractors use InstrumentationContext as their contract."""
    exporter = _setup

    class _CustomExt:
        name = "custom_ext"

        def extract(self, context):
            context.set_attribute("custom.context_contract", "works")

    get_default_registry().register(_CustomExt())

    @trace("custom-test", span_kind=SpanKind.AGENT, telemetry=["custom_ext"])
    def my_node(state):
        return {"data": True}

    my_node({"key": "val"})

    span = next(s for s in exporter.get_finished_spans() if s.name == "custom-test")
    assert span.attributes["custom.context_contract"] == "works"
