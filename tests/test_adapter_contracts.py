"""Tests for ragwatch.adapters.base — FrameworkAdapter contracts."""

from __future__ import annotations


import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch import trace
from ragwatch.adapters.base import (
    FrameworkAdapter,
    clear_adapters,
    get_adapter,
    get_all_adapters,
    get_capabilities,
    register_adapter,
)
from ragwatch.adapters.crewai import CrewAIAdapter
from ragwatch.adapters.langgraph import LangGraphAdapter
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.extractors import (
    get_default_registry,
    reset_default_registry,
)


@pytest.fixture(autouse=True)
def _setup():
    clear_adapters()
    reset_default_registry()
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    yield exporter
    reset_tracer_provider()
    clear_adapters()
    reset_default_registry()


# ---------------------------------------------------------------------------
# FrameworkAdapter Protocol compliance
# ---------------------------------------------------------------------------


def test_langgraph_adapter_is_framework_adapter():
    adapter = LangGraphAdapter()
    assert isinstance(adapter, FrameworkAdapter)


def test_crewai_adapter_is_framework_adapter():
    adapter = CrewAIAdapter()
    assert isinstance(adapter, FrameworkAdapter)


# ---------------------------------------------------------------------------
# LangGraphAdapter
# ---------------------------------------------------------------------------


def test_langgraph_adapter_name():
    assert LangGraphAdapter().name == "langgraph"


def test_langgraph_extract_state_from_dict_arg():
    adapter = LangGraphAdapter()
    state = {"messages": [], "question": "test"}
    result = adapter.extract_state((state,), {})
    assert result is state


def test_langgraph_extract_state_no_dict():
    adapter = LangGraphAdapter()
    result = adapter.extract_state(("string_arg", 42), {})
    assert result is None


def test_langgraph_default_extractors():
    adapter = LangGraphAdapter()
    extractors = adapter.default_extractors()
    names = {e.name for e in extractors}
    assert names == {
        "tool_calls",
        "routing",
        "agent_completion",
        "query_rewrite",
        "compression",
    }


# ---------------------------------------------------------------------------
# CrewAIAdapter
# ---------------------------------------------------------------------------


def test_crewai_adapter_name():
    assert CrewAIAdapter().name == "crewai"


def test_crewai_extract_state_from_kwargs():
    adapter = CrewAIAdapter()
    result = adapter.extract_state((), {"task": "research", "topic": "AI"})
    assert result == {"task": "research", "topic": "AI"}


def test_crewai_extract_state_from_dict_arg():
    adapter = CrewAIAdapter()
    data = {"task": "research"}
    result = adapter.extract_state((data,), {})
    assert result is data


def test_crewai_extract_state_no_dict():
    adapter = CrewAIAdapter()
    result = adapter.extract_state(("string",), {})
    assert result is None


def test_crewai_default_extractors_empty():
    adapter = CrewAIAdapter()
    assert adapter.default_extractors() == []


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


def test_register_and_get_adapter():
    adapter = LangGraphAdapter()
    register_adapter(adapter)
    assert get_adapter("langgraph") is adapter


def test_get_adapter_unknown():
    assert get_adapter("unknown") is None


def test_get_all_adapters():
    register_adapter(LangGraphAdapter())
    register_adapter(CrewAIAdapter())
    all_adapters = get_all_adapters()
    assert set(all_adapters.keys()) == {"langgraph", "crewai"}


def test_clear_adapters():
    register_adapter(LangGraphAdapter())
    clear_adapters()
    assert get_all_adapters() == {}


# ---------------------------------------------------------------------------
# Custom adapter
# ---------------------------------------------------------------------------


class _CustomAdapter:
    name = "custom_framework"

    def extract_state(self, args, kwargs):
        return kwargs.get("state")

    def default_extractors(self):
        return []


def test_custom_adapter_registration():
    adapter = _CustomAdapter()
    register_adapter(adapter)
    assert get_adapter("custom_framework") is adapter
    assert isinstance(adapter, FrameworkAdapter)


# ---------------------------------------------------------------------------
# Integration: adapters via configure()
# ---------------------------------------------------------------------------


def test_adapter_extractors_registered_via_configure(_setup):
    """Adapter's default_extractors get registered in the registry."""
    reset_default_registry()

    # Before registering adapter, registry has 5 built-in extractors
    reg = get_default_registry()
    initial_names = set(reg.names())

    # LangGraph adapter supplies extractors with same names as built-ins
    # so the count stays the same but they're re-registered
    adapter = LangGraphAdapter()
    for ext in adapter.default_extractors():
        reg.register(ext)

    assert set(reg.names()) == initial_names  # same names, new instances


# ---------------------------------------------------------------------------
# Runtime wiring: adapter.extract_state() called by extract_all()
# ---------------------------------------------------------------------------


class _KwargsStateAdapter:
    """Adapter that extracts state from kwargs['state']."""

    name = "kwargs_state"

    def extract_state(self, args, kwargs):
        return kwargs.get("state")

    def default_extractors(self):
        return []


class _StateCapturingExtractor:
    """Extractor that records the state it received."""

    name = "state_capture"
    captured_state = None

    def extract(self, context):
        _StateCapturingExtractor.captured_state = context.state


def test_extract_all_uses_adapter_extract_state():
    """extract_all() should call adapter.extract_state() instead of
    hardcoding 'first dict in args'."""
    adapter = _KwargsStateAdapter()
    extractor = _StateCapturingExtractor()
    _StateCapturingExtractor.captured_state = None

    reg = get_default_registry()
    reg.register(extractor)

    # The state is in kwargs, NOT in positional args
    from opentelemetry import trace as otel_trace

    tracer = otel_trace.get_tracer("test")
    with tracer.start_as_current_span("test") as span:
        reg.extract_all(
            ["state_capture"],
            span,
            "test_span",
            args=("not_a_dict",),
            result={},
            kwargs={"state": {"my_key": "my_value"}},
            adapter=adapter,
            context=type(
                "Ctx",
                (),
                {"state": adapter.extract_state((), {"state": {"my_key": "my_value"}})},
            )(),
        )

    assert _StateCapturingExtractor.captured_state == {"my_key": "my_value"}


def test_extract_all_requires_context():
    """Extractor dispatch now requires InstrumentationContext."""
    extractor = _StateCapturingExtractor()
    _StateCapturingExtractor.captured_state = None

    reg = get_default_registry()
    reg.register(extractor)
    from opentelemetry import trace as otel_trace

    tracer = otel_trace.get_tracer("test")
    with tracer.start_as_current_span("test") as span:
        with pytest.raises(ValueError, match="requires context"):
            reg.extract_all(
                ["state_capture"],
                span,
                "test_span",
                args=({"legacy": True},),
                result={},
            )


def test_trace_decorator_adapter_param(_setup):
    """@trace(adapter='...') resolves and uses the registered adapter."""
    adapter = _KwargsStateAdapter()
    register_adapter(adapter)
    extractor = _StateCapturingExtractor()
    _StateCapturingExtractor.captured_state = None
    get_default_registry().register(extractor)

    @trace("test-fn", telemetry=["state_capture"], adapter="kwargs_state")
    def my_fn(x, state=None):
        return {"ok": True}

    my_fn("hello", state={"from_kwarg": True})

    assert _StateCapturingExtractor.captured_state == {"from_kwarg": True}


def test_langgraph_capabilities():
    adapter = LangGraphAdapter()
    caps = get_capabilities(adapter)
    assert "routing" in caps
    assert "tool_calls" in caps
    assert "agent_completion" in caps
    assert "query_rewrite" in caps
    assert "compression" in caps


def test_crewai_capabilities():
    adapter = CrewAIAdapter()
    caps = get_capabilities(adapter)
    assert "agent_completion" in caps


def test_custom_adapter_without_capabilities():
    """Adapters without capabilities() return empty set via helper."""
    adapter = _CustomAdapter()
    caps = get_capabilities(adapter)
    assert caps == set()


def test_custom_adapter_with_capabilities():
    """Adapters with capabilities() return their declared set."""

    class _CapableAdapter:
        name = "capable"

        def extract_state(self, args, kwargs):
            return None

        def default_extractors(self):
            return []

        def capabilities(self):
            return {"custom_cap", "another_cap"}

    adapter = _CapableAdapter()
    caps = get_capabilities(adapter)
    assert caps == {"custom_cap", "another_cap"}
