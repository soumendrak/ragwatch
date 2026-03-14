"""Tests for configure() idempotency — repeated calls must not accumulate state."""

from __future__ import annotations

import pytest
from tests.conftest import InMemorySpanExporter

import ragwatch
from ragwatch import (
    RAGWatchConfig,
    SpanKind,
    configure,
    get_default_registry,
    trace,
)
from ragwatch.adapters.base import clear_adapters, get_all_adapters
from ragwatch.adapters.langgraph import LangGraphAdapter
from ragwatch.core.tracer import reset_tracer_provider
from ragwatch.instrumentation.extractors import reset_default_registry
from ragwatch.instrumentation.span_hooks import clear_global_hooks, get_global_hooks
from ragwatch.instrumentation.token_usage import clear_token_extractors, get_token_extractors
from ragwatch.instrumentation.result_transformers import (
    get_default_transformer_registry,
    reset_default_transformer_registry,
)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    ragwatch._ACTIVE_CONFIG = None
    reset_tracer_provider()
    clear_global_hooks()
    reset_default_registry()
    clear_adapters()
    clear_token_extractors()
    reset_default_transformer_registry()


# ---------------------------------------------------------------------------
# Hooks do not accumulate
# ---------------------------------------------------------------------------

class _DummyHook:
    def on_start(self, span, args, kwargs):
        pass

    def on_end(self, span, result):
        pass


def test_repeated_configure_does_not_duplicate_hooks():
    """Calling configure() twice with the same hooks should not accumulate."""
    exporter = InMemorySpanExporter()
    hook = _DummyHook()
    cfg = RAGWatchConfig(
        service_name="test",
        exporter=exporter,
        global_span_hooks=[hook],
    )
    configure(cfg)
    configure(cfg)

    hooks = get_global_hooks()
    assert len(hooks) == 1


def test_repeated_configure_replaces_hooks():
    """Second configure() should replace hooks, not append."""
    exporter = InMemorySpanExporter()
    hook1 = _DummyHook()
    hook2 = _DummyHook()

    configure(RAGWatchConfig(
        service_name="test", exporter=exporter,
        global_span_hooks=[hook1],
    ))
    assert len(get_global_hooks()) == 1

    configure(RAGWatchConfig(
        service_name="test", exporter=exporter,
        global_span_hooks=[hook2],
    ))
    assert len(get_global_hooks()) == 1


# ---------------------------------------------------------------------------
# Adapters do not accumulate
# ---------------------------------------------------------------------------

def test_repeated_configure_does_not_duplicate_adapters():
    exporter = InMemorySpanExporter()
    adapter = LangGraphAdapter()
    cfg = RAGWatchConfig(
        service_name="test", exporter=exporter,
        adapters=[adapter],
    )
    configure(cfg)
    configure(cfg)

    assert len(get_all_adapters()) == 1


# ---------------------------------------------------------------------------
# Extractors rebuilt from scratch
# ---------------------------------------------------------------------------

class _CustomExtractor:
    name = "custom_test"
    def extract(self, span, span_name, args, result, state):
        pass


def test_repeated_configure_does_not_duplicate_extractors():
    exporter = InMemorySpanExporter()
    ext = _CustomExtractor()
    cfg = RAGWatchConfig(
        service_name="test", exporter=exporter,
        custom_extractors=[ext],
    )
    configure(cfg)
    first_names = set(get_default_registry().names())

    configure(cfg)
    second_names = set(get_default_registry().names())

    assert first_names == second_names
    assert "custom_test" in second_names


def test_configure_without_custom_extractors_still_has_builtins():
    exporter = InMemorySpanExporter()
    configure(RAGWatchConfig(service_name="test", exporter=exporter))
    expected = {"tool_calls", "routing", "agent_completion", "query_rewrite", "compression"}
    assert expected == set(get_default_registry().names())


# ---------------------------------------------------------------------------
# Transformers and token extractors via configure()
# ---------------------------------------------------------------------------

class _DummyTransformer:
    @property
    def span_kind(self):
        return SpanKind.TOOL

    def transform(self, span, args, kwargs, result, result_formatter):
        return result


class _DummyTokenExtractor:
    def extract(self, span, result):
        pass


def test_configure_registers_custom_transformers():
    exporter = InMemorySpanExporter()
    transformer = _DummyTransformer()
    configure(RAGWatchConfig(
        service_name="test", exporter=exporter,
        custom_transformers=[transformer],
    ))
    reg = get_default_transformer_registry()
    assert reg.get(SpanKind.TOOL) is transformer


def test_configure_registers_custom_token_extractors():
    exporter = InMemorySpanExporter()
    ext = _DummyTokenExtractor()
    configure(RAGWatchConfig(
        service_name="test", exporter=exporter,
        custom_token_extractors=[ext],
    ))
    assert len(get_token_extractors()) == 1


def test_repeated_configure_does_not_duplicate_token_extractors():
    exporter = InMemorySpanExporter()
    ext = _DummyTokenExtractor()
    cfg = RAGWatchConfig(
        service_name="test", exporter=exporter,
        custom_token_extractors=[ext],
    )
    configure(cfg)
    configure(cfg)
    assert len(get_token_extractors()) == 1
