"""Tests for pluggable result transformation, token extraction, and I/O policy."""

from __future__ import annotations

from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest
from tests.conftest import InMemorySpanExporter

import ragwatch
from ragwatch import SpanKind, trace
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.result_transformers import (
    ResultTransformer,
    ResultTransformerRegistry,
    get_default_transformer_registry,
    reset_default_transformer_registry,
)
from ragwatch.instrumentation.token_usage import (
    TokenExtractor,
    clear_token_extractors,
    register_token_extractor,
)


@pytest.fixture(autouse=True)
def _setup():
    reset_default_transformer_registry()
    clear_token_extractors()
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(strict_mode=False)
    yield exporter
    ragwatch._ACTIVE_CONFIG = None
    reset_tracer_provider()
    reset_default_transformer_registry()
    clear_token_extractors()


# ---------------------------------------------------------------------------
# ResultTransformer protocol
# ---------------------------------------------------------------------------

class _UppercaseToolTransformer:
    """Custom transformer that uppercases string results for TOOL spans."""

    @property
    def span_kind(self) -> SpanKind:
        return SpanKind.TOOL

    def transform(self, span, args, kwargs, result, result_formatter):
        if isinstance(result, str):
            return result.upper()
        return result


def test_result_transformer_protocol():
    t = _UppercaseToolTransformer()
    assert isinstance(t, ResultTransformer)


def test_custom_transformer_overrides_builtin():
    """A registered ResultTransformer takes precedence over built-in logic."""
    get_default_transformer_registry().register(_UppercaseToolTransformer())

    @trace("tool-fn", span_kind=SpanKind.TOOL)
    def my_tool():
        return "hello world"

    result = my_tool()
    assert result == "HELLO WORLD"


def test_no_custom_transformer_uses_builtin():
    """Without a custom transformer, built-in TOOL logic applies."""
    @trace("tool-fn", span_kind=SpanKind.TOOL)
    def my_tool():
        return "pass through"

    result = my_tool()
    assert result == "pass through"  # strings pass through in built-in


def test_registry_clear():
    reg = ResultTransformerRegistry()
    reg.register(_UppercaseToolTransformer())
    assert reg.get(SpanKind.TOOL) is not None
    reg.clear()
    assert reg.get(SpanKind.TOOL) is None


def test_chain_spans_unaffected_by_tool_transformer():
    """CHAIN spans should not be affected by a TOOL transformer."""
    get_default_transformer_registry().register(_UppercaseToolTransformer())

    @trace("chain-fn", span_kind=SpanKind.CHAIN)
    def my_chain():
        return {"messages": ["hello"]}

    result = my_chain()
    assert result == {"messages": ["hello"]}


# ---------------------------------------------------------------------------
# TokenExtractor protocol
# ---------------------------------------------------------------------------

class _CustomTokenExtractor:
    """Token extractor that looks for a 'tokens' key in dict results."""
    extracted = False

    def extract(self, span, result):
        _CustomTokenExtractor.extracted = False
        if isinstance(result, dict) and "tokens" in result:
            from ragwatch.instrumentation.attributes import safe_set_attribute
            safe_set_attribute(span, "custom.token_count", result["tokens"])
            _CustomTokenExtractor.extracted = True


def test_token_extractor_protocol():
    t = _CustomTokenExtractor()
    assert isinstance(t, TokenExtractor)


def test_custom_token_extractor_runs(_setup):
    """Registered token extractors run during span lifecycle."""
    exporter = _setup
    ext = _CustomTokenExtractor()
    register_token_extractor(ext)

    @trace("token-fn")
    def my_fn():
        return {"tokens": 42, "data": "hello"}

    my_fn()

    assert _CustomTokenExtractor.extracted is True
    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    attrs = dict(spans[-1].attributes)
    assert attrs.get("custom.token_count") == 42


def test_builtin_token_extraction_still_runs(_setup):
    """Built-in token extraction runs alongside custom extractors."""
    exporter = _setup
    register_token_extractor(_CustomTokenExtractor())

    class FakeMsg:
        usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    @trace("token-fn")
    def my_fn():
        return {"messages": [FakeMsg()]}

    my_fn()

    spans = exporter.get_finished_spans()
    attrs = dict(spans[-1].attributes)
    assert attrs.get("llm.token_count.prompt") == 10
    assert attrs.get("llm.token_count.completion") == 5
    assert attrs.get("llm.token_count.total") == 15


# ---------------------------------------------------------------------------
# global_auto_track_io
# ---------------------------------------------------------------------------

def test_global_auto_track_io_disables_io(_setup):
    """global_auto_track_io=False disables I/O tracking even when decorator says True."""
    exporter = _setup
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(global_auto_track_io=False)

    @trace("io-fn", auto_track_io=True)
    def my_fn(x):
        return x + 1

    my_fn(10)

    spans = exporter.get_finished_spans()
    attrs = dict(spans[-1].attributes)
    assert "input.value" not in attrs
    assert "output.value" not in attrs


def test_global_auto_track_io_default_enables(_setup):
    """With default config, I/O tracking is enabled."""
    exporter = _setup
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig()

    @trace("io-fn", auto_track_io=True)
    def my_fn(x):
        return x + 1

    my_fn(10)

    spans = exporter.get_finished_spans()
    attrs = dict(spans[-1].attributes)
    assert "input.value" in attrs
    assert "output.value" in attrs


def test_per_decorator_auto_track_io_false_respected(_setup):
    """Per-decorator auto_track_io=False still works."""
    exporter = _setup
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(global_auto_track_io=True)

    @trace("io-fn", auto_track_io=False)
    def my_fn(x):
        return x + 1

    my_fn(10)

    spans = exporter.get_finished_spans()
    attrs = dict(spans[-1].attributes)
    assert "input.value" not in attrs
    assert "output.value" not in attrs
