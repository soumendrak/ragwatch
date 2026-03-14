"""Tests for RAGWatchRuntime scoped accessor."""

from __future__ import annotations

import pytest

import ragwatch
from ragwatch import RAGWatchRuntime
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.attribute_policy import AttributePolicy
from tests.conftest import InMemorySpanExporter


@pytest.fixture(autouse=True)
def _setup():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="runtime-test", exporter=exporter),
        _force_flush=True,
    )
    ragwatch._ACTIVE_CONFIG = RAGWatchConfig(
        service_name="runtime-test",
        strict_mode=True,
        global_auto_track_io=False,
        attribute_policy=AttributePolicy(max_value_bytes=64),
    )
    yield
    ragwatch._ACTIVE_CONFIG = None
    reset_tracer_provider()


def test_current_returns_runtime():
    rt = RAGWatchRuntime.current()
    assert isinstance(rt, RAGWatchRuntime)


def test_config_returns_active_config():
    rt = RAGWatchRuntime.current()
    assert rt.config is not None
    assert rt.config.service_name == "runtime-test"


def test_strict_mode():
    rt = RAGWatchRuntime.current()
    assert rt.strict_mode is True


def test_attribute_policy():
    rt = RAGWatchRuntime.current()
    assert rt.attribute_policy is not None
    assert rt.attribute_policy.max_value_bytes == 64


def test_auto_track_io():
    rt = RAGWatchRuntime.current()
    assert rt.auto_track_io is False


def test_extractor_registry():
    rt = RAGWatchRuntime.current()
    reg = rt.extractor_registry
    assert reg is not None
    assert hasattr(reg, "register")
    assert hasattr(reg, "extract_all")


def test_adapter_registry():
    rt = RAGWatchRuntime.current()
    adapters = rt.adapter_registry
    assert isinstance(adapters, dict)


def test_transformer_registry():
    rt = RAGWatchRuntime.current()
    reg = rt.transformer_registry
    assert reg is not None
    assert hasattr(reg, "register")
    assert hasattr(reg, "get")


def test_repr():
    rt = RAGWatchRuntime.current()
    assert "runtime-test" in repr(rt)


def test_unconfigured_defaults():
    ragwatch._ACTIVE_CONFIG = None
    rt = RAGWatchRuntime.current()
    assert rt.config is None
    assert rt.strict_mode is False
    assert rt.attribute_policy is None
    assert rt.auto_track_io is True
    assert "unconfigured" in repr(rt)


def test_token_extractor_registry():
    rt = RAGWatchRuntime.current()
    extractors = rt.token_extractor_registry
    assert isinstance(extractors, list)
