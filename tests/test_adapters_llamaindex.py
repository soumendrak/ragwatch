"""Tests for ragwatch.adapters.llamaindex.

LlamaIndex itself is an optional extra, so the callback handler is driven
with the same ``on_event_start`` / ``on_event_end`` calls (and payload
shapes) that LlamaIndex's ``CallbackManager`` makes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tests.conftest import InMemorySpanExporter

from ragwatch.adapters.base import FrameworkAdapter
from ragwatch.adapters.llamaindex import (
    LlamaIndexAdapter,
    RagWatchCallbackHandler,
    chat_engine,
    ingestion,
    query_engine,
    retriever,
)
from ragwatch.core.config import RAGWatchConfig
from ragwatch.core.context import get_query_embedding
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider
from ragwatch.instrumentation.semconv import (
    CHUNK_RELEVANCE_SCORES,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    ERROR_TYPE,
    INPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_VALUE,
    RESPONSE_LENGTH,
    RETRIEVAL_CHUNKS_RETURNED,
)


@pytest.fixture(autouse=True)
def _setup():
    exporter = InMemorySpanExporter()
    configure_tracer(
        RAGWatchConfig(service_name="t", exporter=exporter), _force_flush=True
    )
    yield exporter
    reset_tracer_provider()


def _node(text, score=None, embedding=None, **meta):
    inner = SimpleNamespace(
        get_content=lambda: text,
        metadata=meta,
        ref_doc_id="doc-1",
        embedding=embedding,
    )
    return SimpleNamespace(node=inner, score=score)


def _spans(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


# -- adapter ----------------------------------------------------------------


def test_adapter_is_framework_adapter():
    assert isinstance(LlamaIndexAdapter(), FrameworkAdapter)
    assert LlamaIndexAdapter().name == "llamaindex"


def test_adapter_extract_state():
    adapter = LlamaIndexAdapter()
    assert adapter.extract_state((), {"q": 1}) == {"q": 1}
    assert adapter.extract_state(("x", {"a": 1}), {}) == {"a": 1}
    assert adapter.extract_state(("x",), {}) is None


def test_adapter_normalize_response_object():
    tool = SimpleNamespace(tool_name="search")
    raw = SimpleNamespace(response="42", sources=[tool])
    norm = LlamaIndexAdapter().normalize_result(raw, None)
    assert norm == {
        "agent_answer": "42",
        "is_fallback": False,
        "tool_calls": [{"name": "search"}],
    }


def test_adapter_normalize_completion_and_unknown():
    adapter = LlamaIndexAdapter()
    assert adapter.normalize_result(SimpleNamespace(text=""), None) == {
        "agent_answer": "",
        "is_fallback": True,
    }
    assert adapter.normalize_result(object(), None) is None


# -- decorators -------------------------------------------------------------


@pytest.mark.parametrize(
    ("decorator", "kind"),
    [
        (query_engine, "CHAIN"),
        (retriever, "RETRIEVER"),
        (chat_engine, "AGENT"),
        (ingestion, "EMBEDDING"),
    ],
)
def test_decorator_span_kind(_setup, decorator, kind):
    @decorator("li-step")
    def step(q):
        return q

    assert step("hi") == "hi"
    span = _spans(_setup)["li-step"]
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == kind


async def test_decorator_bare_async(_setup):
    @query_engine
    async def ask(q):
        return q.upper()

    assert await ask("hi") == "HI"
    assert any("ask" in name for name in _spans(_setup))


# -- callback handler -------------------------------------------------------


def test_handler_query_retrieve_synthesize(_setup):
    h = RagWatchCallbackHandler()
    h.on_event_start("query", {"query_str": "what?"}, event_id="q", parent_id="root")
    h.on_event_start("retrieve", {"query_str": "what?"}, event_id="r", parent_id="q")
    nodes = [_node("a", 0.9, file_name="a.md"), _node("b", 0.7)]
    h.on_event_end("retrieve", {"nodes": nodes}, event_id="r")
    h.on_event_start("synthesize", {"query_str": "what?"}, event_id="s", parent_id="q")
    h.on_event_end("synthesize", {"response": SimpleNamespace(response="yes")}, "s")
    h.on_event_end("query", {"response": "yes"}, event_id="q")

    spans = _spans(_setup)
    query, retrieve, synth = (
        spans["llama_index.query"],
        spans["llama_index.retrieve"],
        spans["llama_index.synthesize"],
    )
    assert retrieve.parent.span_id == query.context.span_id
    assert synth.parent.span_id == query.context.span_id
    assert retrieve.attributes[OPENINFERENCE_SPAN_KIND] == "RETRIEVER"
    assert retrieve.attributes[INPUT_VALUE] == "what?"
    assert retrieve.attributes[RETRIEVAL_CHUNKS_RETURNED] == 2
    assert retrieve.attributes["retrieval.chunk.0.source"] == "a.md"
    assert list(retrieve.attributes[CHUNK_RELEVANCE_SCORES]) == [0.9, 0.7]
    assert synth.attributes[OUTPUT_VALUE] == "yes"
    assert synth.attributes[RESPONSE_LENGTH] == 3


def test_handler_empty_retrieval_records_zero(_setup):
    h = RagWatchCallbackHandler()
    h.on_event_start("retrieve", {"query_str": "x"}, event_id="r")
    h.on_event_end("retrieve", {"nodes": []}, event_id="r")
    span = _spans(_setup)["llama_index.retrieve"]
    assert span.attributes[RETRIEVAL_CHUNKS_RETURNED] == 0


def test_handler_query_embedding_scores_unscored_nodes(_setup):
    h = RagWatchCallbackHandler()
    h.on_event_start("retrieve", {"query_str": "x"}, event_id="r")
    h.on_event_start("embedding", {"serialized": {"model_name": "m"}}, "e", "r")
    h.on_event_end("embedding", {"embeddings": [[1.0, 0.0]]}, event_id="e")
    assert get_query_embedding() == [1.0, 0.0]
    nodes = [_node("a", embedding=[1.0, 0.0]), _node("b", embedding=[0.0, 1.0])]
    h.on_event_end("retrieve", {"nodes": nodes}, event_id="r")

    spans = _spans(_setup)
    emb = spans["llama_index.embedding"]
    assert emb.attributes[OPENINFERENCE_SPAN_KIND] == "EMBEDDING"
    assert emb.attributes[EMBEDDING_MODEL_NAME] == "m"
    assert emb.attributes[EMBEDDING_DIMENSIONS] == 2
    scores = list(spans["llama_index.retrieve"].attributes[CHUNK_RELEVANCE_SCORES])
    assert scores == pytest.approx([1.0, 0.0])


def test_handler_records_exception(_setup):
    h = RagWatchCallbackHandler()
    h.on_event_start("llm", {"messages": ["hi"]}, event_id="l")
    h.on_event_end("llm", {"exception": ValueError("boom")}, event_id="l")
    span = _spans(_setup)["llama_index.llm"]
    assert span.attributes[ERROR_TYPE] == "ValueError"
    assert span.attributes[INPUT_VALUE] == "['hi']"
    assert not span.status.is_ok


def test_handler_unknown_end_is_ignored(_setup):
    RagWatchCallbackHandler().on_event_end("query", {}, event_id="missing")
    assert not _setup.get_finished_spans()


async def test_handler_concurrent_queries_do_not_share_spans(_setup):
    h = RagWatchCallbackHandler()

    async def run(i):
        h.on_event_start("query", {"query_str": f"q{i}"}, event_id=f"q{i}")
        h.on_event_start("retrieve", {}, event_id=f"r{i}", parent_id=f"q{i}")
        await asyncio.sleep(0)
        h.on_event_end("retrieve", {"nodes": []}, event_id=f"r{i}")
        h.on_event_end("query", {"response": f"a{i}"}, event_id=f"q{i}")

    await asyncio.gather(*(run(i) for i in range(3)))
    spans = _setup.get_finished_spans()
    queries = {s.attributes[INPUT_VALUE]: s for s in spans if s.name.endswith("query")}
    for s in spans:
        if s.name.endswith("retrieve"):
            parent = next(
                q for q in queries.values() if q.context.span_id == s.parent.span_id
            )
            assert parent is not None
    assert len(queries) == 3
