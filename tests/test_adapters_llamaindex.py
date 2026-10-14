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
    h.on_event_start("query", {"query_str": "hi"}, event_id="q")
    h.on_event_end("query", {"exception": ValueError("boom")}, event_id="q")
    span = _spans(_setup)["llama_index.query"]
    assert span.attributes[ERROR_TYPE] == "ValueError"
    assert span.attributes[INPUT_VALUE] == "hi"
    assert not span.status.is_ok


def test_handler_skips_llm_and_reparents_children(_setup):
    h = RagWatchCallbackHandler()
    h.on_event_start("query", {"query_str": "q"}, event_id="q")
    h.on_event_start("llm", {"messages": ["hi"]}, event_id="l", parent_id="q")
    h.on_event_start("templating", {}, event_id="t", parent_id="l")
    h.on_event_end("templating", {}, event_id="t")
    h.on_event_end("llm", {"response": "x"}, event_id="l")
    h.on_event_end("query", {"response": "a"}, event_id="q")
    spans = _spans(_setup)
    assert "llama_index.llm" not in spans
    child = spans["llama_index.templating"]
    assert child.parent.span_id == spans["llama_index.query"].context.span_id


def test_handler_function_call_is_tool_span(_setup):
    h = RagWatchCallbackHandler()
    tool = SimpleNamespace(name="search")
    h.on_event_start(
        "function_call", {"function_call": {"q": 1}, "tool": tool}, event_id="f"
    )
    h.on_event_end("function_call", {"function_call_response": "found"}, "f")
    span = _spans(_setup)["llama_index.function_call"]
    assert span.attributes[OPENINFERENCE_SPAN_KIND] == "TOOL"
    assert span.attributes["tool.name"] == "search"
    assert span.attributes[INPUT_VALUE] == "{'q': 1}"
    assert span.attributes[OUTPUT_VALUE] == "found"


def test_handler_does_not_drain_streaming_response(_setup):
    def gen():
        yield "tok"

    stream = SimpleNamespace(response_gen=gen())
    h = RagWatchCallbackHandler()
    h.on_event_start("query", {"query_str": "q"}, event_id="q")
    h.on_event_end("query", {"response": stream}, event_id="q")
    assert list(stream.response_gen) == ["tok"]
    assert OUTPUT_VALUE not in _spans(_setup)["llama_index.query"].attributes


def test_handler_respects_global_auto_track_io_off(_setup, monkeypatch):
    import ragwatch

    cfg = RAGWatchConfig(service_name="t", global_auto_track_io=False)
    monkeypatch.setattr(ragwatch, "_ACTIVE_CONFIG", cfg)
    h = RagWatchCallbackHandler()
    h.on_event_start("query", {"query_str": "secret"}, event_id="q")
    h.on_event_end("query", {"response": "answer"}, event_id="q")
    span = _spans(_setup)["llama_index.query"]
    assert INPUT_VALUE not in span.attributes
    assert OUTPUT_VALUE not in span.attributes
    assert span.attributes[RESPONSE_LENGTH] == 6


def test_handler_does_not_reuse_previous_query_embedding(_setup):
    h = RagWatchCallbackHandler()
    h.on_event_start("retrieve", {"query_str": "a"}, event_id="r1")
    h.on_event_start("embedding", {}, event_id="e1", parent_id="r1")
    h.on_event_end("embedding", {"embeddings": [[1.0, 0.0]]}, event_id="e1")
    h.on_event_end("retrieve", {"nodes": []}, event_id="r1")
    # Second retrieval: unscored nodes and no embedding event of its own.
    h.on_event_start("retrieve", {"query_str": "b"}, event_id="r2")
    nodes = [_node("x", embedding=[1.0, 0.0])]
    h.on_event_end("retrieve", {"nodes": nodes}, event_id="r2")
    spans = [s for s in _setup.get_finished_spans() if s.name.endswith("retrieve")]
    assert CHUNK_RELEVANCE_SCORES not in spans[-1].attributes
    assert spans[-1].attributes[RETRIEVAL_CHUNKS_RETURNED] == 1


def test_decorators_resolve_llamaindex_adapter():
    from ragwatch.instrumentation.decorators import _resolve_adapter

    assert isinstance(_resolve_adapter("llamaindex"), LlamaIndexAdapter)


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
