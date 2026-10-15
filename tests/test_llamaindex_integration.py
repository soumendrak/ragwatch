"""Regression tests against real llama-index-core (skipped when not installed)."""

from __future__ import annotations

import os

import pytest

# CI sets RAGWATCH_REQUIRE_LLAMAINDEX=1 so a missing install fails loudly
# instead of silently skipping these tests.
if os.environ.get("RAGWATCH_REQUIRE_LLAMAINDEX") != "1":
    pytest.importorskip("llama_index.core")

from llama_index.core import Document  # noqa: E402
from llama_index.core.callbacks import CallbackManager  # noqa: E402
from llama_index.core.embeddings import MockEmbedding  # noqa: E402
from llama_index.core.ingestion import IngestionPipeline  # noqa: E402
from tests.conftest import InMemorySpanExporter  # noqa: E402

from ragwatch.adapters.llamaindex import RagWatchCallbackHandler  # noqa: E402
from ragwatch.adapters.llamaindex.hooks import EVICTED_ATTRIBUTE  # noqa: E402
from ragwatch.core.config import RAGWatchConfig  # noqa: E402
from ragwatch.core.tracer import configure_tracer, reset_tracer_provider  # noqa: E402
from ragwatch.instrumentation.semconv import EMBEDDING_DIMENSIONS  # noqa: E402


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    configure_tracer(RAGWatchConfig(service_name="t", exporter=exp), _force_flush=True)
    yield exp
    reset_tracer_provider()


async def _ingest(handler, n_docs=3000):
    embed = MockEmbedding(embed_dim=8, callback_manager=CallbackManager([handler]))
    pipeline = IngestionPipeline(transformations=[embed])
    docs = [Document(text=f"doc {i}") for i in range(n_docs)]
    return await pipeline.arun(documents=docs)


async def test_async_ingestion_of_3000_docs_evicts_nothing(exporter):
    nodes = await _ingest(RagWatchCallbackHandler())
    assert len(nodes) == 3000
    spans = [
        s for s in exporter.get_finished_spans() if s.name == "llama_index.embedding"
    ]
    assert len(spans) == 300  # default embed_batch_size is 10
    assert not any(s.attributes.get(EVICTED_ATTRIBUTE) for s in spans)
    assert all(s.attributes.get(EMBEDDING_DIMENSIONS) == 8 for s in spans)


async def test_async_ingestion_with_small_cap_does_evict(exporter):
    # The batch really opens every event at once; a 256 cap would evict.
    await _ingest(RagWatchCallbackHandler(max_open_events=256))
    spans = [
        s for s in exporter.get_finished_spans() if s.name == "llama_index.embedding"
    ]
    assert any(s.attributes.get(EVICTED_ATTRIBUTE) for s in spans)


def test_real_pydantic_response_output_is_scrubbed(exporter, monkeypatch):
    import ragwatch
    from llama_index.core.base.response.schema import PydanticResponse
    from pydantic import BaseModel

    from ragwatch.instrumentation.attribute_policy import AttributePolicy
    from ragwatch.instrumentation.semconv import OUTPUT_VALUE

    class Answer(BaseModel):
        answer: str
        api_key: str

    cfg = RAGWatchConfig(
        service_name="t", attribute_policy=AttributePolicy(redact_io_keys=["api_key"])
    )
    monkeypatch.setattr(ragwatch, "_ACTIVE_CONFIG", cfg)
    h = RagWatchCallbackHandler()
    h.on_event_start("synthesize", {}, event_id="s")
    out = PydanticResponse(response=Answer(answer="hi", api_key="sk-secret"))
    h.on_event_end("synthesize", {"response": out}, event_id="s")
    (span,) = exporter.get_finished_spans()
    assert "sk-secret" not in span.attributes[OUTPUT_VALUE]
