"""Tests for ragwatch.core.context — context-local query embedding storage."""

from __future__ import annotations

import asyncio
import threading

from ragwatch.core.context import (
    clear_query_embedding,
    get_query_embedding,
    set_query_embedding,
)


def test_context_set_and_get():
    embedding = [0.1, 0.2, 0.3]
    set_query_embedding(embedding)
    try:
        assert get_query_embedding() == embedding
    finally:
        clear_query_embedding()


def test_context_stores_vector():
    embedding = [float(i) for i in range(512)]
    set_query_embedding(embedding)
    try:
        result = get_query_embedding()
        assert result is not None
        assert len(result) == 512
        assert result == embedding
    finally:
        clear_query_embedding()


def test_context_is_thread_local():
    embedding = [1.0, 2.0, 3.0]
    set_query_embedding(embedding)
    other_thread_result = [None]

    def check():
        other_thread_result[0] = get_query_embedding()

    t = threading.Thread(target=check)
    t.start()
    t.join()

    try:
        assert (
            other_thread_result[0] is None
        ), "Query embedding should not leak across threads"
    finally:
        clear_query_embedding()


def test_context_returns_none_when_not_set():
    clear_query_embedding()
    assert get_query_embedding() is None


async def test_context_is_isolated_across_concurrent_tasks():
    async def pipeline(i: int) -> list[float] | None:
        set_query_embedding([float(i)] * 3)
        for _ in range(3):
            await asyncio.sleep(0)
        return get_query_embedding()

    clear_query_embedding()
    results = await asyncio.gather(*(pipeline(i) for i in range(5)))

    assert results == [[float(i)] * 3 for i in range(5)]
    assert get_query_embedding() is None
