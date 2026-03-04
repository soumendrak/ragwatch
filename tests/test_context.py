"""Tests for ragwatch.core.context — thread-local query embedding storage."""

from __future__ import annotations

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
