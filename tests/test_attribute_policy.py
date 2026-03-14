"""Tests for ragwatch.instrumentation.attribute_policy — schema & redaction."""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

import ragwatch
from ragwatch.core.config import RAGWatchConfig
from ragwatch.instrumentation.attribute_policy import (
    AttributePolicy,
    validate_attribute_name,
)
from ragwatch.instrumentation.attributes import safe_set_attribute, safe_set_attributes


# ---------------------------------------------------------------------------
# validate_attribute_name
# ---------------------------------------------------------------------------

class TestValidateAttributeName:

    def test_valid_simple(self):
        assert validate_attribute_name("ragwatch.custom.latency") is True

    def test_valid_two_segments(self):
        assert validate_attribute_name("custom.metric") is True

    def test_valid_single_segment(self):
        assert validate_attribute_name("latency") is True

    def test_valid_with_underscores(self):
        assert validate_attribute_name("my_company.rag_pipeline.chunk_count") is True

    def test_valid_with_numeric_index(self):
        assert validate_attribute_name("retrieval.chunk.0.score") is True

    def test_valid_with_multiple_numeric(self):
        assert validate_attribute_name("retrieval.chunk.0.embedding.1") is True

    def test_invalid_empty(self):
        assert validate_attribute_name("") is False

    def test_invalid_too_long(self):
        assert validate_attribute_name("a" * 129) is False

    def test_invalid_starts_with_number(self):
        assert validate_attribute_name("1invalid") is False

    def test_invalid_uppercase(self):
        assert validate_attribute_name("Ragwatch.Custom") is False

    def test_invalid_special_chars(self):
        assert validate_attribute_name("my-attr.name") is False

    def test_invalid_spaces(self):
        assert validate_attribute_name("my attr") is False

    def test_invalid_leading_dot(self):
        assert validate_attribute_name(".leading") is False

    def test_invalid_trailing_dot(self):
        assert validate_attribute_name("trailing.") is False

    def test_invalid_double_dot(self):
        assert validate_attribute_name("double..dot") is False

    def test_max_length_exactly_128(self):
        name = "a" * 128
        assert validate_attribute_name(name) is True


# ---------------------------------------------------------------------------
# AttributePolicy — truncation
# ---------------------------------------------------------------------------

class TestAttributePolicyTruncation:

    def test_short_string_unchanged(self):
        policy = AttributePolicy(max_value_bytes=100)
        assert policy.apply("key", "short") == "short"

    def test_long_string_truncated(self):
        policy = AttributePolicy(max_value_bytes=20)
        long_val = "a" * 100
        result = policy.apply("key", long_val)
        assert result.endswith("...[truncated]")
        assert len(result) < 100 + len("...[truncated]")

    def test_non_string_unchanged(self):
        policy = AttributePolicy(max_value_bytes=10)
        assert policy.apply("key", 42) == 42
        assert policy.apply("key", 3.14) == 3.14
        assert policy.apply("key", True) is True
        assert policy.apply("key", [1, 2, 3]) == [1, 2, 3]

    def test_default_max_bytes(self):
        policy = AttributePolicy()
        assert policy.max_value_bytes == 4096


# ---------------------------------------------------------------------------
# AttributePolicy — pattern-based redaction
# ---------------------------------------------------------------------------

class TestAttributePolicyPatternRedaction:

    def test_email_pattern_redacted(self):
        policy = AttributePolicy(
            redact_patterns=[r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"]
        )
        assert policy.apply("msg", "contact user@example.com") == "[REDACTED]"

    def test_no_match_not_redacted(self):
        policy = AttributePolicy(redact_patterns=[r"\d{3}-\d{2}-\d{4}"])
        assert policy.apply("msg", "no ssn here") == "no ssn here"

    def test_multiple_patterns(self):
        policy = AttributePolicy(
            redact_patterns=[r"password=\S+", r"secret=\S+"]
        )
        assert policy.apply("log", "password=abc123") == "[REDACTED]"
        assert policy.apply("log", "secret=xyz") == "[REDACTED]"
        assert policy.apply("log", "safe value") == "safe value"

    def test_pattern_not_applied_to_non_string(self):
        policy = AttributePolicy(redact_patterns=[r".*"])
        assert policy.apply("key", 42) == 42


# ---------------------------------------------------------------------------
# AttributePolicy — key-based redaction
# ---------------------------------------------------------------------------

class TestAttributePolicyKeyRedaction:

    def test_key_contains_password(self):
        policy = AttributePolicy(redact_keys=["password"])
        assert policy.apply("user.password", "secret123") == "[REDACTED]"

    def test_key_contains_secret(self):
        policy = AttributePolicy(redact_keys=["secret", "token"])
        assert policy.apply("api.secret_key", "abc") == "[REDACTED]"
        assert policy.apply("auth.token", "xyz") == "[REDACTED]"

    def test_key_no_match(self):
        policy = AttributePolicy(redact_keys=["password"])
        assert policy.apply("user.name", "alice") == "alice"

    def test_key_case_insensitive(self):
        policy = AttributePolicy(redact_keys=["password"])
        assert policy.apply("USER.PASSWORD", "val") == "[REDACTED]"

    def test_key_redaction_applies_to_non_strings(self):
        policy = AttributePolicy(redact_keys=["secret"])
        assert policy.apply("api.secret", 42) == "[REDACTED]"


# ---------------------------------------------------------------------------
# AttributePolicy — combined rules
# ---------------------------------------------------------------------------

class TestAttributePolicyCombined:

    def test_key_redaction_takes_precedence(self):
        """Key-based redaction fires before pattern/truncation."""
        policy = AttributePolicy(
            max_value_bytes=10,
            redact_patterns=[r"never_match"],
            redact_keys=["secret"],
        )
        assert policy.apply("my.secret", "short") == "[REDACTED]"

    def test_pattern_fires_before_truncation(self):
        policy = AttributePolicy(
            max_value_bytes=100,
            redact_patterns=[r"ssn:\d+"],
        )
        assert policy.apply("data", "ssn:123456789") == "[REDACTED]"

    def test_empty_policy_is_passthrough(self):
        policy = AttributePolicy()
        assert policy.apply("key", "value") == "value"
        assert policy.apply("key", 42) == 42


# ---------------------------------------------------------------------------
# safe_set_attribute — centralized writer
# ---------------------------------------------------------------------------

class TestSafeSetAttribute:

    def setup_method(self):
        self._prev = ragwatch._ACTIVE_CONFIG

    def teardown_method(self):
        ragwatch._ACTIVE_CONFIG = self._prev

    def test_passthrough_without_policy(self):
        ragwatch._ACTIVE_CONFIG = None
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attribute(span, "my.key", "hello")
        span.set_attribute.assert_called_once_with("my.key", "hello")

    def test_policy_truncates_value(self):
        policy = AttributePolicy(max_value_bytes=10)
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attribute(span, "my.key", "a" * 100)
        written_value = span.set_attribute.call_args[0][1]
        assert written_value.endswith("...[truncated]")
        assert len(written_value) < 100

    def test_policy_redacts_by_key(self):
        policy = AttributePolicy(redact_keys=["secret"])
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attribute(span, "api.secret", "my_token_123")
        span.set_attribute.assert_called_once_with("api.secret", "[REDACTED]")

    def test_explicit_policy_overrides_global(self):
        global_policy = AttributePolicy(redact_keys=["secret"])
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=global_policy)
        local_policy = AttributePolicy()  # no redaction
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attribute(span, "api.secret", "visible", policy=local_policy)
        span.set_attribute.assert_called_once_with("api.secret", "visible")

    def test_skips_non_recording_span(self):
        span = MagicMock()
        span.is_recording.return_value = False
        safe_set_attribute(span, "my.key", "value")
        span.set_attribute.assert_not_called()

    def test_safe_set_attributes_batch(self):
        ragwatch._ACTIVE_CONFIG = None
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attributes(span, {"a.key": 1, "b.key": 2})
        assert span.set_attribute.call_count == 2

    def test_policy_flows_through_helpers(self):
        """Verify that helpers using safe_set_attribute pick up the global policy."""
        policy = AttributePolicy(redact_keys=["question"])
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(attribute_policy=policy)
        span = MagicMock()
        span.is_recording.return_value = True
        # record_routing uses safe_set_attribute internally
        from ragwatch.instrumentation.helpers import record_routing
        record_routing("node_a", "node_b", reason="test", span=span)
        # All values should have been set (none redacted since keys don't match)
        assert span.set_attribute.call_count >= 2


# ---------------------------------------------------------------------------
# Strict mode: invalid attribute names are skipped
# ---------------------------------------------------------------------------

class TestStrictInvalidAttributeNames:

    def setup_method(self):
        self._prev = ragwatch._ACTIVE_CONFIG

    def teardown_method(self):
        ragwatch._ACTIVE_CONFIG = self._prev

    def test_strict_mode_skips_invalid_name(self):
        """In strict mode with a policy, invalid attribute names are skipped."""
        policy = AttributePolicy()
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(
            attribute_policy=policy, strict_mode=True,
        )
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attribute(span, "INVALID-NAME!", "value")
        span.set_attribute.assert_not_called()
        # Should record an event
        span.add_event.assert_called_once()
        event_args = span.add_event.call_args
        assert event_args[0][0] == "ragwatch.invalid_attribute"

    def test_non_strict_mode_still_writes_invalid_name(self):
        """In non-strict mode, invalid names are warned but still written."""
        policy = AttributePolicy()
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(
            attribute_policy=policy, strict_mode=False,
        )
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attribute(span, "INVALID-NAME!", "value")
        span.set_attribute.assert_called_once_with("INVALID-NAME!", "value")

    def test_strict_mode_allows_valid_name(self):
        """Valid attribute names pass through even in strict mode."""
        policy = AttributePolicy()
        ragwatch._ACTIVE_CONFIG = RAGWatchConfig(
            attribute_policy=policy, strict_mode=True,
        )
        span = MagicMock()
        span.is_recording.return_value = True
        safe_set_attribute(span, "my.valid.name", "value")
        span.set_attribute.assert_called_once_with("my.valid.name", "value")


# ---------------------------------------------------------------------------
# Collection/cardinality controls
# ---------------------------------------------------------------------------

class TestCollectionControls:

    def test_list_truncated_at_max_list_length(self):
        policy = AttributePolicy(max_list_length=3)
        result = policy.apply("my.key", [1, 2, 3, 4, 5])
        assert result == [1, 2, 3]

    def test_list_under_limit_unchanged(self):
        policy = AttributePolicy(max_list_length=10)
        result = policy.apply("my.key", [1, 2, 3])
        assert result == [1, 2, 3]

    def test_tuple_truncated_at_max_list_length(self):
        policy = AttributePolicy(max_list_length=2)
        result = policy.apply("my.key", (10, 20, 30, 40))
        assert result == [10, 20]

    def test_default_max_list_length(self):
        policy = AttributePolicy()
        assert policy.max_list_length == 128

    def test_list_truncation_with_redact_keys(self):
        """Key-based redaction takes precedence over list truncation."""
        policy = AttributePolicy(max_list_length=2, redact_keys=["secret"])
        result = policy.apply("my.secret", [1, 2, 3])
        assert result == "[REDACTED]"
