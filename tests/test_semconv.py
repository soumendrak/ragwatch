"""Tests for ragwatch.instrumentation.semconv — AST-based guards."""

from __future__ import annotations

import ast
from pathlib import Path

import ragwatch.instrumentation.semconv as semconv


def test_semconv_version_defined():
    assert hasattr(semconv, "SEMCONV_VERSION")
    assert semconv.SEMCONV_VERSION == "v1.40"


def test_all_semconv_keys_are_strings():
    for name in dir(semconv):
        if name.startswith("_") or name == "SEMCONV_VERSION":
            continue
        value = getattr(semconv, name)
        if isinstance(value, str):
            assert isinstance(value, str), f"{name} should be a string"


def test_attribute_stability_sets_are_disjoint():
    assert semconv.STANDARD_ATTRIBUTES.isdisjoint(semconv.STABLE_ATTRIBUTES)
    assert semconv.STANDARD_ATTRIBUTES.isdisjoint(semconv.EXPERIMENTAL_ATTRIBUTES)
    assert semconv.STABLE_ATTRIBUTES.isdisjoint(semconv.EXPERIMENTAL_ATTRIBUTES)


def test_all_attributes_registry_covers_known_attributes():
    expected = _get_semconv_values()
    assert expected <= semconv.ALL_ATTRIBUTES


def test_get_attribute_stability():
    assert (
        semconv.get_attribute_stability(semconv.OPENINFERENCE_SPAN_KIND) == "standard"
    )
    assert semconv.get_attribute_stability(semconv.USER_FEEDBACK_SCORE) == "stable"
    assert semconv.get_attribute_stability(semconv.QUERY_IS_CLEAR) == "experimental"
    assert semconv.get_attribute_stability("unknown.attribute") is None


def _get_ragwatch_source_files() -> list[Path]:
    """Return all .py files under ragwatch/, excluding semconv.py."""
    root = Path(__file__).resolve().parent.parent / "ragwatch"
    files = []
    for p in root.rglob("*.py"):
        if p.name == "semconv.py":
            continue
        files.append(p)
    return files


def _get_semconv_values() -> set[str]:
    """Return all string constant values defined in semconv.py."""
    values = set()
    for name in dir(semconv):
        if name.startswith("_") or name == "SEMCONV_VERSION":
            continue
        value = getattr(semconv, name)
        if isinstance(value, str):
            values.add(value)
    return values


def test_no_hardcoded_attribute_strings():
    """No semconv attribute strings should appear as literals outside semconv.py."""
    semconv_values = _get_semconv_values()
    violations = []

    for filepath in _get_ragwatch_source_files():
        source = filepath.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in semconv_values:
                    violations.append(
                        f"{filepath.relative_to(filepath.parent.parent)}:"
                        f"{node.lineno}: '{node.value}'"
                    )

    assert not violations, (
        "Hardcoded semconv strings found outside semconv.py:\n" + "\n".join(violations)
    )
