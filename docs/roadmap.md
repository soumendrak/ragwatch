# Roadmap

RAGWatch is currently focused on making the SDK runtime and extension contracts
stable before adding more framework integrations.

## v0.4 Runtime Hardening

- Return `RAGWatchRuntime` from `configure()`.
- Treat `InstrumentationContext` as the canonical extension contract.
- Keep built-in framework decorators self-contained so users do not need to
  manually register built-in adapters.
- Use async-safe query embedding context for concurrent RAG pipelines.

## v0.5 Semantic Schema Stabilization

- Finalize stable vs experimental attribute groups. Initial machine-readable
  groups are available via `STANDARD_ATTRIBUTES`, `STABLE_ATTRIBUTES`,
  `EXPERIMENTAL_ATTRIBUTES`, and `get_attribute_stability()`.
- Add compatibility tests for documented telemetry examples. README-style
  minimal RAG, LangGraph decorator, and runtime tracing examples are covered.
- Clarify feedback correlation semantics and add span-link support where useful.

## v0.6 Adapter Expansion

- Harden adapter capability negotiation. Initial mismatch reporting is emitted
  as `ragwatch.unsupported_telemetry` span events.
- Add cookbook-style examples for production LangGraph and CrewAI workflows.
  Initial production cookbook is available in `docs/production-cookbook.md`.
- Evaluate additional adapters based on real user demand.
