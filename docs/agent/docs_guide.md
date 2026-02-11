# Documentation Guide For Agents

This file classifies documentation by trust level and usage intent.

## Trust Levels

- `L1 (authoritative)`: use as current source of truth.
- `L2 (operational)`: accurate usage references, may lag small internal changes.
- `L3 (historical/supporting)`: useful context but not authoritative for current state.

## Docs Directory Map

| Path | Level | Purpose | Agent Usage Rule |
|---|---|---|---|
| `README.md` | L1 | Root navigation and command patterns | Read at task start |
| `docs/dev/status.md` | L1 | Current implementation status | Prefer over older planning notes |
| `docs/dev/progress.md` | L1 | Chronological project change log | Use for recency/context |
| `docs/CLI.md` | L1 | CLI command contracts and options | Validate CLI changes against this |
| `docs/src/README.md` | L2 | API docs index | Use for module-level orientation |
| `docs/src/*` | L2 | Module-level docs snapshots | Cross-check function/module behavior |
| `docs/dev/training_cookbook.md` | L2 | Training workflow methodology | Use for tuning/process guidance |
| `docs/dev/research.md` | L3 | Curated external/technical findings | Use as supporting rationale only |
| `docs/dev/discovery.md` | L3 | Discovery decisions and constraints | Use to understand historical intent |
| `docs/dev/audit_report.md` | L3 | 2026-02-11 audit snapshot | Treat as historical snapshot |
| `docs/context/main.md` | L3 | Full historical context thread | Optional deep context when blocked |
| `external/README.md` | L2 | External reference repo policy and sync source list | Follow for upstream code inspection only; never treat as runtime dependency |

## Documentation Consumption Strategy

1. Start with `L1` docs only.
2. Use `L2` docs for implementation details and workflow conventions.
3. Use `L3` docs only when needing historical rationale or unresolved ambiguity.
4. If `L1` and `L3` conflict, follow `L1`.

## Documentation Update Expectations

When making code changes:
- CLI contract changes: update `docs/CLI.md`.
- Module contract or behavior changes: update `docs/src/*` docs for impacted modules.
- State/progress changes: update `docs/dev/status.md` and `docs/dev/progress.md`.
- New cross-cutting agent guidance: update docs in `docs/agent/`.

## Redundancy Policy

- Prefer linking to canonical docs instead of copying text.
- Put rules and routing in `docs/agent/*`.
- Keep detailed usage examples in `docs/CLI.md` and module specifics in `docs/src/*`.
