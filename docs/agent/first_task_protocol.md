# First Task Protocol (Agent Onboarding Procedure)

Use this protocol when an agent touches the repository for the first time.

## Phase A: Context Load

1. Read:
- `docs/agent/context_index.yaml`
- `docs/agent/engineering_contract.md`
- `README.md`
- `docs/dev/status.md`
- `docs/CLI.md`

2. Confirm high-level map:
- CLI dispatch: `src/od_training/cli/main.py`
- Feature modules: `src/od_training/{dataset,train,infer,utility}`
- External references policy: `external/README.md` (reference-only clones; not dependencies)

## Phase B: Environment Sanity

Run:
```bash
venv/bin/python scripts/odt.py utility verify-env
```

If unavailable, use local equivalent and report what failed.

## Phase C: Task Scoping

1. Identify affected layer(s):
- dataset
- training
- inference
- utility
- CLI routing

2. Identify impacted contracts:
- command-line surface
- config behavior
- export semantics
- test expectations
- documentation

## Phase D: Implementation Rules

- Prefer edits in `src/od_training/*`.
- Keep compatibility modules in `src/*` aligned when shared behavior changes.
- Keep docstrings accurate and concise.
- Avoid changing historical docs unless task requires it.

## Phase E: Validation

Minimum validation set:
1. Targeted unit tests for changed modules.
2. Any integration test directly related to changed behavior.
3. Lint/syntax sanity for edited Python files.

## Phase F: Documentation Sync

If applicable, update in same PR/commit:
- `docs/CLI.md` for command changes
- `docs/src/*` for module contract changes
- `docs/dev/status.md` and `docs/dev/progress.md` for significant capability/state updates
- `docs/agent/*` when onboarding context or engineering rules change

## Phase G: Handoff Output

Agent completion summary should include:
- files changed
- behavior changes
- tests run
- remaining risks/open assumptions
