# AGENTS.md

Repository-wide instructions for coding agents and contributors.

## Primary Task Context

The explicit user request or GitHub issue is the primary task packet. Read it fully and treat its scope, base commit/branch, acceptance criteria, and non-goals as authoritative unless current source evidence contradicts it.

Do **not** read every project-history document before starting work.

## Context Tiers

### Hot context — read routinely

1. the current issue / user task;
2. this `AGENTS.md`;
3. the source and tests directly relevant to the task.

If the issue links a specific research note or runbook, read that targeted document or section.

### Cold context — retrieve only when needed

- `CURRENT_STATE.md` — current cross-project facts and accepted baselines;
- `PLANS.md` — broader priorities and deferred work;
- `DECISIONS.md` and `.agents/notes/**` — durable rationale and rejected alternatives;
- `MECHANICS.md` — verified engine/mechanics evidence;
- `HISTORY.md` — prior implementation/results/failures;
- `research/**` and `docs/**` — detailed reports and runbooks.

Search/grep for relevant headings, symbols, issue numbers, or terms first, then read only the useful section. Do not sequentially or wholesale ingest these files merely "for context." The issue packet should normally contain enough task-local context to begin source inspection.

## Source of Truth

For current behavior/mechanics, prefer:

1. current source at the task's actual base/ref;
2. controlled behavioral tests or reproduced traces;
3. pinned official specification/engine source;
4. durable project notes and prior reports.

Historical docs explain why something exists; they do not override current code or fresh evidence.

## Scope Discipline

- Work in one bounded packet at a time.
- Follow the issue's explicit base/branch/worktree constraints and avoid disturbing unrelated concurrent work.
- Prefer the simplest change that satisfies the stated contract.
- Do not quietly expand research into implementation or a bug fix into architecture redesign.
- Do not modify vendored third-party engine/baseline artifacts in place.
- Preserve original artifacts; record derived/local patches separately.

## Validation

Use the smallest realistic validation that proves the changed contract.

- Prefer focused behavioral tests and real execution paths over giant test matrices.
- Reuse still-valid evidence; do not rerun broad suites only for completeness.
- If a change affects lifecycle, persistence/resume, multiprocessing, packaging, or engine semantics, include a small real-path check when practical.
- Competitive evaluations should use fixed recorded seeds, both seat assignments when relevant, immutable identities, exact engine provenance, and machine-readable results.
- PASS/random agents are plumbing smokes, not competitive evidence.
- When the issue names official `kaggle_environments==1.32.7` as promotion authority, fast-engine score magnitude is diagnostic only.

## Expensive Runs and Kaggle

Before a genuinely expensive run, ensure the exact command/config, branch+commit, engine/artifact identities, seed/seat policy, outputs, and stop/recovery conditions are recorded somewhere appropriate for that task. Do not launch from remembered configuration when repository/run artifacts disagree.

For Kaggle work:

- preserve attached datasets and accelerator/notebook metadata unless explicitly changing them;
- use Kaggle secrets for Git authentication;
- prefer a fresh clone over risky in-notebook repository surgery;
- keep long runs foregrounded unless background execution is explicitly requested;
- log enough phase/timing/output information to diagnose a failed run.

## Durable Documentation

Update durable docs **only when the task materially changes durable knowledge**. Do not treat every file below as a mandatory completion checklist.

- `CURRENT_STATE.md`: concise current truths/baselines; replace stale facts rather than accumulating history.
- `DECISIONS.md` / `.agents/notes/**`: decisions whose rationale or rejected alternatives will matter later.
- `MECHANICS.md`: verified mechanics/evidence/regressions.
- `HISTORY.md`: notable completed work, measurements, failures, and artifact identities worth preserving.
- `PLANS.md`: only when project priorities or sequencing actually change.
- `research/**`: detailed experiment reports/runbooks when the issue needs them.

Avoid duplicating the same fact across several documents. Prefer pointers to one authoritative detailed record.

## Coding Expectations

- Python 3.12 unless the live environment requires otherwise.
- Deterministic behavior by default.
- Type hints for public interfaces.
- Clear failures and explicit validation at contract boundaries.
- Version/schema fields for persisted JSON/JSONL/checkpoints when semantics matter.
- Avoid hidden global state and unrecorded randomness.

## Submission Safety

Before packaging a Kaggle submission, validate representative `agent(obs)` calls, clean-environment imports/paths, no unintended network dependency, runtime/memory limits, deterministic fallback behavior, and exact archive/source provenance.

## Communication

Be direct about uncertainty, failed checks, unavailable artifacts, and evidence limits. Partial verified progress is better than a confident guess.