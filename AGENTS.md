# AGENTS.md

Instructions for any chat, coding agent, or contributor working in this repository.

## Read Order

Before substantial work, read:

1. `CURRENT_STATE.md`
2. `DECISIONS.md`
3. `PLANS.md`
4. `MECHANICS.md`
5. the newest relevant section of `HISTORY.md`

Do not rely on chat memory when these files provide a durable answer.

## Current Project Mode

The project is presently in a bounded implementation phase for the first closed-loop BC executor.

Broad implementation, reinforcement learning, expensive experiments, search/VRP optimization, opponent modeling, and BC architecture changes remain unauthorized unless the user explicitly expands scope.

**Codex authorization:** the user explicitly authorized the bounded implementation packet in GitHub issue #1, `Codex packet: minimal closed-loop BC executor V0`. Codex may implement exactly that packet and should stop at its acceptance criteria. No other speculative Codex work is authorized by this note.

## Core Technical Direction

Prefer:

- exact engine tracking;
- deterministic evaluation;
- strong public-baseline reproduction;
- structured route execution;
- closed-loop repair;
- opponent-aware macro planning;
- market simulation and optimization.

Do not assume primitive-action deep RL is the correct starting point.

## Scope Discipline

- Work in one bounded packet at a time.
- Avoid speculative abstractions before the first concrete use case.
- Do not quietly expand a research task into implementation.
- Do not modify vendored third-party engine or baseline files in place.
- Preserve original artifacts and record local patches separately.
- Prefer simple executable smoke scripts over enormous test suites when a direct behavioral test is clearer and faster.

## Source of Truth

Mechanics priority:

1. exact current engine source;
2. controlled behavioral test;
3. official current specification;
4. discussion or notebook claims.

Use the confidence labels defined in `MECHANICS.md`. Never promote a discussion claim to a confirmed mechanic without evidence.

## Engine and Artifact Provenance

Every locked engine or third-party agent must record:

- original source;
- retrieval date;
- version or commit;
- SHA-256 hashes;
- local modifications;
- assumed server compatibility;
- relevant known behavioral differences.

Never overwrite the original artifact when adapting it.

## Evaluation Rules

Every serious competitive evaluation must:

- use a fixed recorded seed list;
- play both seat assignments;
- use immutable agent identities;
- record exact engine identity;
- compare against a frozen versioned opponent pool;
- save machine-readable results;
- separate plumbing baselines from competitive baselines.

Pass, random, and other trivial agents are smoke tests only.

## Compute Safety

Before an expensive run, update `CURRENT_STATE.md` with:

- objective and hypothesis;
- exact command or configuration;
- branch and commit;
- engine version and hashes;
- agents and opponent-pool versions;
- seeds and seat policy;
- expected output paths;
- stop conditions;
- recovery or resume plan.

After the run, record results and failures before starting the next experiment.

Do not launch a run from a remembered configuration when the durable files disagree or are incomplete.

## Kaggle Workflow

- Prefer one clear setup cell.
- Use Kaggle secrets for Git authentication.
- Avoid `git fetch` in Kaggle notebooks when a fresh clone is safer.
- Keep runs foregrounded unless the user explicitly requests background execution.
- Use CPU for plumbing and evaluation unless accelerators provide a measured benefit.
- Track `/kaggle/working` storage use and avoid unnecessary duplication.
- Do not overwrite notebook dataset attachments or accelerator metadata when exporting or editing notebooks.
- Log phase, elapsed time, throughput, worker counts, skips, and output paths for long jobs.

## Documentation Update Contract

At the end of substantial work:

- `CURRENT_STATE.md`: replace stale active facts; keep concise.
- `HISTORY.md`: append what happened, including failures and exact artifacts.
- `PLANS.md`: revise priorities and future work.
- `DECISIONS.md`: add or supersede durable choices.
- `MECHANICS.md`: update evidence, confidence, engine identity, and regressions.

A task is not complete if its durable state has not been recorded.

## Coding Expectations Once Implementation Begins

- Python 3.12 unless the live Kaggle environment requires otherwise.
- Deterministic behavior by default.
- Type hints for public interfaces.
- Clear error messages and explicit validation.
- JSON/JSONL outputs with schemas or version fields.
- Small targeted behavioral tests for engine contracts.
- Fast smoke commands that can run before larger suites.
- Avoid hidden global state and unrecorded random seeds.

## Submission Safety

Before packaging a Kaggle submission:

- validate `agent(obs)` with representative observations;
- confirm no network dependency;
- confirm paths and imports work in a clean environment;
- verify runtime and memory limits;
- include deterministic fallback behavior;
- record the exact archive hash and source commit;
- do not change notebook datasets or accelerator settings unintentionally.

## Communication

Be direct about uncertainty, failed checks, or missing access. Partial verified progress is better than a confident guess. Do not promise background work or future delivery.
