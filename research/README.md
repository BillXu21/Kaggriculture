# Research Workspace

This directory stores durable research notes that are too detailed for `CURRENT_STATE.md` and too topic-specific for `HISTORY.md`.

## Intended Files

- `PUBLIC_BASELINES.md`: catalog of important public notebooks and agents.
- `ENGINE_CHANGES.md`: dated engine and rules changes with behavioral impact.
- `COMPETITION_SNAPSHOT.md`: rules, deadlines, runtime constraints, and submission contract.
- `REPLAY_FINDINGS.md`: recurring layouts, strategies, failures, and market behaviors observed in replays.
- `OPEN_QUESTIONS.md`: mechanics and strategy questions requiring source inspection or experiments.

Create these files only when there is concrete material to record. Avoid empty bureaucracy.

## Evidence Template

Each finding should include:

```markdown
## Finding title

- Date checked:
- Source:
- Source version/commit:
- Artifact hash:
- Confidence: CONFIRMED_SOURCE | CONFIRMED_EXPERIMENT | DISCUSSION_CLAIM | OUTDATED | UNKNOWN
- Finding:
- Reproduction method:
- Strategic impact:
- Follow-up:
```

## Public Artifact Template

```markdown
## Agent or notebook name

- Author:
- Kaggle URL or identifier:
- Downloaded:
- Original filename:
- SHA-256:
- License/provenance note:
- Engine assumptions:
- Strategy summary:
- Known score or match evidence:
- Local modifications:
- Frozen evaluation ID:
```

## Research Rules

- Preserve raw downloaded artifacts outside Git when they are large.
- Record hashes before modifying anything.
- Keep original and adapted versions separate.
- Distinguish observed behavior from interpretation.
- Use exact dates and engine identities.
- Move only currently actionable conclusions into `CURRENT_STATE.md`.
