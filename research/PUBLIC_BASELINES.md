# Public Baseline Catalog

Last updated: 2026-08-06

This is an initial catalog from pre-repository research. Exact files, Kaggle identifiers, hashes, and current engine compatibility still need to be recorded after download.

## Catalog Status

- Raw public notebook artifacts archived: no
- Immutable hashes recorded: no
- Local reproduction completed: no
- Frozen competitive pool established: no
- Live engine compatibility checked: no

Do not use names below as reproducible evaluation identities until their downloaded artifacts are hashed.

## V18 Closed-Loop / Top-10 Future Holdout

- Public title observed: `40/53 Top-10 Future Holdout | v18 Closed Loop`
- Author observed: Kaito Fukami
- Artifact status: public notebook code was found mirrored during research
- Reported license in mirror: Apache-2.0
- Exact Kaggle version/hash: pending

### Apparent Strategy

- deterministic or nearly deterministic full-route execution;
- three unlocked quadrants;
- approximately thirteen field workers at peak;
- fourteen livestock sites;
- approximately eight cows and six sheep;
- wheat and melon opening with limited carrots;
- substantial recurring strawberry production;
- fertilizer collection and selective use;
- wheat purchase/feed/sale cash-flow loop;
- replay-derived worker routes;
- public-state expert or schedule selection.

### Importance

This appears to be one of the central public strategy families currently copied or adapted across the leaderboard. It should be one of the first frozen competitive references.

### Required Follow-Up

- download exact Kaggle notebook output and source;
- record notebook version and SHA-256;
- identify embedded expert schedules;
- separate route execution from selector logic;
- test both seats against exact copies and variants;
- map engine assumptions and silent no-op dependencies.

## C20 Exact Replication Control

- Public title observed: `Kaggriculture C20 Exact Replication Control`
- Author observed: Kun Zhang / Beiciccc
- Exact Kaggle version/hash: pending

### Apparent Strategy

Likely a close replication/control implementation of the same broad V18-style industrial mixed-farm route.

### Importance

Useful for distinguishing whether performance comes from the underlying route, public-state selector, implementation details, or accidental engine-specific behavior.

### Required Follow-Up

- download and hash exact artifact;
- compare action stream and configuration against V18;
- identify meaningful divergences rather than naming differences;
- measure direct head-to-head and seat effects.

## Hamburger V27

- Public nickname/title observed: `Hamburger V27`
- Exact author and Kaggle identifier: pending
- Exact hash: pending

### Preliminary Finding

A public audit reportedly found this newer version weaker than the V18/C20 family in at least one tested setting. That finding is not yet independently reproduced.

Confidence: `DISCUSSION_CLAIM` / third-party audit evidence.

### Required Follow-Up

- obtain exact artifact;
- verify whether weakness is general or opponent/engine-specific;
- inspect whether it trades raw bank for robustness or matchup performance;
- do not assume a larger version number means stronger play.

## Replay-Lab V5.1 Example

- Source project observed: `titakatrwa/kaggriculture-replay-lab`
- Exact agent artifact/hash: pending

### Reported Example

One reported episode produced approximately:

- final bank 129,401 versus 93,477;
- 4.9% `PASS` actions;
- 112 unbanked terminal value;
- 1,108 wheat;
- 682 milk;
- 636 strawberries;
- 445 fertilizer;
- 373 wool;
- 224 melons.

This is useful as a scale and telemetry example, not yet as a verified competitive benchmark.

### Required Follow-Up

- inspect replay and metric definitions;
- determine engine version;
- obtain agent source if available;
- reproduce metrics independently.

## Other Public Repositories Found

### `Seyamalam/Kaggriculture`

Apparent strengths:

- current competition documentation;
- tests and reports;
- public-agent mirrors;
- Kaggle submission records;
- explicit competition and engine auditing.

Use as a research/reference source, not as authority over current engine source.

### `basavarajbaboji/kaggriculture`

Apparent contents:

- simulation agent;
- tooling;
- market/parser work.

Needs exact artifact and compatibility review.

### `deepeshumrao/kaggriculture-agent`

Apparent history:

- began with guessed mechanics;
- later decoded parts of the real contract;
- may contain useful development notes but is less authoritative than current official source.

### `titakatrwa/kaggriculture-replay-lab`

Apparent use:

- replay collection;
- metrics;
- Streamlit analysis.

Potentially useful for telemetry ideas and replay forensics.

### `rooklift/krobus`

Third-party Electron replay viewer. Potentially useful for human inspection, but not central to automated evaluation.

## Initial Strategy Families to Track

1. V18/C20-style mixed industrial route.
2. Strawberry-heavy recurring-crop route.
3. Cow-heavy livestock route.
4. Sheep-heavy livestock route.
5. Wheat/feed/cash-flow loop variants.
6. Market-collision-resistant diversified route.
7. Low-capital or delayed-expansion route.
8. Closed-loop repair variants of copied schedules.
9. Public-state expert-selector variants.
10. Deliberate market-interference strategies.

## Baseline Admission Criteria

An agent can enter the frozen evaluation pool only after recording:

- stable identifier;
- source and download date;
- SHA-256 hash;
- engine assumptions;
- packaging method;
- deterministic/reproducibility status;
- seat-swapped smoke result;
- known failures;
- whether it is original, copied, or locally modified.

## Performance Scale — Provisional

Very rough pre-repository observations:

- 3,000–10,000 bank: primitive agent;
- 20,000–50,000: decent basic heuristic;
- 100,000+: serious public-meta territory;
- sampled strong public games reportedly averaged around 120,000 final bank, with substantial episode variance.

These are not competitive thresholds. Win rate against the frozen pool is the primary metric; final bank is supporting telemetry.
