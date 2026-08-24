# BC V1 Economic Context — Schema-v3 Provenance Audit (issue #6, Stage 0)

Status: audit only. No code, test, executor, or schema changes in this stage.
Verified against current `main` source at commit `d8bc8ac` (files listed per
section). Current source is authoritative over any research summary.

Authoritative sources cross-checked for this note:

- `replay_daily/constants.py` (pinned 1.32.7 tables, `SCHEMA_VERSION = 3`)
- `replay_daily/extractor.py` (record construction, event attribution,
  `previous_execution_of_day`)
- `replay_daily/storage.py` (Arrow schema: `start`/`events`/`targets`/`end`)
- `bc_manager/adapter.py` (`_input_arrays_from_starts`, `load_selected_table`,
  `build_targets`, `_eval_metadata`)
- `bc_manager/live.py` (`encode_live_inputs`, `validate_previous_execution`)
- `bc_manager/model.py` (`GlobalEncoders`, money encoding, input key contract)
- `bc_manager/training.py` (`CHECKPOINT_FORMAT = "bc_manager_checkpoint_v1"`,
  `save_checkpoint`/`load_model_from_checkpoint`)
- `bc_manager/constants.py` (vocab orders, `MAX_HANDS`, `TOTAL_DAYS`)
- tests: `tests/test_bc_manager_live.py`, `tests/test_replay_daily_storage.py`

---

## 1. Scope and hard rule

- **Schema-v3 only.** Every feature below is derived exclusively from fields
  that already exist in canonical schema-v3 rows (`SCHEMA_VERSION = 3`,
  `replay_daily/constants.py`). No schema-v4 migration in this packet.
- **Submitted intents are not realized fills.** `events.buys`
  (seeds/products/animals), `events.sells`, and
  `events.market_events_ordered` record the *submitted market orders* read
  from `action["market"]` (`extractor._events_from_action`). Nothing in the
  schema proves any individual buy/sell order cleared, partially cleared, or
  was rejected for affordability/stock. Therefore **gross revenue, gross
  spend, and any sell/market fill quantity must never be inferred from these
  fields** and are omitted from V1.
- The only economic quantities the schema proves as *realized* are:
  - `start.self.money` / `end.self.money` (observed state snapshots), and
  - `start.previous_execution.{workers_hired, hire_cost}` (derived from the
    observed `hires_today` counter, not from HIRE intents;
    `extractor.previous_execution_of_day`).
- If a candidate feature needs data that schema v3 does not carry exactly, it
  is omitted here and listed in §7 as future regeneration work. No
  approximation is substituted.

## 2. Provenance / timing table (current row fields)

Row identity: one Parquet row per `(metadata.episode_id, metadata.seat, day)`;
`day` is also a top-level column. Day boundaries come exclusively from
explicit observation `day`/`hour` fields: day start = first observation with
`day == d, hour == 0`; end = first observation with `day == d+1, hour == 0`
(`end.boundary == "next_day_start"`), else the final terminal observation
(`end.boundary == "terminal"`) (`extractor.extract_replay`).

| Field | Source | Timing | Notes |
|---|---|---|---|
| row key | `metadata.episode_id`, `metadata.seat`, top-level `day` | — | grouping key for all history derivation |
| `start.self.money` | `farms[seat]["money"]` of the day-start observation | exact snapshot at first obs with `day==d, hour==0` | float; engine decodes money exactly (see `tests/test_oracle_mechanics.py::test_regression_money_decode_f32_noise`) |
| `start.self.hires_today` | `farms[seat]["hires_today"]` | day-start snapshot | resets to 0 at the morning refresh |
| `start.self.unlocked_quadrants` | `farms[seat]["unlocked_quadrants"]` | day-start snapshot | NW always unlocked; unlock order pinned `LAND_ORDER = ["NE","SW","SE"]` |
| `start.market.prices` / `start.market.inventory` | shared `obs["market"]` | day-start snapshot | live market product prices remain authoritative for products; missing product keys encode as 0 downstream |
| `start.previous_execution.workers_hired` / `.hire_cost` | observed `hires_today` increases within day d−1; cost via exact Fibonacci `total_hire_cost` with episode config `farmHandCostMult` | realized labor of **day d−1** | zeros when day d−1 has no start observation; never derived from submitted HIRE intents |
| `events.hires.realized` | same derivation applied to day d itself | realized hires **of day d** | "used as next day's feedback mirror"; equals next row's `start.previous_execution` when the next row exists |
| `events.buys.*`, `events.sells`, `events.market_events_ordered` | submitted `action["market"]` orders | attributed to pre-action observation inside day d | **intent status only**; no realization proof |
| `events.land_purchases[].quadrant` | submitted BUY_LAND, quadrant filled only when the adjacent observation transition validates an ordered prefix addition | intent + optional observed validation | `quadrant` stays null when unvalidated |
| `end.self.money` etc. | end-boundary observation | first obs of day d+1, else terminal | full self/shared snapshot; usable only as label-side or offline analysis, never as a decision-time input |

## 3. Feature decision table

Legend: INCLUDE for E / ALREADY PRESENT / OMIT. All constants are pinned in
`replay_daily/constants.py`: seed costs `CROPS[c]["seed"]` → WHEAT 10, CARROT
20, TOMATO 50, STRAWBERRY 100, MELON 80; animal costs `ANIMALS[a]["cost"]` →
GOOSE 300, COW 400, SHEEP 500; land `LAND_PRICES = [1000, 2000, 4000]` along
`LAND_ORDER = ["NE", "SW", "SE"]`.

### INCLUDE for E

**E1. Improved current-cash channels** (raw source preserved:
`start.self.money`).

Let `m = start.self.money` (float). Two deterministic channels:

- `cash_log = clamp(sign(m) * log1p(|m| * 1e-4), -8.0, +8.0)`
- `cash_lin  = clamp(m * 1e-4, -8.0, +8.0)`

Rationale: V0's single money channel
(`model.GlobalEncoders.forward`: `clamp(_sign_log1p(scalars[:,0:1]*1e-4),
-8, 8)`) compresses cash aggressively; the added clipped-linear channel
preserves linear resolution near the failure regime (tens–thousands of
dollars: `m*1e-4 ∈ [0.001, 0.75]` there, well inside the ±8 clip). The log
channel keeps large-bank episodes bounded. Both transforms are pure functions
of the raw current-cash source; nothing else feeds them.

**E2. Cash-affordability transforms** — one channel per pinned cost, using
current unlocked state:

```
afford(c) = clamp(log1p(max(m, 0) / c), 0.0, 8.0)
```

Channels (9 total):

| Channel | c | Source of c | State dependence |
|---|---|---|---|
| afford_seed_WHEAT | 10 | `CROPS["WHEAT"]["seed"]` | none (constant) |
| afford_seed_CARROT | 20 | `CROPS["CARROT"]["seed"]` | none |
| afford_seed_TOMATO | 50 | `CROPS["TOMATO"]["seed"]` | none |
| afford_seed_STRAWBERRY | 100 | `CROPS["STRAWBERRY"]["seed"]` | none |
| afford_seed_MELON | 80 | `CROPS["MELON"]["seed"]` | none |
| afford_animal_GOOSE | 300 | `ANIMALS["GOOSE"]["cost"]` | none |
| afford_animal_COW | 400 | `ANIMALS["COW"]["cost"]` | none |
| afford_animal_SHEEP | 500 | `ANIMALS["SHEEP"]["cost"]` | none |
| afford_land_next | next locked quadrant price | `LAND_PRICES[len(unlocked)-1]` where `unlocked = start.self.unlocked_quadrants` | yes |

The constant-cost channels individually carry no state information (the cost
is fixed); the *ratio to current cash* is the intended state-varying signal,
per issue #6's design principle.

- **All-unlocked behavior:** when `len(unlocked) == 4` there is no next land
  cost. Then `afford_land_next := 8.0` (the saturated transform value) **and**
  the companion validity bit `land_next_valid := 0` (see vector layout §4).
  When a next locked quadrant exists, `land_next_valid := 1`.
- **Zero/negative cash stability:** `max(m, 0)` makes every channel finite
  and monotone; `afford(0) = 0` exactly. Negative money (not observed in
  practice) yields the same 0 as zero cash — conservative and stable.
- Land ordering: the "next" quadrant is `LAND_ORDER[len(unlocked)-1]` with
  price `LAND_PRICES[len(unlocked)-1]`; this matches the extractor's
  validated prefix-addition semantics (`_observed_land_additions`).

**E3. Previous-day net cash change with validity bit.**

- Derivation condition (batch): a row for the *same* `episode_id` **and**
  same `seat` exists whose top-level `day == d - 1` exactly. Then
  `prev_net_cash = start[d].money - start[d-1].money` (both day-start
  snapshots), `prev_valid = 1`.
- Day 0, any gap (`day-1` row absent), or episode/reset boundary:
  `prev_net_cash := 0.0`, `prev_valid := 0`. Never join across episodes,
  seats, or non-adjacent days.
- Live tracking contract (equivalent, action-free): the live caller tracks
  the previous daily **start** money it actually observed for the current
  episode/seat. At encode time for day d ≥ 1 with a stored prior-day start
  money `m_{d-1}` observed at that day's `hour == 0` boundary and
  `last_tracked_day == d - 1`: `prev_net_cash = m_d - m_{d-1}`,
  `prev_valid = 1`. Otherwise (day 0, new episode, missed/skipped day):
  zeros + invalid. **Never inferred from actions or market events.**
- Encoding: `prev_net_cash_enc = clamp(sign(x)*log1p(|x|*1e-4), -8, +8)`
  where `x = prev_net_cash`; stored raw alongside for diagnostics.

### ALREADY PRESENT / DO NOT DUPLICATE

- **Previous realized `workers_hired` and `hire_cost`**: already carried by
  `start.previous_execution` and encoded in V0 (`scalars[:, 2:4]`, fed to the
  labor token, `model.GlobalEncoders`). E adds nothing here; compatibility
  preserved.

### OMIT (and why)

| Candidate | Verdict | Reason |
|---|---|---|
| Gross revenue (previous day) | OMIT | Sell proceeds are not derivable: `events.sells` are submitted intents; no fill/price-at-execution proof exists in schema v3. |
| Gross spend (previous day) | OMIT | Same: `events.buys.*` are submitted intents; affordability rejections leave no ledger trace. |
| Sell/market fills | OMIT | Not recorded anywhere in schema v3. |
| Min intraday cash | OMIT | Only day-start/end snapshots exist; intra-day observations are not persisted per row. |
| Harvest revenue estimates | OMIT | Would require price-at-sale-time × realized sale quantity; neither is proven. |
| Requested-vs-achieved / unfinished-task / missed-maintenance feedback | OMIT | Per issue #6: elite demonstrations contain no comparable failures; reserved for later closed-loop RL. Also not derivable from expert rows alone. |
| Marginal next-hire affordability | OMIT | Next-hire cost = `farmHandCostMult * fib(hires_today)` (`replay_daily.constants.hire_cost`). `hires_today` IS persisted, but `farmHandCostMult` is consumed upstream in `extractor.extract_replay` (`config.get("farmHandCostMult")`) and **is not persisted in any schema-v3 field**, so universal exact derivation is impossible. Inferring the multiplier from past realized hire cost would be an approximation and is **forbidden** under the hard rule. |
| Previous-day starting cash | OMIT as algebraically redundant | Given the current row's `start[d].money` (already an input) and the included `prev_net_cash`, the previous starting cash is exactly `start[d].money - prev_net_cash`. A separate channel would duplicate information with zero marginal content. |

## 4. Proposed economic-context vector (ordering / dimension / formulas)

Proposed name: `econ` — `float32 [N, 14]`, appended to the adapter's
predictive `inputs` and consumed only by E/JE variants (§8). Exact channel
order:

| idx | name | formula (deterministic) |
|---|---|---|
| 0 | `cash_log` | `clamp(sign(m)*log1p(\|m\|*1e-4), -8, 8)` |
| 1 | `cash_lin` | `clamp(m*1e-4, -8, 8)` |
| 2–6 | `afford_seed_{WHEAT,CARROT,TOMATO,STRAWBERRY,MELON}` | `clamp(log1p(max(m,0)/c), 0, 8)`, c ∈ {10,20,50,100,80} |
| 7–9 | `afford_animal_{GOOSE,COW,SHEEP}` | `clamp(log1p(max(m,0)/c), 0, 8)`, c ∈ {300,400,500} |
| 10 | `afford_land_next` | `clamp(log1p(max(m,0)/next_land_price), 0, 8)` if `len(unlocked)<4` else `8.0` |
| 11 | `land_next_valid` | `1.0` if `len(unlocked)<4` else `0.0` |
| 12 | `prev_net_cash_log` | `clamp(sign(Δ)*log1p(\|Δ\|*1e-4), -8, 8)` with Δ per §E3; `0.0` when invalid |
| 13 | `prev_net_cash_valid` | `1.0` iff exact same-`(episode_id, seat)` day−1 row used; else `0.0` |

where `m = start.self.money`, `unlocked = start.self.unlocked_quadrants`,
`next_land_price = LAND_PRICES[len(unlocked)-1]`.

Design notes:

- Simple bounded transforms only; every channel is finite for all inputs;
  no NaN/Inf can be emitted.
- Dimension 14 is minimal for the included feature set (§3): 2 cash + 9
  affordability + 1 land validity + 1 prev-delta + 1 prev validity.
- Implementation may adjust names/placement after source verification but
  must keep this minimality and determinism; any change re-documents here.

## 5. Batch derivation algorithm

1. After `load_selected_table` concatenates all selected Parquet parts
   (`pa.concat_tables`) and after the train/val date split, group rows by
   `(metadata.episode_id, metadata.seat)`.
2. Within each group build a dict keyed by the exact integer `day` column.
   Duplicate `(episode_id, seat, day)` keys are an error (fail loudly);
   lookups use the key dict — **no positional joining, no assumed sort
   order, no assumed contiguity**.
3. For each row of day `d`: `prev` lookup is `group[d - 1]`; present →
   compute Δ and set valid bit; absent (day 0, gap, reset) → `0.0`/invalid.
   All other channels depend only on the row itself.
4. **File/partition boundaries:** the loader may read multiple files and the
   selected split may interleave rows arbitrarily; therefore grouping MUST be
   done on metadata keys over the fully concatenated split, never per-file
   and never positionally. Corpus organization does *not* guarantee that one
   episode's rows are contiguous or in one file, so correctness must not
   assume it. Conversely, an episode's rows share one `partition_date` and
   one `min_score`, so the date/score filters can never split a group
   internally; still, the key-based gap default handles any filtered-out
   neighbor defensively without special-casing.
5. Cross-episode leakage is structurally impossible: the join key includes
   `episode_id` and `seat`; tests must cover two interleaved episodes plus
   both seats asserting no cross-group reads.

## 6. Live derivation contract and reset semantics

- Live path (`bc_manager.live.encode_live_inputs`) gains the same `econ`
  computation for one observation, driven by:
  - current observation fields (money, unlocked quadrants) — identical
    formulas to batch;
  - a caller-maintained tracker of the previous daily start money/day for
    the current episode/seat (§E3 contract); reset on new episode, on seat
    change, and whenever the tracked day is not exactly `d - 1`.
- Day 0 / reset ⇒ channels 12–13 = `0.0 / 0.0`; channels 0–11 need no
  history and are always exact.
- Train/live parity is enforced by sharing one implementation for the econ
  transform, mirroring the existing `_input_arrays_from_starts` pattern; the
  parity test requirement of issue #6 applies.
- **Executor scope unchanged:** this changes only the BC model input
  vector. `ExecutorAgent` task selection, action emission, opening-book
  behavior, and all executor configs stay byte-identical.

## 7. Schema / checkpoint implications

- **No schema version bump and no data regeneration are required**: every
  included source (`start.self.money`, `start.self.unlocked_quadrants`,
  pinned cost tables, exact day−1 row existence) already exists in schema v3.
- Features that WOULD require new extraction fields (i.e., schema-v4 +
  regeneration) if ever wanted, and are therefore explicitly excluded here:
  realized buy/sell fills and executed prices (needs engine-side fill
  ledger), intraday minimum cash (needs denser snapshots), and
  `farmHandCostMult` persisted per row (would make marginal hire
  affordability exact).
- Checkpoints: V0 checkpoints keep format `bc_manager_checkpoint_v1`
  (`training.CHECKPOINT_FORMAT`) and load unchanged as V0. E/JE introduce a
  variant selector in the serialized model config; old payloads lacking the
  selector default to V0 so existing artifacts keep loading. JAX V1 mirror
  (`bc_manager_jax/`) is out of scope for this experiment (issue #6 JAX
  scope) and is not altered.

## 8. V0 compatibility constraints

- V0 uses the exact old inputs, architecture, and state dict; its forward
  path, parameter shapes, and checkpoint bytes are untouched.
- The economic vector is consumed **only** by variants E and JE; J shares
  V0's inputs exactly.
- An existing `bc_manager_checkpoint_v1` payload loads as V0 (selector
  defaults to V0 when absent).
- The PyTorch JAX V1 mirror is not modified in this experiment.

## 9. Diagnostics provenance (plan-coherence metrics)

All diagnostics are computed from current state + predicted targets + exact
pinned constants. **Diagnostics only: they never clip predictions, never
enter the loss, and never gate training.**

Definitions (per prediction row):

- Current counts (same definitions as the extractor's target builders):
  - current crop counts: board PLANT tiles per crop
    (`extractor._crop_composition`);
  - current animal counts: board tiles carrying an `animal` key
    (`extractor._animal_counts`);
  - current unlocked count: `len(start.self.unlocked_quadrants)`.
- **Lower-bound implied acquisition cost** (positive deltas only):
  - animals: `Σ_species max(pred_end_count − current_count, 0) × ANIMALS[cost]`;
  - crops: `Σ_crop max(pred_end_count − current_count, 0) × CROPS[seed]`;
  - land: see below.
  This deliberately excludes routing, maintenance, feed, fertilizer, and all
  future costs; it is labeled a lower bound and reported as such.
- **Land expansion cost — ambiguity resolution:** a predicted land count may
  imply multiple expansions. Deterministic conservative formula using the
  ordered remaining quadrant costs: with current count `u` and predicted
  count `t` (each in 1..4),
  `land_cost = Σ_{i=u-1}^{t-2} LAND_PRICES[i]` when `t > u` (buying
  quadrants `LAND_ORDER[u-1 .. t-2]` in canonical order), else `0`.
- **Ratio to current cash with stable zero handling:**
  `ratio = total_lower_bound_cost / m` when `m > 0`; when `m <= 0`, ratio is
  defined as `+inf` (and such rows count toward every `>` threshold). This is
  deterministic and conservative; no epsilon smoothing is applied.
- Reported aggregates: fraction of validation predictions with ratio > 1×
  and > 2× current cash; predicted land-regression rate
  (`t < u`, which the executor currently has to clip).
- **Expert targets too:** the identical diagnostics are computed for
  ground-truth expert targets (`targets.*`) to separate inherently aggressive
  demonstrations from model-generated incoherence.

## 10. Open limitations

- **Corpus gap prevalence is not locally audited**: the real processed
  corpus is not present in this working tree, so the empirical frequency of
  missing day−1 rows (invalid `prev_net_cash`) cannot be measured here; the
  derivation handles gaps correctly but their prevalence is unknown until
  first real training run.
- **Real checkpoint/corpus work is Kaggle-only**: no local training or
  closed-loop numbers are produced or claimed by this audit.
- **Submitted-intent realization is absent** from schema v3; any future
  feature requiring realized fills requires engine-side extraction changes
  (schema-v4 territory, explicitly out of scope here).
