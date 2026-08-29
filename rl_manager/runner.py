"""Lockstep N-env self-play runner (issue #9 A3, architecture reqs. 2-9).

Single-process correctness mode. The runner owns N independent scalar backend
instances stepped in lockstep; every day-boundary manager request across all
envs/seats is grouped by policy identity and answered with ONE batched
policy call per group/day (never one call per environment). Host loops over
envs/backends/executors are plain Python; there is no Python loop over
examples inside JAX.

Per episode/seat the runner owns:
- one independent opening agent (`standard_mixed`, literal d0-d3 playback,
  clean d4h0 handoff), executor agent, queued-plan provider, and economic
  daily-start state — never shared across seats or games;
- exact `bc_manager.live.encode_live_inputs` E semantics via the stateless
  `economic_prev_start` path fed from the runner-owned previous daily-start
  `(day, cash)` observation (reset per episode/seat, never inferred from
  submitted orders);
- prior-day realized labor (`workers_hired` observed via `hires_today`,
  never from HIRE intents).

Rewards are terminal-only: +1 win / 0 tie / -1 loss at the final manager
transition; raw bank margins stay diagnostics.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bc_manager.constants import TOTAL_DAYS
from bc_manager.live import encode_live_inputs
from executor_v0.plan import DailyPlan
from opening_book.agent import make_opening_agent
from oracle.backend import EngineBackend, make_backend
from oracle.batched_backend import BatchedEngineBackend, make_batched_backend
from oracle.canonical import canonical_state_fast
from replay_daily.constants import total_hire_cost

from rl_manager.decode import plans_from_action_tensors
from rl_manager.debug_trace import TraceRecorder
from rl_manager.executor_factory import make_default_executor_factory
from rl_manager.provenance import (
    backend_provenance,
    canonical_json,
    opening_provenance,
    sha256_hex,
)
from rl_manager.provider import QueuedPlanProvider
from rl_manager.trajectory import TrajectoryBuffer, Transition, \
    TransitionMetadata
from rl_manager.types import (
    CANDIDATE_VS_FROZEN,
    E_VS_E,
    E_VS_PASS,
    FROZEN_VS_CANDIDATE,
    BatchedPlanPolicy,
    seat_policies,
)

MANAGER_START_DAY = 4
TOTAL_MANAGER_DAYS = TOTAL_DAYS - MANAGER_START_DAY  # 26 decisions/seat
GAME_TURNS = 719  # post-reset primitive turns in one 720-step game
INFERENCE_BATCH_SCOPES = ("policy_day", "policy")
InferenceBatchScope = Literal["policy_day", "policy"]

# Artifact provenance sidecar schema (issue #9 A1 correction): the
# `run_metadata` block written by `build_artifact_metadata` carries its own
# version so consumers can pin against exactly these mandatory fields.
ARTIFACT_METADATA_SCHEMA_VERSION = 1


class _ReadOnlyDict(dict):
    """Dict-shaped observation view that rejects accidental agent mutation."""

    def __readonly(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("self-play agent observations are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __readonly


class _ReadOnlyList(list):
    """List-shaped observation view preserving existing list type checks."""

    def __readonly(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("self-play agent observations are read-only")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = append = extend = insert = (
        remove
    ) = reverse = sort = __readonly


def _readonly_observation(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _ReadOnlyDict({
            key: _readonly_observation(item) for key, item in value.items()})
    if isinstance(value, list):
        return _ReadOnlyList(_readonly_observation(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_readonly_observation(item) for item in value)
    return value


@dataclass(frozen=True)
class RunnerConfig:
    """Knobs are explicit and small-host friendly; nothing is hard-coded."""

    backend_name: str = "fast"
    backend_configuration: Mapping[str, Any] = field(
        default_factory=lambda: {"seed": 0, "numThreads": 1})
    opening: str = "standard_mixed"
    manager_start_day: int = MANAGER_START_DAY
    max_turns: int = GAME_TURNS
    num_envs: int = 1  # lockstep envs per chunk (single-process mode)
    low_telemetry: bool = False  # skip executor turn snapshots for training
    read_only_agent_observations: bool = False  # avoid per-call deep copies
    batch_backend: bool = False  # one native fast engine for each lockstep chunk
    inference_batch_scope: InferenceBatchScope = "policy_day"
    fixed_inference_batch_size: int | None = None
    inference_batch_wait_seconds: float = 0.02
    record_rollout: bool = False  # capture full primitive trace for parity
    record_debug_trace: bool = False  # capture canonical viewer trace opt-in
    debug_trace_seat: int | None = None  # requested private-seat/view selector
    debug_trace_view: str = "joint"

    def __post_init__(self) -> None:
        if self.inference_batch_scope not in INFERENCE_BATCH_SCOPES:
            raise ValueError(
                "inference_batch_scope must be one of "
                f"{INFERENCE_BATCH_SCOPES}, got {self.inference_batch_scope!r}")
        size = self.fixed_inference_batch_size
        if (size is not None
                and (isinstance(size, bool) or not isinstance(size, int)
                     or size < 1)):
            raise ValueError("fixed_inference_batch_size must be a positive int")
        if (not math.isfinite(self.inference_batch_wait_seconds)
                or self.inference_batch_wait_seconds < 0):
            raise ValueError(
                "inference_batch_wait_seconds must be finite and >= 0")


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode: explicit seed, ownership index, and seat policies."""

    episode_index: int
    seed: int
    composition: str
    policies: tuple[BatchedPlanPolicy, BatchedPlanPolicy]
    trainable_seats: tuple[int, ...]
    controlled_seat: int | None = None


def build_episode_spec(
    episode_index: int,
    seed: int,
    composition: str,
    candidate: BatchedPlanPolicy,
    frozen: BatchedPlanPolicy,
    *,
    controlled_seat: int = 0,
) -> EpisodeSpec:
    """Resolve a composition into seat policies + trainable ownership.

    Seat resolution goes through the single authoritative
    `rl_manager.types.seat_policies` resolver so the E-vs-E identical-
    identity guard cannot be bypassed by this entry point.
    """
    policies = seat_policies(
        composition, candidate, frozen, controlled_seat=controlled_seat)
    trainable = {
        E_VS_E: (),
        E_VS_PASS: (),
        CANDIDATE_VS_FROZEN: (0,),
        FROZEN_VS_CANDIDATE: (1,),
    }[composition]
    return EpisodeSpec(
        episode_index=episode_index, seed=seed, composition=composition,
        policies=policies, trainable_seats=trainable,
        controlled_seat=(controlled_seat if composition == E_VS_PASS else None))


@dataclass
class RolloutRecord:
    """Full primitive-level record of one episode (parity/diagnostics only).

    Kept out of the training buffer; used by the official/fast comparison
    seam and rich sidecars.
    """

    seed: int
    backend_name: str
    composition: str
    joint_actions: list[tuple[int, int, int, dict, dict]] = field(
        default_factory=list)
    manager_input_digests: dict[tuple[int, int], str] = field(
        default_factory=dict)
    plans: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    opening_handoff: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    composition: str
    final_banks: list[float]
    margin: float
    winner_seat: int  # -1 on tie
    rewards: list[float]
    statuses: list[str]
    transitions: int
    terminated: bool
    opening_diagnostics: list[dict[str, Any]]
    trace_digest: str
    rollout: RolloutRecord | None
    timing_seconds: dict[str, float]
    # Per-seat JSON-safe policy/opponent snapshot identities (issue #9 A1):
    # [{"seat", "policy": {name, version, fingerprint, identity_id},
    #   "opponent": {...}, "trainable"} x 2] — recorded at finalize time so
    # artifact metadata never depends on caller-side policy bookkeeping.
    policy_identities: tuple[dict[str, Any], ...]
    debug_trace: dict[str, Any] | None = None
    # Keep runtime failures separate from informational opening handoff data.
    executor_diagnostics: list[dict[str, Any]] = field(default_factory=list)


def _executor_factory_provenance(factory: Any) -> dict[str, Any]:
    """JSON-safe executor factory identity: name/version/identifier/hash."""
    name = str(getattr(factory, "name", "unknown"))
    version = str(getattr(factory, "version", "unknown"))
    identifier = f"{name}@{version}"
    return {
        "name": name,
        "version": version,
        "identifier": identifier,
        "version_sha256": sha256_hex(identifier),
    }


def _debug_trace_metadata(
    spec: EpisodeSpec,
    config: RunnerConfig,
    provenance: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic metadata for one opt-in full-state trace."""
    # Refresh after backend construction so lazy fast-engine loading can be
    # reflected in the trace's engine identity.
    backend = backend_provenance(config.backend_name, configuration)
    engine = backend.get("engine_module") or config.backend_name
    metadata: dict[str, Any] = {
        "seed": int(spec.seed),
        "view": str(config.debug_trace_view),
        "backend": str(config.backend_name),
        "engine": str(engine),
        "provenance": {
            "schema_producer": "rl_manager.runner",
            "trace_schema_version": 1,
            "episode_index": int(spec.episode_index),
            "composition": str(spec.composition),
            "opening": copy.deepcopy(dict(provenance["opening"])),
            "backend": backend,
            "backend_configuration": copy.deepcopy(dict(configuration)),
            "executor_factory": _executor_factory_provenance(
                provenance["executor_factory"]),
            "action_state_alignment": {
                "initial": "step 0 reset snapshot carries its primitive decision",
                "decision": "canonical_state is observed before joint_actions",
                "terminal": "final observed canonical_state has no action or sidecar",
                "sidecars": "executor_debug contains only same-turn available seats",
            },
        },
    }
    if config.debug_trace_seat is not None:
        metadata["seat"] = int(config.debug_trace_seat)
    return metadata


def build_artifact_metadata(
    provenance: Mapping[str, Any],
    result: EpisodeResult,
) -> dict[str, Any]:
    """Typed automatic artifact metadata for one episode's trajectory save.

    Merges the runner provenance snapshot (opening identity/digest, backend/
    engine provenance, executor factory version/hash/identifier, master seed,
    manager start day) with the exact per-episode outcome (final banks,
    margin, winner, rewards, terminal statuses, episode trace digest, trace
    reference) and per-seat policy/opponent identities into ONE JSON-safe,
    deterministically ordered dict for
    `TrajectoryBuffer.save(run_metadata=...)`. Callers never assemble this by
    hand; the full primitive trace stays in `result.rollout` and is never
    duplicated into the training core.
    """
    return {
        "artifact_schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "episode": {
            "episode_index": int(result.episode_index),
            "seed": int(result.seed),
            "composition": str(result.composition),
            "final_banks": [float(bank) for bank in result.final_banks],
            "margin": float(result.margin),
            "winner_seat": int(result.winner_seat),
            "rewards": [float(reward) for reward in result.rewards],
            "statuses": [str(status) for status in result.statuses],
            "transitions": int(result.transitions),
            "terminated": bool(result.terminated),
            "trace_digest": str(result.trace_digest),
            "rollout_recorded": result.rollout is not None,
            "timing_seconds": {
                key: float(result.timing_seconds[key])
                for key in sorted(result.timing_seconds)},
        },
        "opening": copy.deepcopy(dict(provenance["opening"])),
        "backend": copy.deepcopy(dict(provenance["backend"])),
        "executor_factory": _executor_factory_provenance(
            provenance["executor_factory"]),
        "policies": [copy.deepcopy(record)
                     for record in result.policy_identities],
        "master_seed": provenance["master_seed"],
        "manager_start_day": int(provenance["manager_start_day"]),
    }


class _PassAgent:
    """Primitive-action opponent that passes every available worker slot."""

    debug_trace_turn = None

    def __init__(self, seat: int) -> None:
        self.seat = seat

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        farms = obs.get("farms") if isinstance(obs, Mapping) else None
        hands: Any = []
        if isinstance(farms, list) and self.seat < len(farms):
            farm = farms[self.seat]
            if isinstance(farm, Mapping):
                hands = farm.get("hands") or []
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in hands],
            "market": [],
        }

    def diagnostics_json(self) -> dict[str, Any]:
        return {"composition": "pass", "days": {}}


class _EpisodeState:
    """All per-episode/per-seat mutable state; never shared across games."""

    def __init__(self, spec: EpisodeSpec, config: RunnerConfig,
                 provenance: Mapping[str, Any],
                 backend: EngineBackend | None = None) -> None:
        self.spec = spec
        self.config = config
        configuration = dict(config.backend_configuration)
        configuration["seed"] = int(spec.seed)
        self.configuration = configuration
        self.backend: EngineBackend = backend or make_backend(
            config.backend_name, configuration)
        self.trace_recorder = (
            TraceRecorder(_debug_trace_metadata(
                spec, config, provenance, configuration))
            if config.record_debug_trace else None)
        self.current_canonical_state: dict[str, Any] | None = None
        self.providers = [QueuedPlanProvider(), QueuedPlanProvider()]
        factory = provenance["executor_factory"]
        self.executors = []
        self.openings = []
        for seat in range(2):
            if (spec.composition == E_VS_PASS
                    and seat != spec.controlled_seat):
                pass_agent = _PassAgent(seat)
                self.executors.append(pass_agent)
                self.openings.append(pass_agent)
                continue
            executor = factory.create(
                backend_name=config.backend_name, seat=seat,
                configuration=configuration, provider=self.providers[seat])
            self.executors.append(executor)
            self.openings.append(
                make_opening_agent(config.opening, downstream=executor,
                                   seat=seat))
        # Runner-owned per-seat tracking (exact E history semantics).
        self.last_seen_day = [-1, -1]
        self.daily_start: list[tuple[int, float] | None] = [None, None]
        self.previous_execution: list[dict[str, int]] = [
            {"workers_hired": 0, "hire_cost": 0} for _ in range(2)]
        self.hires_current_day = [0, 0]
        self.planned_days: list[set[int]] = [set(), set()]
        self.transition_index: dict[tuple[int, int], int] = {}
        self.day_hashers: dict[int, Any] = {}
        self.day_digests: dict[int, bytes] = {}
        self.obs: list[dict[str, Any]] = []
        self.agent_obs: list[Mapping[str, Any]] = []
        self.done = False
        self.truncated = False
        self.finalized = False
        self.rollout = RolloutRecord(
            seed=spec.seed, backend_name=config.backend_name,
            composition=spec.composition) if config.record_rollout else None

    def observe_reset(self) -> None:
        self.obs = self._adapt_observations(self.backend.reset())
        for seat in range(2):
            self._note_day_start(seat)

    def _adapt_observations(
        self,
        observations: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Canonicalize farms through the PUBLIC oracle backend seam
        (`EngineBackend.canonical_state`, i.e. `oracle.canonical.
        canonical_state_fast/official`) so fast-engine tile aliases never
        reach the executor or the live encoder. The backend snapshot is read
        immediately after reset/step, so it describes exactly these
        observations; no private oracle helper is imported here."""
        canonical = self.backend.canonical_state()
        if self.trace_recorder is not None:
            self.current_canonical_state = canonical
        canonical_farms = canonical["farms"]
        adapted = []
        for obs in observations:
            view = dict(obs)
            day = int(obs["day"])
            hour = int(obs.get("hour", 0))
            view.setdefault("step", day * 24 + hour)
            view["farms"] = [copy.deepcopy(farm) for farm in canonical_farms]
            adapted.append(view)
        if self.config.read_only_agent_observations:
            self.agent_obs = [_readonly_observation(view) for view in adapted]
        return adapted

    def _executor_debug_for_turn(
            self, *, day: int, hour: int) -> dict[str, Any] | None:
        """Return only same-turn public executor snapshots, if available."""
        if self.trace_recorder is None:
            return None
        available: dict[str, Any] = {}
        for seat, executor in enumerate(self.executors):
            snapshot = getattr(executor, "debug_trace_turn", None)
            if snapshot is None:
                continue
            if not isinstance(snapshot, dict):
                raise TypeError(
                    f"executor seat {seat} debug_trace_turn must be a dict or None")
            if (("day" in snapshot and snapshot["day"] != day)
                    or ("hour" in snapshot and snapshot["hour"] != hour)):
                continue
            available[str(seat)] = snapshot
        return available or None

    def record_decision_trace(
            self, actions: Sequence[Mapping[str, Any]]) -> None:
        if self.trace_recorder is None:
            return
        if self.current_canonical_state is None:  # pragma: no cover - seam guard
            raise RuntimeError("trace decision recorded before canonical state")
        canonical = self.current_canonical_state
        self.trace_recorder.append_turn(
            step=canonical["step"],
            day=canonical["day"],
            hour=canonical["hour"],
            canonical_state=canonical,
            joint_actions={str(seat): copy.deepcopy(dict(action))
                           for seat, action in enumerate(actions)},
            executor_debug=self._executor_debug_for_turn(
                day=canonical["day"], hour=canonical["hour"]),
        )

    def record_terminal_trace(self) -> dict[str, Any] | None:
        if self.trace_recorder is None:
            return None
        if self.current_canonical_state is None:  # pragma: no cover - seam guard
            raise RuntimeError("trace finalized before canonical state")
        canonical = self.current_canonical_state
        self.trace_recorder.append_turn(
            step=canonical["step"],
            day=canonical["day"],
            hour=canonical["hour"],
            canonical_state=canonical,
        )
        return self.trace_recorder.build()

    def _note_day_start(self, seat: int) -> None:
        obs = self.obs[seat]
        day = int(obs["day"])
        if day <= self.last_seen_day[seat]:
            return
        if self.last_seen_day[seat] >= 0:
            # Day rollover: realized labor of the finished day comes from the
            # observed hires_today progression, never from HIRE intents.
            hires = self.hires_current_day[seat]
            self.previous_execution[seat] = {
                "workers_hired": hires,
                "hire_cost": total_hire_cost(hires),
            }
            # The finished day's primitive turns are all hashed by now:
            # seal its joint-action digest (idempotent; the other seat's
            # rollover finds the hasher already popped).
            self.seal_day_digest(self.last_seen_day[seat])
        self.last_seen_day[seat] = day
        self.hires_current_day[seat] = int(
            obs["farms"][seat].get("hires_today", 0) or 0)
        self.daily_start[seat] = (
            day, float(obs["farms"][seat]["money"]))

    def track_post_step(self) -> None:
        for seat in range(2):
            hires = int(self.obs[seat]["farms"][seat].get("hires_today", 0) or 0)
            self.hires_current_day[seat] = max(
                self.hires_current_day[seat], hires)
            self._note_day_start(seat)

    def encode_seat(self, seat: int) -> dict[str, np.ndarray]:
        obs = self.obs[seat]
        return encode_live_inputs(
            obs, seat, dict(self.previous_execution[seat]),
            step=int(obs["step"]),
            economic_prev_start=self.daily_start[seat])

    def hash_joint_action(self, day: int, hour: int,
                          actions: list[Mapping[str, Any]]) -> None:
        hasher = self.day_hashers.get(day)
        if hasher is None:
            hasher = hashlib.sha256()
            self.day_hashers[day] = hasher
        payload = canonical_json({
            "hour": hour,
            "actions": [
                {"farmer": list(a.get("farmer", [])),
                 "hands": [list(h) for h in a.get("hands", [])],
                 "market": [list(m) for m in a.get("market", [])]}
                for a in actions
            ],
        })
        hasher.update(payload.encode("utf-8"))

    def seal_day_digest(self, day: int) -> None:
        """Finalize a completed day's primitive joint-action digest."""
        hasher = self.day_hashers.pop(day, None)
        if hasher is not None:
            self.day_digests[day] = hasher.digest()


class _BatchedSlotBackend:
    """Scalar-shaped view over one slot in a shared batched backend."""

    name = "fast-batched"

    def __init__(self, batch: BatchedEngineBackend, index: int) -> None:
        self._observations: list[dict[str, Any]] = []
        self._rewards = [0.0, 0.0]
        self._statuses = ["ACTIVE", "ACTIVE"]

    def update(
        self,
        observations: list[dict[str, Any]],
        rewards: list[float],
        statuses: list[str],
    ) -> None:
        self._observations = observations
        self._rewards = list(rewards)
        self._statuses = list(statuses)

    def canonical_state(self) -> dict[str, Any]:
        return canonical_state_fast(self._observations, self._rewards,
                                    self._statuses)

    @property
    def rewards(self) -> list[float]:
        return list(self._rewards)

    @property
    def statuses(self) -> list[str]:
        return list(self._statuses)


class SelfPlayRunner:
    """Batched-lockstep self-play over `oracle.backend` engine instances."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        trajectory_buffer: TrajectoryBuffer | None = None,
        executor_factory: Any | None = None,
        master_seed: int | None = None,
    ) -> None:
        self.config = config
        if config.debug_trace_seat not in (None, 0, 1):
            raise ValueError("debug_trace_seat must be None, 0, or 1")
        if not isinstance(config.debug_trace_view, str) \
                or not config.debug_trace_view:
            raise ValueError("debug_trace_view must be a non-empty string")
        self.buffer = trajectory_buffer
        if executor_factory is None:
            if config.low_telemetry:
                from executor_v0.agent import AgentConfig

                executor_factory = make_default_executor_factory(
                    AgentConfig(strict=True, record_turn_snapshot=False,
                                optional_spare_watering=True))
            else:
                # Preserve the zero-argument factory seam used by callers and
                # tests; the default factory already creates strict agents.
                executor_factory = make_default_executor_factory()
        self.executor_factory = executor_factory
        self.master_seed = master_seed
        self.timing_totals: dict[str, float] = {
            "manager_inference": 0.0, "agent_actions": 0.0,
            "env_step": 0.0, "orchestration": 0.0}
        self.provenance: dict[str, Any] = {
            "opening": opening_provenance(config.opening),
            "backend": backend_provenance(
                config.backend_name, config.backend_configuration),
            "executor_factory": self.executor_factory,
            "executor_factory_version": getattr(
                self.executor_factory, "version", "unknown"),
            "master_seed": master_seed,
            "manager_start_day": config.manager_start_day,
        }
        if config.batch_backend and config.backend_name != "fast":
            raise ValueError("batch_backend currently supports backend_name='fast' only")

    # ---------------------------------------------------------------- run
    def run(self, specs: Sequence[EpisodeSpec]) -> list[EpisodeResult]:
        if not specs:
            return []
        num_envs = max(1, int(self.config.num_envs))
        results: list[EpisodeResult] = []
        for start in range(0, len(specs), num_envs):
            results.extend(self._run_chunk(list(specs[start:start + num_envs])))
        return results

    # ------------------------------------------------------------- lockstep
    def _run_chunk(self, specs: list[EpisodeSpec]) -> list[EpisodeResult]:
        if self.config.batch_backend:
            return self._run_chunk_batched(specs)
        states = [_EpisodeState(spec, self.config, self.provenance)
                  for spec in specs]
        for state in states:
            state.observe_reset()
        results: list[EpisodeResult] = []

        for _turn in range(self.config.max_turns):
            active = [s for s in states if not s.done]
            if not active:
                break

            t0 = time.perf_counter()
            self._collect_and_apply_decisions(active)
            t1 = time.perf_counter()

            per_state_actions: list[list[Mapping[str, Any]]] = []
            for state in active:
                observations = (
                    state.agent_obs if state.config.read_only_agent_observations
                    else [copy.deepcopy(view) for view in state.obs])
                actions = [state.openings[seat](observations[seat])
                           for seat in range(2)]
                day = int(state.obs[0]["day"])
                hour = int(state.obs[0]["hour"])
                state.hash_joint_action(day, hour, actions)
                if state.rollout is not None:
                    state.rollout.joint_actions.append((
                        int(state.obs[0]["step"]), day, hour,
                        copy.deepcopy(dict(actions[0])),
                        copy.deepcopy(dict(actions[1]))))
                state.record_decision_trace(actions)
                per_state_actions.append(actions)
            t2 = time.perf_counter()

            for state, actions in zip(active, per_state_actions):
                obs, _rewards, _statuses = state.backend.step(actions)
                state.obs = state._adapt_observations(obs)
            t3 = time.perf_counter()

            newly_done: list[_EpisodeState] = []
            for state in active:
                state.track_post_step()
                if state.backend.statuses == ["DONE", "DONE"]:
                    state.done = True
                    newly_done.append(state)
            t4 = time.perf_counter()

            self.timing_totals["manager_inference"] += t1 - t0
            self.timing_totals["agent_actions"] += t2 - t1
            self.timing_totals["env_step"] += t3 - t2
            self.timing_totals["orchestration"] += (t4 - t3)

            for state in newly_done:
                results.append(self._finalize(state))

        # Turn budget exhausted without terminal status: truncate remaining.
        for state in states:
            if not state.finalized:
                state.truncated = True
                results.append(self._finalize(state))
        return results

    def _run_chunk_batched(self, specs: list[EpisodeSpec]) -> list[EpisodeResult]:
        """Run a chunk through one native batch owner.

        The executor and manager remain per-episode/per-seat Python objects;
        only explicit action submission and native state ownership are batched.
        Done slots receive PASS rows so a shorter episode cannot shift rows for
        the still-active slots.
        """
        batch: BatchedEngineBackend = make_batched_backend(
            self.config.backend_name, len(specs), self.config.backend_configuration
        )
        states: list[_EpisodeState] = []
        for index, spec in enumerate(specs):
            slot = _BatchedSlotBackend(batch, index)
            states.append(_EpisodeState(spec, self.config, self.provenance, slot))
        observations = batch.reset([state.spec.seed for state in states])
        for index, (state, observation) in enumerate(zip(states, observations)):
            state.backend.update(observation, batch.rewards(index),
                                 batch.statuses(index))
            state.obs = state._adapt_observations(observation)
            for seat in range(2):
                state._note_day_start(seat)

        results: list[EpisodeResult] = []
        pass_pair = [
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        for _turn in range(self.config.max_turns):
            active = [state for state in states if not state.done]
            if not active:
                break

            t0 = time.perf_counter()
            self._collect_and_apply_decisions(active)
            t1 = time.perf_counter()

            batch_actions = [pass_pair for _ in states]
            for index, state in enumerate(states):
                if state.done:
                    continue
                observations = (
                    state.agent_obs
                    if state.config.read_only_agent_observations
                    else [copy.deepcopy(view) for view in state.obs])
                actions = [state.openings[seat](observations[seat])
                           for seat in range(2)]
                day = int(state.obs[0]["day"])
                hour = int(state.obs[0]["hour"])
                state.hash_joint_action(day, hour, actions)
                if state.rollout is not None:
                    state.rollout.joint_actions.append((
                        int(state.obs[0]["step"]), day, hour,
                        copy.deepcopy(dict(actions[0])),
                        copy.deepcopy(dict(actions[1]))))
                state.record_decision_trace(actions)
                batch_actions[index] = actions
            t2 = time.perf_counter()

            observations, _rewards, _statuses = batch.step(batch_actions)
            for index, state in enumerate(states):
                state.backend.update(observations[index], batch.rewards(index),
                                     batch.statuses(index))
                if not state.done:
                    state.obs = state._adapt_observations(observations[index])
            t3 = time.perf_counter()

            newly_done: list[_EpisodeState] = []
            for state in active:
                state.track_post_step()
                if state.backend.statuses == ["DONE", "DONE"]:
                    state.done = True
                    newly_done.append(state)
            t4 = time.perf_counter()

            self.timing_totals["manager_inference"] += t1 - t0
            self.timing_totals["agent_actions"] += t2 - t1
            self.timing_totals["env_step"] += t3 - t2
            self.timing_totals["orchestration"] += t4 - t3
            for state in newly_done:
                results.append(self._finalize(state))

        for state in states:
            if not state.finalized:
                state.truncated = True
                results.append(self._finalize(state))
        return results

    # ------------------------------------------------- batched manager day
    def _collect_and_apply_decisions(
        self,
        active: list[_EpisodeState],
    ) -> None:
        start_day = self.config.manager_start_day
        requests: list[tuple[_EpisodeState, int, int]] = []
        for state in active:
            day = int(state.obs[0]["day"])
            hour = int(state.obs[0]["hour"])
            if hour != 0 or day < start_day:
                continue
            for seat in range(2):
                if day not in state.planned_days[seat]:
                    requests.append((state, seat, day))
        if not requests:
            return

        # Group by policy identity across ALL envs: exactly one policy call
        # per (identity, day) with a contiguous stacked batch.
        groups: dict[tuple[str, int], list[tuple[_EpisodeState, int, int]]] = {}
        encodings: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
        for state, seat, day in requests:
            inputs = state.encode_seat(seat)
            encodings[(id(state), seat, day)] = inputs
            key = (state.spec.policies[seat].identity.identity_id(), day)
            groups.setdefault(key, []).append((state, seat, day))

        for (identity_id, day), group in sorted(groups.items()):
            policy = group[0][0].spec.policies[group[0][1]]
            first_state, first_seat, first_day = group[0]
            keys = sorted(encodings[(id(first_state), first_seat,
                                     first_day)].keys())
            batch: dict[str, np.ndarray] = {}
            for key in keys:
                batch[key] = np.concatenate(
                    [encodings[(id(state), seat, day)][key]
                     for state, seat, day in group], axis=0)
            set_context = getattr(policy, "set_request_context", None)
            if callable(set_context):
                set_context([
                    (state.spec.episode_index, seat, request_day)
                    for state, seat, request_day in group])
            prng_id = f"episode={group[0][0].spec.episode_index}/day={day}/" \
                      f"policy={identity_id}"
            outputs = policy.plan_batch(batch, prng_id)
            expected = len(group)
            if outputs.batch_size != expected:
                raise ValueError(
                    f"policy returned batch {outputs.batch_size}, expected "
                    f"{expected}")
            plans = plans_from_action_tensors(outputs.action_tensors)
            for row, (state, seat, day) in enumerate(group):
                self._record_transition(state, seat, day, policy, outputs,
                                        row, encodings[(id(state), seat, day)],
                                        plans[row])

    def _record_transition(
        self,
        state: _EpisodeState,
        seat: int,
        day: int,
        policy: BatchedPlanPolicy,
        outputs: Any,
        row: int,
        inputs: Mapping[str, np.ndarray],
        plan: DailyPlan,
    ) -> None:
        action_tensors = {
            name: array[row:row + 1]
            for name, array in outputs.action_tensors.items()}
        logprob_groups = {
            name: float(array[row]) for name, array in
            outputs.logprob_groups.items()}
        opponent = state.spec.policies[1 - seat]
        trainable = seat in state.spec.trainable_seats
        # `inputs` is already this request's own single-row encoding
        # ([1, ...]); only the batched policy outputs are row-sliced.
        single_inputs = dict(inputs)

        transition = Transition(
            episode_index=state.spec.episode_index,
            seed=state.spec.seed,
            seat=seat,
            day=day,
            trainable=trainable,
            inputs=single_inputs,
            action_tensors=action_tensors,
            logprob_groups=logprob_groups,
            logprob_total=float(outputs.logprob_total[row]),
            value=float(outputs.value[row]),
            trace_digest=hashlib.sha256(b"unsealed").digest(),
        )
        metadata = TransitionMetadata(
            index=-1,
            episode_index=state.spec.episode_index,
            seed=state.spec.seed,
            seat=seat,
            day=day,
            policy_id=policy.identity.identity_id(),
            policy_version=policy.identity.version,
            policy_fingerprint=policy.identity.fingerprint,
            opponent_id=opponent.identity.identity_id(),
            trainable=trainable,
            plan_json=plan.to_json_dict(),
        )
        # Queue the decoded plan for the unmodified executor to consume at
        # its first turn of this day (exact once-per-day manager decision).
        state.providers[seat].queue(day, plan)
        # Track (seat, day) ownership regardless of buffering so episode
        # transition counts and digest patching stay correct in no-buffer
        # evaluation mode; the value is the buffer row index or None.
        index = len(self.buffer) if self.buffer is not None else None
        if self.buffer is not None:
            self.buffer.append(transition, metadata)
        state.transition_index[(seat, day)] = index
        state.planned_days[seat].add(day)

        if state.rollout is not None:
            digest = hashlib.sha256()
            for key in sorted(single_inputs):
                digest.update(key.encode("utf-8"))
                digest.update(
                    np.ascontiguousarray(single_inputs[key]).tobytes())
            state.rollout.manager_input_digests[(seat, day)] = \
                digest.hexdigest()

    # ------------------------------------------------------------ finalize
    def _finalize(self, state: _EpisodeState) -> EpisodeResult:
        state.finalized = True
        debug_trace = state.record_terminal_trace()
        # Seal the final day's primitive-action digest and patch rows.
        current_day = int(state.obs[0]["day"])
        state.seal_day_digest(current_day)
        for day, digest in state.day_digests.items():
            for seat in range(2):
                index = state.transition_index.get((seat, day))
                if index is not None and self.buffer is not None:
                    self.buffer.patch_trace_digest(index, digest)

        banks = [float(state.obs[seat]["farms"][seat]["money"])
                 for seat in range(2)]
        margin = banks[0] - banks[1]
        winner_seat = 0 if margin > 0 else (1 if margin < 0 else -1)
        rewards = [0.0, 0.0]
        if winner_seat >= 0:
            rewards[winner_seat] = 1.0
            rewards[1 - winner_seat] = -1.0
        terminated = not state.truncated

        if self.buffer is not None:
            for seat in range(2):
                days = sorted(
                    day for (s, day) in state.transition_index if s == seat)
                if days:
                    last_index = state.transition_index[(seat, days[-1])]
                    self.buffer.patch_terminal(last_index, rewards[seat],
                                               terminated)
                    if state.truncated:
                        self.buffer.patch_truncated(last_index)
            self._patch_executor_diagnostics(state)

        opening_diagnostics = [agent.diagnostics_json()
                               for agent in state.openings]
        executor_diagnostics: list[dict[str, Any]] = []
        for seat, executor in enumerate(state.executors):
            try:
                diagnostics_fn = getattr(executor, "diagnostics_json", None)
                diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
                compact = {
                    "seat": seat,
                    "fallback_errors": copy.deepcopy(
                        diagnostics.get("fallback_errors", [])),
                }
                for key in ("runtime_errors", "runtime_error", "exception",
                            "illegal_actions", "provider_diagnostics"):
                    if key in diagnostics:
                        compact[key] = copy.deepcopy(diagnostics[key])
                executor_diagnostics.append(compact)
            except Exception as exc:  # noqa: BLE001 - report diagnostic failure
                executor_diagnostics.append({
                    "seat": seat, "runtime_error": repr(exc),
                })
        if state.rollout is not None:
            state.rollout.opening_handoff = copy.deepcopy(opening_diagnostics)
            state.rollout.plans = {
                (seat, day): dict(self.buffer.sidecar_records[index].plan_json)
                for (seat, day), index in state.transition_index.items()
            } if self.buffer is not None else {}

        episode_digest = hashlib.sha256()
        for day in sorted(state.day_digests):
            episode_digest.update(state.day_digests[day])

        return EpisodeResult(
            episode_index=state.spec.episode_index,
            seed=state.spec.seed,
            composition=state.spec.composition,
            final_banks=banks,
            margin=margin,
            winner_seat=winner_seat,
            rewards=rewards,
            statuses=list(state.backend.statuses),
            transitions=len(state.transition_index),
            terminated=terminated,
            opening_diagnostics=opening_diagnostics,
            trace_digest=episode_digest.hexdigest(),
            rollout=state.rollout,
            timing_seconds=dict(self.timing_totals),
            policy_identities=tuple(
                {
                    "seat": seat,
                    "policy": state.spec.policies[seat].identity
                        .to_json_dict(),
                    "opponent": state.spec.policies[1 - seat].identity
                        .to_json_dict(),
                    "trainable": seat in state.spec.trainable_seats,
                }
                for seat in range(2)),
            debug_trace=debug_trace,
            executor_diagnostics=executor_diagnostics,
        )

    # ------------------------------------------------------------- artifact
    def build_artifact_metadata(self, result: EpisodeResult) -> dict[str, Any]:
        """Automatic artifact metadata: episode outcome + runner provenance.

        The normal artifact path — callers never hand-assemble `run_metadata`.
        """
        return build_artifact_metadata(self.provenance, result)

    def save_trajectory_artifact(
        self,
        path: str | Path,
        buffer: TrajectoryBuffer,
        result: EpisodeResult,
    ) -> Path:
        """Persist `<path>.npz` + `<path>.json` with full provenance merged
        automatically (low-level `TrajectoryBuffer.save` stays unchanged)."""
        return buffer.save(
            path, run_metadata=self.build_artifact_metadata(result))

    def _patch_executor_diagnostics(self, state: _EpisodeState) -> None:
        """Compact per-day executor diagnostics into sidecar metadata."""
        for seat in range(2):
            try:
                diagnostics = state.executors[seat].diagnostics_json()
            except Exception:  # noqa: BLE001 - diagnostics must never break
                continue
            days = diagnostics.get("days", {})
            for (s, day), index in state.transition_index.items():
                if s != seat or str(day) not in days:
                    continue
                record = days[str(day)]
                compact = {
                    "unfinished_tasks": len(record.get("unfinished_tasks", [])),
                    "missed_maintenance": len(
                        record.get("missed_maintenance", [])),
                    "foreman_counts": dict(record.get("foreman_counts", {})),
                    "hires_submitted": record.get("hires", {}).get(
                        "submitted", 0),
                    "sells_bins": {
                        anchor: sum(entry["submitted"] for entry in bins.values())
                        for anchor, bins in record.get("sells", {}).items()},
                }
                self.buffer.sidecar_records[index].executor_day_diagnostics = \
                    compact
