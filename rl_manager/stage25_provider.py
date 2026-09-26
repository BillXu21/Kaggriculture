"""Persistent Stage 2.5 manager lifecycle/provider seam.

The provider owns the state which must outlive one executor call: the physical
crop baseline observed at the accepted boundary, one corrected-E daily-start
tracker, and one immutable accepted plan for each episode/seat.  The external
path accepts class indices from a parent process and imports no policy
framework.  The optional native path is loaded lazily so importing this module
remains framework-neutral.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from numbers import Integral
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    normalize_e_history_version,
)
from bc_manager.live import encode_live_inputs, validate_previous_execution
from executor_v0.plan import DailyPlan, SELL_BIN_ANCHORS
from replay_daily.constants import PRODUCTS
from replay_daily.lifecycle import (
    canonical_board,
    replaceable_today,
    resolve_observation_step,
)

from .stage25_config import (
    Stage25CurriculumConfig,
    apply_animal_curriculum,
    apply_crop_curriculum,
    apply_land_curriculum,
    curriculum_fingerprint,
)
from .stage25_mechanics import (
    ACTION_CLASS_COUNTS,
    ACTION_ORDER,
    CROP_ORDER,
    PhysicalContext,
    animal_target_support_mask,
    crop_class_to_delta,
    crop_delta_support_mask,
    available_crop_slots,
    physical_context_from_board,
    physical_crop_capacity,
    land_target_support_mask,
    land_class_to_target,
    animal_class_to_target,
    transition_crop_goal,
    unplaced_animal_counts,
)
from .stage25_types import Stage25BehaviorIdentity, stage25_row_token

STATE_VERSION = "stage25_provider_state_v3"
SOURCE_TRANSFER = "encoder_only"
DecisionMode = Literal["deterministic", "stochastic"]


def _lower_plan(action_classes: Sequence[int], crop_goals: Sequence[int]) -> DailyPlan:
    # Import the root-owned lowering adapter only at the transport boundary.
    module = importlib.import_module("rl_manager.stage25_plan")
    return module.lower_stage25_daily_plan(action_classes, crop_goals)


class Stage25ProviderError(ValueError):
    """Base error for an invalid lifecycle boundary."""


class Stage25DuplicateDecisionError(Stage25ProviderError):
    """A second class submission was made for an already accepted boundary."""


class Stage25OutOfOrderError(Stage25ProviderError):
    """A decision boundary was delivered in an invalid order."""


class Stage25TerminalError(Stage25ProviderError):
    """Terminal observations never create or apply a manager action."""


@dataclass(frozen=True)
class _FrozenMapping(Mapping[str, Any]):
    """Small pickle-safe immutable mapping for retained inference premises."""

    entries: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class Stage25InferenceContext:
    """Read-only pre-decision payload handed to a parent inference owner."""

    decision_key: Stage25DecisionKey
    behavior_identity: Stage25BehaviorIdentity
    inputs: Mapping[str, np.ndarray]
    crop_capacity: tuple[int, ...]
    physical_context: PhysicalContext
    daily_start: tuple[int, float]
    curriculum: Stage25CurriculumConfig
    seed: int = 0
    support: Mapping[str, Any] | None = None
    row_token: int | None = None

    @property
    def request_id(self) -> str:
        return (f"episode={self.decision_key.episode_id}/"
                f"seat={self.decision_key.seat}/day={self.decision_key.day}"
                f"/behavior={self.behavior_identity.identity_id()}")


def _scalar_int(value: object, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Stage25ProviderError(f"{what} must be an integer, got {value!r}")
    result = int(value)
    if result < minimum:
        raise Stage25ProviderError(
            f"{what} must be >= {minimum}, got {result}")
    return result


def _class_tuple(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise Stage25ProviderError("action classes must be a sequence of integers")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise Stage25ProviderError(
            "action classes must be a sequence of integers") from exc
    if len(result) != len(ACTION_ORDER):
        raise Stage25ProviderError(
            f"action classes must contain {len(ACTION_ORDER)} values, "
            f"got {len(result)}")
    normalized: list[int] = []
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise Stage25ProviderError(
                f"action class {index} must be an integer, got {value!r}")
        normalized.append(int(value))
    return tuple(normalized)


def _mode(value: str) -> DecisionMode:
    if value not in ("deterministic", "stochastic"):
        raise ValueError(
            f"mode must be 'deterministic' or 'stochastic', got {value!r}")
    return value  # type: ignore[return-value]


def _json_scalar(value: object, what: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise Stage25ProviderError(
            f"{what} must be a JSON string or integer, got {value!r}")
    return value


def _goal_int(value: object, what: str) -> int:
    result = _scalar_int(value, what)
    if result > 100:
        raise Stage25ProviderError(f"{what} must be in [0, 100], got {result}")
    return result


@dataclass(frozen=True)
class Stage25DecisionKey:
    """Immutable identity of one episode/seat/day manager boundary."""

    episode_id: str | int
    seat: int
    day: int
    row_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _json_scalar(
            self.episode_id, "episode_id"))
        object.__setattr__(self, "seat", _scalar_int(self.seat, "seat"))
        if self.seat not in (0, 1):
            raise Stage25ProviderError(f"seat must be 0 or 1, got {self.seat}")
        object.__setattr__(self, "day", _scalar_int(self.day, "day"))
        if self.row_id is not None and (not isinstance(self.row_id, str)
                                         or not self.row_id):
            raise Stage25ProviderError("row_id must be a non-empty string")

    @property
    def identity(self) -> str:
        value = f"episode={self.episode_id}/seat={self.seat}/day={self.day}"
        return value if self.row_id is None else f"{value}/row={self.row_id}"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seat": self.seat,
            "day": self.day,
            "row_id": self.row_id,
        }

    @classmethod
    def from_json_dict(cls, value: Mapping[str, Any]) -> "Stage25DecisionKey":
        if not isinstance(value, Mapping) or set(value) != {
                "episode_id", "seat", "day", "row_id"}:
            raise Stage25ProviderError(
                "decision key must contain exactly episode_id, seat, day, row_id")
        return cls(value["episode_id"], value["seat"], value["day"],
                   value["row_id"])


def _curriculum_json(config: Stage25CurriculumConfig) -> dict[str, Any]:
    return {
        "version": config.version,
        "enabled": config.enabled,
        "max_positive_crop_delta": config.max_positive_crop_delta,
        "max_land_expansion_per_decision":
            config.max_land_expansion_per_decision,
        "max_animal_additions_per_species_per_decision":
            config.max_animal_additions_per_species_per_decision,
    }


def _curriculum_from_json(value: object) -> Stage25CurriculumConfig:
    if not isinstance(value, Mapping) or set(value) != {
            "version", "enabled", "max_positive_crop_delta",
            "max_land_expansion_per_decision",
            "max_animal_additions_per_species_per_decision"}:
        raise Stage25ProviderError("curriculum has an unexpected schema")
    try:
        return Stage25CurriculumConfig(**dict(value))
    except (TypeError, ValueError) as exc:
        raise Stage25ProviderError(f"curriculum is invalid: {exc}") from exc


@dataclass(frozen=True)
class Stage25LifecycleState:
    """Serializable snapshot of lifecycle state, excluding executor queues."""

    episode_id: str | int
    seat: int
    manager_start_day: int
    physical_crop_baseline: tuple[int, ...] | None = None
    accepted_decision: Stage25DecisionKey | None = None
    accepted_classes: tuple[int, ...] | None = None
    crop_goals: tuple[int, ...] | None = None
    cached_plan: DailyPlan | None = None
    e_history: tuple[int, float] | None = None
    e_history_version: str = E_HISTORY_CORRECTED_V1
    source_history_version: str | None = None
    curriculum: Stage25CurriculumConfig | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "episode_id": self.episode_id,
            "seat": self.seat,
            "manager_start_day": self.manager_start_day,
            "physical_crop_baseline": (
                None if self.physical_crop_baseline is None
                else list(self.physical_crop_baseline)),
            "accepted_decision": None if self.accepted_decision is None
            else self.accepted_decision.to_json_dict(),
            "accepted_classes": None if self.accepted_classes is None
            else list(self.accepted_classes),
            "crop_goals": None if self.crop_goals is None else list(self.crop_goals),
            "cached_plan": None if self.cached_plan is None
            else self.cached_plan.to_json_dict(),
            "e_history": None if self.e_history is None else {
                "day": self.e_history[0], "money": self.e_history[1],
            },
            "e_history_version": self.e_history_version,
            "source_history_version": self.source_history_version,
            "curriculum": None if self.curriculum is None
            else _curriculum_json(self.curriculum),
        }

    @classmethod
    def from_json_dict(cls, value: Mapping[str, Any]) -> "Stage25LifecycleState":
        expected = {
            "version", "episode_id", "seat", "manager_start_day",
            "physical_crop_baseline", "accepted_decision", "accepted_classes",
            "crop_goals", "cached_plan", "e_history", "e_history_version",
            "source_history_version", "curriculum",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Stage25ProviderError(
                "provider state has an unexpected schema or missing fields")
        if value["version"] != STATE_VERSION:
            raise Stage25ProviderError(
                f"provider state version {value.get('version')!r} is incompatible")
        physical_crop_baseline = (
            None if value["physical_crop_baseline"] is None else tuple(
                _goal_int(item, "physical crop baseline item")
                for item in value["physical_crop_baseline"]))
        if (physical_crop_baseline is not None
                and len(physical_crop_baseline) != len(CROP_ORDER)):
            raise Stage25ProviderError(
                "physical_crop_baseline must contain five values")
        accepted = value["accepted_decision"]
        key = None if accepted is None else Stage25DecisionKey.from_json_dict(accepted)
        classes = None if value["accepted_classes"] is None else _class_tuple(
            value["accepted_classes"])
        goals = None if value["crop_goals"] is None else tuple(
            _goal_int(item, "crop goal") for item in value["crop_goals"])
        if goals is not None and len(goals) != len(CROP_ORDER):
            raise Stage25ProviderError("crop_goals must contain five values")
        plan_value = value["cached_plan"]
        plan = None
        if plan_value is not None:
            # The canonical DailyPlan view stores sell rows by anchor, whereas
            # create() consumes product -> anchor.  Rebuild that one field
            # explicitly so import remains strict and lossless.
            if not isinstance(plan_value, Mapping):
                raise Stage25ProviderError("cached_plan must be an object")
            if set(plan_value) != {
                    "crop_targets", "animal_targets", "land_count",
                    "fertilizer_by_crop", "care_by_animal", "sell_quantities"}:
                raise Stage25ProviderError("cached_plan has unexpected fields")
            sell_anchors, products = SELL_BIN_ANCHORS, PRODUCTS
            sell_rows = plan_value["sell_quantities"]
            if (not isinstance(sell_rows, Mapping)
                    or set(sell_rows) != {str(anchor) for anchor in sell_anchors}
                    or any(not isinstance(sell_rows[str(anchor)], Mapping)
                           or set(sell_rows[str(anchor)]) != set(products)
                           for anchor in sell_anchors)):
                raise Stage25ProviderError("cached_plan sell quantities are not canonical")
            sell = {
                product: {
                    anchor: sell_rows[str(anchor)][product]
                    for anchor in sell_anchors
                }
                for product in products
            }
            plan = DailyPlan.create(
                crop_targets=plan_value["crop_targets"],
                animal_targets=plan_value["animal_targets"],
                land_count=plan_value["land_count"],
                fertilizer_by_crop=plan_value["fertilizer_by_crop"],
                care_by_animal=plan_value["care_by_animal"],
                sell_quantities=sell,
            )
        history_value = value["e_history"]
        history = None
        if history_value is not None:
            if (not isinstance(history_value, Mapping)
                    or set(history_value) != {"day", "money"}):
                raise Stage25ProviderError("e_history must contain day and money")
            day = _scalar_int(history_value["day"], "e_history.day")
            money = float(history_value["money"])
            if not math.isfinite(money):
                raise Stage25ProviderError("e_history.money must be finite")
            history = (day, money)
        operating = normalize_e_history_version(value["e_history_version"])
        source = value["source_history_version"]
        if source is not None:
            source = normalize_e_history_version(source)
        curriculum_value = value["curriculum"]
        curriculum = None if curriculum_value is None else \
            _curriculum_from_json(curriculum_value)
        return cls(
            episode_id=_json_scalar(value["episode_id"], "episode_id"),
            seat=value["seat"], manager_start_day=value["manager_start_day"],
            physical_crop_baseline=physical_crop_baseline,
            accepted_decision=key,
            accepted_classes=classes, crop_goals=goals, cached_plan=plan,
            e_history=history, e_history_version=operating,
            source_history_version=source, curriculum=curriculum,
        )


def _validate_action(
    classes: tuple[int, ...], crop_baseline: tuple[int, ...],
    context: PhysicalContext, curriculum: Stage25CurriculumConfig,
) -> tuple[int, ...]:
    """Validate one action against the current physical crop baseline."""
    for index, (value, count) in enumerate(zip(classes, ACTION_CLASS_COUNTS)):
        if not 0 <= value < count:
            raise Stage25ProviderError(
                f"action class {index}={value} is outside [0, {count})")
    land_class = classes[0]
    land_support = apply_land_curriculum(
        land_target_support_mask(context.observed_land),
        context.observed_land, curriculum)
    if not land_support[land_class]:
        raise Stage25ProviderError(
            f"land class {land_class} is outside physical/curriculum support")
    land_target = land_class_to_target(land_class)

    animal_targets: list[int] = []
    for species in range(3):
        support = animal_target_support_mask(
            context, land_target, species, animal_targets)
        support = apply_animal_curriculum(
            support, context.placed_animals[species], curriculum)
        class_index = classes[1 + species]
        if not support[class_index]:
            raise Stage25ProviderError(
                f"animal class {class_index} at step {species + 1} is "
                "outside physical/curriculum support")
        animal_targets.append(animal_class_to_target(class_index))

    total_capacity = physical_crop_capacity(context, land_target, animal_targets)
    goals: list[int] = []
    for crop_index, class_index in enumerate(classes[4:]):
        previous = crop_baseline[crop_index]
        residual = total_capacity - sum(goals)
        support = apply_crop_curriculum(
            crop_delta_support_mask(previous, residual), curriculum)
        if not support[class_index]:
            raise Stage25ProviderError(
                f"crop {CROP_ORDER[crop_index]} class {class_index} is "
                "outside physical/curriculum support")
        goals.append(transition_crop_goal(
            previous, crop_class_to_delta(class_index)))
    return tuple(goals)


def _terminal_observation(obs: Mapping[str, Any]) -> bool:
    if any(bool(obs.get(key, False)) for key in ("terminal", "terminated", "done")):
        return True
    status = str(obs.get("status", "")).upper()
    return status in {"DONE", "TERMINAL", "TERMINATED", "FINISHED"}


class Stage25NativePolicy:
    """Lazy native-checkpoint adapter; importing it does not import JAX."""

    def __init__(
        self, checkpoint_path: str | Path, *, config: Any = None,
        seed: int = 0, mode: DecisionMode = "deterministic",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.seed = _scalar_int(seed, "native seed")
        self.mode = _mode(mode)
        self._params: Any = None
        self._config: Any = config
        self._loaded = False

    @staticmethod
    def _row_id(value: str) -> int:
        return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4],
                              "little") % (2**31 - 1)

    def _ensure_loaded(self) -> tuple[Any, Any, Any, Any]:
        if not self._loaded:
            # These imports are intentionally inside the first native call.
            from rl_manager.stage25_checkpoint import (
                load_stage25_inference_checkpoint,
            )
            from rl_manager.stage25_policy import (
                Stage25ModelConfig,
            )
            config = self._config
            if config is None:
                # The checkpoint loader validates and returns the stored
                # config through metadata; reconstruct it from that JSON.
                params, meta = load_stage25_inference_checkpoint(
                    self.checkpoint_path)
                # Rebuild the static config from the pure-JAX mirror.  The
                # native path must not pull in the legacy Torch model merely
                # to deserialize checkpoint metadata.
                from bc_manager_jax.model import ManagerConfig
                config_payload = dict(meta["config"])
                config_payload["manager_config"] = ManagerConfig(
                    **dict(config_payload["manager_config"]))
                config_payload["curriculum"] = Stage25CurriculumConfig(
                    **dict(config_payload["curriculum"]))
                config = Stage25ModelConfig(**config_payload)
            else:
                params, _ = load_stage25_inference_checkpoint(
                    self.checkpoint_path, config=config)
            self._params = params
            self._config = config
            self._loaded = True
        from rl_manager.stage25_policy import greedy_act, stochastic_act
        return self._params, self._config, greedy_act, stochastic_act

    def load_config(self) -> Any:
        """Return the effective static config, loading the checkpoint once.

        The provider uses this to bind its validation curriculum to the exact
        checkpoint curriculum before sampling or mutating lifecycle state.
        """
        self._ensure_loaded()
        return self._config

    def act(
        self, inputs: Mapping[str, np.ndarray], physical_context: PhysicalContext,
        *, row_id: str, mode: DecisionMode | None = None,
        seed: int | None = None,
    ) -> tuple[int, ...]:
        selected_mode = _mode(self.mode if mode is None else mode)
        params, config, greedy_act, stochastic_act = self._ensure_loaded()
        if selected_mode == "deterministic":
            result = greedy_act(
                params, inputs, config, physical_contexts=(physical_context,),
                crop_capacity=inputs["crop_capacity"],
                row_ids=np.asarray([self._row_id(row_id)], dtype=np.int32),
            )
        else:
            import jax

            root_key = jax.random.PRNGKey(
                self.seed if seed is None else _scalar_int(seed, "native seed"))
            result = stochastic_act(
                params, inputs, config, rng_keys=np.asarray([root_key]),
                physical_contexts=(physical_context,),
                crop_capacity=inputs["crop_capacity"],
                row_ids=np.asarray([self._row_id(row_id)], dtype=np.int32),
            )
        values = np.asarray(result["classes"])[0]
        return _class_tuple(values)


class Stage25PlanProvider:
    """One lifecycle manager owner for one configured episode and seat."""

    def __init__(
        self, episode_id: str | int, seat: int, manager_start_day: int, *,
        native_checkpoint: str | Path | None = None,
        native_policy: Stage25NativePolicy | Any | None = None,
        policy: Stage25NativePolicy | Any | None = None,
        mode: DecisionMode = "deterministic", seed: int = 0,
        validation_mode: str = "strict",
        curriculum: Stage25CurriculumConfig | None = None,
        source_history_version: str | None = None,
        behavior_identity: Stage25BehaviorIdentity | None = None,
    ) -> None:
        self.episode_id = _json_scalar(episode_id, "episode_id")
        self.seat = _scalar_int(seat, "seat")
        if self.seat not in (0, 1):
            raise Stage25ProviderError(f"seat must be 0 or 1, got {self.seat}")
        self.manager_start_day = _scalar_int(
            manager_start_day, "manager_start_day")
        self.mode = _mode(mode)
        self.seed = _scalar_int(seed, "seed")
        if validation_mode not in ("strict", "fast", "none"):
            raise ValueError(
                "validation_mode must be 'strict', 'fast', or 'none'")
        self.validation_mode = validation_mode
        if curriculum is not None and not isinstance(
                curriculum, Stage25CurriculumConfig):
            raise TypeError("curriculum must be Stage25CurriculumConfig")
        # ``None`` means "adopt the native checkpoint's curriculum"; an explicit
        # value must agree with it before inference or state mutation.
        self._curriculum_explicit = curriculum is not None
        self.curriculum = curriculum or Stage25CurriculumConfig()
        self._bound_curriculum: Stage25CurriculumConfig | None = None
        self.e_history_version = E_HISTORY_CORRECTED_V1
        self.source_history_version = None if source_history_version is None else \
            normalize_e_history_version(source_history_version)
        if behavior_identity is not None and not isinstance(
                behavior_identity, Stage25BehaviorIdentity):
            raise TypeError("behavior_identity must be Stage25BehaviorIdentity")
        self.behavior_identity = behavior_identity
        if native_policy is not None and policy is not None:
            raise ValueError("pass native_policy or policy, not both")
        self._native_policy = native_policy or policy
        if self._native_policy is None and native_checkpoint is not None:
            self._native_policy = Stage25NativePolicy(
                native_checkpoint, seed=self.seed, mode=self.mode)
        self._native_checkpoint = None if native_checkpoint is None else Path(native_checkpoint)
        self.reset()

    def reset(self) -> None:
        """Clear all lifecycle state, including corrected-E history."""
        self._physical_crop_baseline: tuple[int, ...] | None = None
        self._accepted_key: Stage25DecisionKey | None = None
        self._accepted_classes: tuple[int, ...] | None = None
        self._crop_goals: tuple[int, ...] | None = None
        self._cached_plan: DailyPlan | None = None
        self._e_history: tuple[int, float] | None = None
        self._last_inputs: dict[str, np.ndarray] | None = None
        self._diagnostics: dict[str, Any] = {}
        self._pending_context: Stage25InferenceContext | None = None

    @property
    def expected_behavior_identity(self) -> Stage25BehaviorIdentity | None:
        """The immutable policy snapshot expected for external responses."""
        return self.behavior_identity

    @property
    def crop_capacity(self) -> tuple[int, ...] | None:
        """Return the last accepted physical baseline for key compatibility."""
        return self._physical_crop_baseline

    @property
    def physical_crop_baseline(self) -> tuple[int, ...] | None:
        """Return the physical crop counts observed at the last boundary."""
        return self._physical_crop_baseline

    @property
    def last_accepted_decision(self) -> Stage25DecisionKey | None:
        return self._accepted_key

    @property
    def last_accepted_day(self) -> int | None:
        return None if self._accepted_key is None else self._accepted_key.day

    @property
    def last_decision_identity(self) -> str | None:
        return None if self._accepted_key is None else self._accepted_key.identity

    @property
    def crop_goals(self) -> tuple[int, ...] | None:
        return self._crop_goals

    @property
    def cached_plan(self) -> DailyPlan | None:
        return self._cached_plan

    @property
    def e_history(self) -> tuple[int, float] | None:
        return self._e_history

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics

    def diagnostics_json(self) -> dict[str, Any]:
        """Return JSON-safe lifecycle diagnostics for executor telemetry."""
        return copy.deepcopy(dict(self._diagnostics))

    @property
    def encoded_inputs(self) -> Mapping[str, np.ndarray] | None:
        if self._last_inputs is None:
            return None
        return {key: np.array(value, copy=True)
                for key, value in self._last_inputs.items()}

    def effective_curriculum(self) -> Stage25CurriculumConfig:
        """Return the one effective curriculum, binding before first use.

        With a native checkpoint and no explicit curriculum, adopt the
        checkpoint's curriculum. An explicitly supplied curriculum must match
        it exactly; otherwise sampling would use one support while validation
        used another.
        """
        if self._bound_curriculum is None:
            checkpoint_curriculum = None
            loader = getattr(self._native_policy, "load_config", None)
            if callable(loader):
                checkpoint_curriculum = getattr(
                    loader(), "curriculum", None)
            if checkpoint_curriculum is None:
                candidate = self.curriculum
            elif self._curriculum_explicit:
                if self.curriculum != checkpoint_curriculum:
                    raise Stage25ProviderError(
                        "explicit provider curriculum does not match the native "
                        "checkpoint curriculum; refusing to sample or mutate state")
                candidate = self.curriculum
            else:
                candidate = checkpoint_curriculum
            if self.behavior_identity is not None:
                if (self.behavior_identity.curriculum_version !=
                        candidate.version or
                        self.behavior_identity.curriculum_fingerprint !=
                        curriculum_fingerprint(candidate)):
                    raise Stage25ProviderError(
                        "effective curriculum does not match the advertised "
                        "behavior identity; explicit configuration is required")
            # Commit the binding only after every identity/configuration check
            # has passed, so a rejected rollout leaves lifecycle state intact.
            self.curriculum = candidate
            self._bound_curriculum = candidate
        return self._bound_curriculum

    @property
    def state(self) -> Stage25LifecycleState:
        return Stage25LifecycleState(
            episode_id=self.episode_id, seat=self.seat,
            manager_start_day=self.manager_start_day,
            physical_crop_baseline=self._physical_crop_baseline,
            accepted_decision=self._accepted_key,
            accepted_classes=self._accepted_classes,
            crop_goals=self._crop_goals,
            cached_plan=self._cached_plan,
            e_history=self._e_history,
            e_history_version=self.e_history_version,
            source_history_version=self.source_history_version,
            curriculum=self.effective_curriculum(),
        )

    @property
    def provenance(self) -> dict[str, Any]:
        curriculum = self.effective_curriculum()
        result = {
            "e_history_version": self.e_history_version,
            "e_identity": {"variant": "E", "history_version": self.e_history_version},
            "curriculum_version": curriculum.version,
            "curriculum": _curriculum_json(curriculum),
        }
        if self.behavior_identity is not None:
            result["behavior_identity"] = self.behavior_identity.to_json_dict()
        if self.source_history_version is not None:
            result["source_history_version"] = self.source_history_version
            result["source_e_identity"] = {
                "variant": "E", "history_version": self.source_history_version,
                "transfer": SOURCE_TRANSFER,
            }
        return result

    def _key(self, day: int, key: Stage25DecisionKey | None,
             decision_id: str | None) -> Stage25DecisionKey:
        if key is not None and decision_id is not None:
            raise Stage25ProviderError("pass decision_key or decision_id, not both")
        if key is None:
            key = Stage25DecisionKey(
                self.episode_id, self.seat, day, decision_id)
        if (key.episode_id != self.episode_id or key.seat != self.seat):
            raise Stage25ProviderError(
                "decision key episode_id/seat does not match this provider")
        if key.day != day:
            raise Stage25ProviderError(
                f"decision key day {key.day} does not match observation day {day}")
        return key

    def _check_delivery(self, key: Stage25DecisionKey) -> None:
        if self._accepted_key is None:
            if key.day != self.manager_start_day:
                raise Stage25OutOfOrderError(
                    f"first manager boundary must be day {self.manager_start_day}, "
                    f"got {key.day}")
            return
        if key == self._accepted_key or key.day == self._accepted_key.day:
            raise Stage25DuplicateDecisionError(
                f"decision {key.identity} was already accepted or conflicts "
                "with the accepted day")
        if key.day != self._accepted_key.day + 1:
            raise Stage25OutOfOrderError(
                f"expected manager day {self._accepted_key.day + 1}, got {key.day}")

    def _stage_observation(
        self, obs: Mapping[str, Any], previous_execution: Mapping[str, int] | None,
        *, profile: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], PhysicalContext, tuple[int, ...]]:
        if not isinstance(obs, Mapping):
            raise Stage25ProviderError("obs must be a mapping")
        if "day" not in obs:
            raise Stage25ProviderError("obs is missing day")
        # One resolved absolute step feeds the live encoder and the physical
        # context, so both consumers share lifecycle timing. The caller's
        # observation is not mutated.
        #
        # Subphase timers are guarded so the profile-OFF path performs no
        # timing calls; when `profile` is None the body is unchanged.
        timer = time.perf_counter
        if profile is not None:
            phase_started = timer()
        step = resolve_observation_step(obs)
        if profile is not None:
            profile["stage25_provider_step_resolution_seconds"] += (
                timer() - phase_started)
            phase_started = timer()
        previous = validate_previous_execution(previous_execution)
        if profile is not None:
            profile["stage25_provider_previous_execution_validation_seconds"] += (
                timer() - phase_started)
            phase_started = timer()
        farm = obs["farms"][self.seat]
        board = canonical_board(farm["tiles"], int(obs["day"]), step)
        if profile is not None:
            profile["stage25_provider_canonical_board_seconds"] += (
                timer() - phase_started)
            phase_started = timer()
        inputs = encode_live_inputs(
            obs, self.seat, previous, step=step,
            economic_prev_start=self._e_history,
            e_history_version=self.e_history_version,
            canonical_self_board=board,
        )
        if profile is not None:
            profile["stage25_provider_encode_live_inputs_seconds"] += (
                timer() - phase_started)
            phase_started = timer()
        private = obs.get("private") or {}
        unplaced = unplaced_animal_counts(
            private.get("shed") or {}, private.get("inventories") or ())
        if profile is not None:
            profile["stage25_provider_unplaced_animals_seconds"] += (
                timer() - phase_started)
            phase_started = timer()
        context = physical_context_from_board(
            board, farm["unlocked_quadrants"],
            unplaced_animals=unplaced,
        )
        if profile is not None:
            profile["stage25_provider_physical_context_seconds"] += (
                timer() - phase_started)
        initial = context.observed_crop_counts
        inputs["crop_capacity"] = np.asarray(
            [initial], dtype=np.int16)
        inputs["replaceable_today"] = np.asarray(
            [replaceable_today(board, int(obs["day"]), step)], dtype=np.int16)
        inputs["available_crop_slots"] = np.asarray(
            [available_crop_slots(context)], dtype=np.int16)
        return inputs, context, initial

    @staticmethod
    def _freeze_inputs(
        inputs: Mapping[str, np.ndarray],
        profile: Mapping[str, Any] | None,
    ) -> dict[str, np.ndarray]:
        """Defensively copy inputs to immutable arrays.

        When `profile` is None this is exactly the historical copy loop with no
        timing calls or extra allocations beyond the required copies.
        """
        frozen_inputs: dict[str, np.ndarray] = {}
        if profile is not None:
            copy_started = time.perf_counter()
            copied_bytes = 0
        for name, value in inputs.items():
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            frozen_inputs[name] = copied
            if profile is not None:
                copied_bytes += int(copied.nbytes)
        if profile is not None:
            profile["stage25_provider_input_freeze_copy_seconds"] += (
                time.perf_counter() - copy_started)
            profile["stage25_provider_input_freeze_bytes"] += copied_bytes
        return frozen_inputs

    def _support_payload(
        self, context: PhysicalContext, physical_baseline: tuple[int, ...]
    ) -> dict[str, Any]:
        """Materialize the pre-decision support without sampling anything."""
        land = tuple(land_target_support_mask(context.observed_land))
        # Animal/crop support is autoregressive.  The parent receives the
        # immutable physical premises and these first-step masks; the native
        # policy remains authoritative for prefix-conditioned masks.
        animals = tuple(
            animal_target_support_mask(
                context, context.observed_land, species,
                context.placed_animals[:species])
            for species in range(3)
        )
        total_capacity = max(
            physical_crop_capacity(context, context.observed_land,
                                   context.placed_animals), 0)
        crops = tuple(
            crop_delta_support_mask(baseline, total_capacity)
            for baseline in physical_baseline)
        return {"land": land, "animals": animals, "crops": crops}

    def prepare_inference_context(
        self, obs: Mapping[str, Any],
        previous_execution: Mapping[str, int] | None = None, *,
        decision_key: Stage25DecisionKey | None = None,
        decision_id: str | None = None,
        behavior_identity: Stage25BehaviorIdentity | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> Stage25InferenceContext:
        """Build a read-only parent request before accepting a decision."""
        timer = time.perf_counter
        if profile is not None:
            phase_started = timer()
        if not isinstance(obs, Mapping):
            raise Stage25ProviderError("obs must be a mapping")
        if _terminal_observation(obs):
            raise Stage25TerminalError(
                "terminal delivery does not sample or apply")
        if self._pending_context is not None:
            raise Stage25ProviderError(
                "a Stage 2.5 inference request is already pending")
        day = _scalar_int(obs.get("day"), "obs.day")
        key = self._key(day, decision_key, decision_id)
        self._check_delivery(key)
        identity = behavior_identity or self.behavior_identity
        if not isinstance(identity, Stage25BehaviorIdentity):
            raise Stage25ProviderError(
                "external Stage 2.5 inference requires behavior_identity")
        curriculum = self.effective_curriculum()
        if profile is not None:
            profile["stage25_provider_prelude_seconds"] += (
                timer() - phase_started)
        inputs, context, initial = self._stage_observation(
            obs, previous_execution, profile=profile)
        frozen_inputs = self._freeze_inputs(inputs, profile)
        if profile is not None:
            phase_started = timer()
        support_payload = (None if self.validation_mode != "strict" else
                           self._support_payload(context, initial))
        if profile is not None:
            profile["stage25_provider_support_payload_seconds"] += (
                timer() - phase_started)
            phase_started = timer()
        row_token = stage25_row_token(
            f"episode={key.episode_id}/seat={key.seat}/day={key.day}"
            f"/behavior={identity.identity_id()}")
        if profile is not None:
            profile["stage25_provider_row_token_seconds"] += (
                timer() - phase_started)
            phase_started = timer()
        prepared = Stage25InferenceContext(
            decision_key=key,
            behavior_identity=identity,
            inputs=_FrozenMapping(tuple(frozen_inputs.items())),
            crop_capacity=tuple(initial),
            physical_context=context,
            daily_start=(day, float(obs["farms"][self.seat]["money"])),
            curriculum=curriculum,
            seed=self.seed,
            support=(None if support_payload is None else
                     _FrozenMapping(tuple(support_payload.items()))),
            row_token=row_token,
        )
        if profile is not None:
            profile["stage25_provider_context_object_build_seconds"] += (
                timer() - phase_started)
        self._pending_context = prepared
        return prepared

    def prepare_bootstrap_context(
        self, obs: Mapping[str, Any],
        previous_execution: Mapping[str, int] | None = None,
    ) -> Stage25InferenceContext:
        """Build a non-deliverable next-state context for truncation value only.

        This intentionally bypasses ``_check_delivery`` and uses a distinct
        row identity.  Callers may evaluate the critic at the final observed
        state, but cannot accidentally submit the resulting classes as the
        next manager decision or apply a crop delta twice.
        """
        if not isinstance(obs, Mapping) or _terminal_observation(obs):
            raise Stage25TerminalError("terminal state has no bootstrap context")
        day = _scalar_int(obs.get("day"), "obs.day")
        identity = self.behavior_identity
        if not isinstance(identity, Stage25BehaviorIdentity):
            raise Stage25ProviderError(
                "Stage 2.5 bootstrap requires an immutable behavior identity")
        self.effective_curriculum()
        # Bootstrap is a finalization seam, not a manager-boundary preparation,
        # so it deliberately does not contribute to the provider preparation
        # subphase timers (which partition `stage25_provider_prepare_seconds`).
        inputs, context, initial = self._stage_observation(obs, previous_execution)
        frozen_inputs = self._freeze_inputs(inputs, None)
        return Stage25InferenceContext(
            decision_key=Stage25DecisionKey(
                self.episode_id, self.seat, day, "bootstrap"),
            behavior_identity=identity,
            inputs=_FrozenMapping(tuple(frozen_inputs.items())),
            crop_capacity=tuple(initial),
            physical_context=context,
            daily_start=(day, float(obs["farms"][self.seat]["money"])),
            curriculum=self.effective_curriculum(),
            seed=self.seed,
            support=None,
        )

    def accept_inference_response(
        self, request: Stage25InferenceContext,
        action_classes: Sequence[int], *,
        behavior_identity: Stage25BehaviorIdentity | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> DailyPlan:
        """Validate response identity, then apply one board-relative delta."""
        if not isinstance(request, Stage25InferenceContext):
            raise TypeError("request must be Stage25InferenceContext")
        if request is not self._pending_context:
            raise Stage25ProviderError(
                "Stage 2.5 response does not match the pending request")
        expected = behavior_identity or self.behavior_identity
        if expected is None or request.behavior_identity != expected:
            raise Stage25ProviderError(
                "Stage 2.5 response behavior identity does not match request")
        if self.behavior_identity is not None and expected != self.behavior_identity:
            raise Stage25ProviderError(
                "Stage 2.5 response behavior identity does not match provider")
        key = self._key(
            request.decision_key.day, request.decision_key, None)
        self._check_delivery(key)
        curriculum = self.effective_curriculum()
        if request.curriculum != curriculum:
            raise Stage25ProviderError(
                "Stage 2.5 response curriculum does not match request")
        classes = _class_tuple(action_classes)
        goals = _validate_action(
            classes, request.crop_capacity, request.physical_context,
            curriculum)
        plan = _lower_plan(classes, goals)
        result = self._commit(
            key, classes, goals, plan, request.inputs,
            request.daily_start, profile=profile)
        self._pending_context = None
        return result

    def _commit(
        self, key: Stage25DecisionKey, classes: tuple[int, ...],
        goals: tuple[int, ...], plan: DailyPlan,
        inputs: Mapping[str, np.ndarray],
        daily_start: tuple[int, float], *,
        profile: Mapping[str, Any] | None = None,
    ) -> DailyPlan:
        # Compute every fallible value before mutating lifecycle fields so a
        # malformed observation (e.g. missing daily-start money) can never
        # leave a partially-applied decision.
        e_history = (int(daily_start[0]), float(daily_start[1]))
        if profile is not None:
            copy_started = time.perf_counter()
            copied_bytes = 0
        last_inputs = {name: np.array(value, copy=True)
                       for name, value in inputs.items()}
        if profile is not None:
            copied_bytes = sum(
                int(value.nbytes) for value in last_inputs.values())
            profile["stage25_provider_accept_input_copy_seconds"] += (
                time.perf_counter() - copy_started)
            profile["stage25_provider_accept_copy_bytes"] += copied_bytes
        curriculum = self.effective_curriculum()
        diagnostics = {
            "decision_identity": key.identity,
            "requested_classes": classes,
            "physical_crop_baseline": physical_baseline,
            "requested_crop_goals": goals,
            "plan": plan.to_json_dict(),
            "curriculum_version": curriculum.version,
            "e_history_version": self.e_history_version,
        }
        # All validation and policy work has completed before this point.
        self._physical_crop_baseline = physical_baseline
        self._accepted_key = key
        self._accepted_classes = classes
        self._crop_goals = goals
        self._cached_plan = plan
        self._e_history = e_history
        self._last_inputs = last_inputs
        self._diagnostics = diagnostics
        return plan

    def accept_classes(
        self, obs: Mapping[str, Any], action_classes: Sequence[int],
        previous_execution: Mapping[str, int] | None = None, *,
        decision_key: Stage25DecisionKey | None = None,
        decision_id: str | None = None, terminal: bool = False,
        expected_behavior_identity: Stage25BehaviorIdentity | None = None,
    ) -> DailyPlan:
        """Accept one external nine-class decision and lower it once."""
        if terminal or (isinstance(obs, Mapping) and _terminal_observation(obs)):
            raise Stage25TerminalError("terminal delivery does not sample or apply")
        if self._pending_context is not None:
            raise Stage25ProviderError(
                "pending inference request must be accepted through its context")
        if not isinstance(obs, Mapping):
            raise Stage25ProviderError("obs must be a mapping")
        if (expected_behavior_identity is not None
                and self.behavior_identity is not None
                and expected_behavior_identity != self.behavior_identity):
            raise Stage25ProviderError(
                "Stage 2.5 response behavior identity does not match provider")
        day = _scalar_int(obs.get("day"), "obs.day")
        key = self._key(day, decision_key, decision_id)
        self._check_delivery(key)
        classes = _class_tuple(action_classes)
        curriculum = self.effective_curriculum()
        inputs, context, initial = self._stage_observation(obs, previous_execution)
        goals = _validate_action(classes, initial, context, curriculum)
        plan = _lower_plan(classes, goals)
        return self._commit(
            key, classes, goals, plan, inputs,
            (day, float(obs["farms"][self.seat]["money"])))

    submit_classes = accept_classes
    accept_decision = accept_classes

    def daily_plan(
        self, obs: Mapping[str, Any], seat: int,
        previous_execution: Mapping[str, int] | None = None, *,
        action_classes: Sequence[int] | None = None,
        classes: Sequence[int] | None = None,
        decision_key: Stage25DecisionKey | None = None,
        decision_id: str | None = None, terminal: bool = False,
    ) -> DailyPlan:
        """Return the cached plan or resolve one new manager boundary."""
        if seat != self.seat:
            raise Stage25ProviderError(
                f"seat {seat} does not match configured seat {self.seat}")
        if terminal or (isinstance(obs, Mapping) and _terminal_observation(obs)):
            raise Stage25TerminalError("terminal delivery does not sample or apply")
        if action_classes is not None and classes is not None:
            raise Stage25ProviderError("pass action_classes or classes, not both")
        supplied = action_classes if action_classes is not None else classes
        if supplied is not None:
            return self.accept_classes(
                obs, supplied, previous_execution,
                decision_key=decision_key, decision_id=decision_id)
        day = _scalar_int(obs.get("day"), "obs.day")
        if day < self.manager_start_day:
            # The executor still asks for a daily plan during opening days.
            # Supply a neutral, non-deliverable plan without creating a
            # manager trajectory row or lifecycle state; the first real boundary is
            # accepted by the runner at manager_start_day.
            return _lower_plan(
                (0, 0, 0, 0, 100, 100, 100, 100, 100),
                (0, 0, 0, 0, 0))
        if self._cached_plan is not None and self.last_accepted_day == day:
            # A read with no action identity is benign.  If a caller supplies
            # one, it is a duplicate submission check, not a cache read.
            if decision_key is not None or decision_id is not None:
                key = self._key(day, decision_key, decision_id)
                if key != self._accepted_key:
                    raise Stage25DuplicateDecisionError(
                        f"decision {key.identity} conflicts with the accepted "
                        f"decision {self._accepted_key.identity}")
            return self._cached_plan
        if self._native_policy is None:
            raise Stage25ProviderError(
                "external provider requires accept_classes before daily_plan")
        key = self._key(day, decision_key, decision_id)
        self._check_delivery(key)
        # Bind the effective curriculum before sampling so a checkpoint/config
        # mismatch fails before inference or state mutation.
        curriculum = self.effective_curriculum()
        inputs, context, initial = self._stage_observation(obs, previous_execution)
        native = self._native_policy
        if hasattr(native, "act"):
            sampled = native.act(
                inputs, context, row_id=key.identity, mode=self.mode, seed=self.seed)
        elif callable(native):
            sampled = native(inputs, context, key, self.mode, self.seed)
        else:
            raise TypeError("native_policy must provide act() or be callable")
        classes_tuple = _class_tuple(sampled)
        goals = _validate_action(classes_tuple, initial, context, curriculum)
        plan = _lower_plan(classes_tuple, goals)
        return self._commit(
            key, classes_tuple, goals, plan, inputs,
            (day, float(obs["farms"][self.seat]["money"])))

    def export_state(self) -> dict[str, Any]:
        """Export lifecycle state only; executor queues are deliberately absent."""
        return self.state.to_json_dict()

    def export_json(self) -> str:
        return json.dumps(self.export_state(), sort_keys=True, separators=(",", ":"))

    def import_state(self, value: Mapping[str, Any] | str) -> None:
        """Restore a strict snapshot atomically after full validation."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise Stage25ProviderError("provider state is not valid JSON") from exc
        snapshot = Stage25LifecycleState.from_json_dict(value)
        if (snapshot.episode_id != self.episode_id or snapshot.seat != self.seat
                or snapshot.manager_start_day != self.manager_start_day):
            raise Stage25ProviderError(
                "provider state identity/configuration does not match owner")
        if snapshot.e_history_version != self.e_history_version:
            raise Stage25ProviderError("only corrected-E operating history is supported")
        if snapshot.curriculum is None \
                or snapshot.curriculum != self.effective_curriculum():
            raise Stage25ProviderError(
                "provider state curriculum does not match the effective curriculum")
        # Validate cross-field invariants before mutating this provider.
        if (snapshot.accepted_decision is None) != (snapshot.cached_plan is None):
            raise Stage25ProviderError("accepted decision and cached plan must agree")
        if snapshot.accepted_decision is None and any(
                item is not None for item in (
                    snapshot.physical_crop_baseline, snapshot.accepted_classes,
                    snapshot.crop_goals, snapshot.e_history)):
            raise Stage25ProviderError("empty provider state contains lifecycle data")
        if snapshot.accepted_decision is not None:
            if snapshot.accepted_decision.episode_id != self.episode_id \
                    or snapshot.accepted_decision.seat != self.seat:
                raise Stage25ProviderError("accepted decision identity mismatch")
            if (snapshot.physical_crop_baseline is None
                    or snapshot.accepted_classes is None
                    or snapshot.crop_goals is None):
                raise Stage25ProviderError("accepted state is incomplete")
        self._physical_crop_baseline = snapshot.physical_crop_baseline
        self._accepted_key = snapshot.accepted_decision
        self._accepted_classes = snapshot.accepted_classes
        self._crop_goals = snapshot.crop_goals
        self._cached_plan = snapshot.cached_plan
        self._e_history = snapshot.e_history
        self.source_history_version = snapshot.source_history_version
        self._last_inputs = None
        self._diagnostics = {}
        self._pending_context = None


__all__ = [
    "STATE_VERSION", "Stage25ProviderError", "Stage25DuplicateDecisionError",
    "Stage25OutOfOrderError", "Stage25TerminalError", "Stage25DecisionKey",
    "Stage25LifecycleState", "Stage25NativePolicy", "Stage25PlanProvider",
]
