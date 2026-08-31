"""Adapters for the repository's opening, manager, and executor stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from bc_manager.live import encode_live_inputs
from executor_v0.agent import AgentConfig
from executor_v0.foreman import ForemanConfig
from opening_book.agent import make_opening_agent
from replay_daily.constants import total_hire_cost
from rl_manager.decode import plans_from_action_tensors
from rl_manager.executor_factory import EXECUTOR_FACTORY_VERSION
from rl_manager.provider import QueuedPlanProvider


def _repository_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def serialize_agent_config(config: AgentConfig) -> dict[str, Any]:
    """Return the complete JSON-safe executor configuration."""
    return asdict(config)


class _PassController:
    observation_mode = "raw"

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        farm = observation.get("farms", [])[int(observation["player"])]
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in farm.get("hands", [])],
            "market": [],
        }

    __call__ = act

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class PassControllerFactory:
    """A legal plumbing controller for deterministic harness tests."""

    display_name: str = "PASS"

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "display_name": self.display_name,
            "kind": "pass",
            "execution_mode": "in_process",
            "identity": self.display_name,
        }

    def create(self, *, seat: int, configuration: Mapping[str, Any]) -> _PassController:
        del seat, configuration
        return _PassController()


class InternalController:
    """Stateful primitive view over one internal policy/executor stack."""

    observation_mode = "canonical"

    def __init__(
        self,
        *,
        seat: int,
        policy: Any,
        opening_name: str,
        executor_factory: Any,
        configuration: Mapping[str, Any],
    ) -> None:
        self.seat = seat
        self.policy = policy
        self.provider = QueuedPlanProvider()
        self.executor = executor_factory.create(
            backend_name=str(configuration.get("__backend_name", "fast")),
            seat=seat,
            configuration=configuration,
            provider=self.provider,
        )
        self.opening = make_opening_agent(
            opening_name, downstream=self.executor, seat=seat
        )
        self._last_day = -1
        self._current_start: tuple[int, float] | None = None
        self._previous_start: tuple[int, float] | None = None
        self._previous_execution = {"workers_hired": 0, "hire_cost": 0}
        self._hires_today = 0
        self._planned_days: set[int] = set()

    def _observe_day(self, observation: Mapping[str, Any]) -> None:
        day = int(observation["day"])
        farm = observation["farms"][self.seat]
        if day <= self._last_day:
            return
        if self._last_day >= 0:
            self._previous_execution = {
                "workers_hired": self._hires_today,
                "hire_cost": total_hire_cost(self._hires_today),
            }
            self._previous_start = self._current_start
        self._last_day = day
        self._hires_today = int(farm.get("hires_today", 0) or 0)
        self._current_start = (day, float(farm["money"]))

    def _queue_plan(self, observation: Mapping[str, Any]) -> None:
        day = int(observation["day"])
        if day < 4 or int(observation["hour"]) != 0 or day in self._planned_days:
            return
        inputs = encode_live_inputs(
            observation,
            self.seat,
            self._previous_execution,
            step=int(observation["step"]),
            economic_prev_start=self._previous_start,
        )
        outputs = self.policy.plan_batch(
            inputs, f"evaluation/seat={self.seat}/day={day}"
        )
        if int(outputs.batch_size) != 1:
            raise ValueError(
                f"internal policy returned batch {outputs.batch_size}, expected 1"
            )
        plans = plans_from_action_tensors(outputs.action_tensors)
        self.provider.queue(day, plans[0])
        self._planned_days.add(day)

    def act(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        self._observe_day(observation)
        self._queue_plan(observation)
        return self.opening(observation)

    __call__ = act

    def diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for owner, value in (("opening", self.opening), ("executor", self.executor)):
            function = getattr(value, "diagnostics_json", None)
            if callable(function):
                result[owner] = function()
        return result

    def close(self) -> None:
        return None


@dataclass
class InternalControllerFactory:
    """Fresh internal controller instances with an explicit seat config."""

    policy: Any
    executor_config: AgentConfig
    opening_name: str = "standard_mixed"
    display_name: str = "internal"
    checkpoint_path: str | None = None
    policy_source: Mapping[str, Any] | None = None
    executor_factory: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.executor_config, AgentConfig):
            raise TypeError("executor_config must be an AgentConfig")
        self._executor_factory = self.executor_factory

    @property
    def provenance(self) -> Mapping[str, Any]:
        identity = getattr(self.policy, "identity", None)
        policy_identity = identity.to_json_dict() if identity is not None else {}
        checkpoint_digest = None
        if self.checkpoint_path:
            checkpoint_digest = sha256_file(self.checkpoint_path)
        return {
            "display_name": self.display_name,
            "kind": "internal",
            "identity": f"{self.display_name}+{self.executor_config.aggressive_sell_all and 'aggressive_sell_all' or 'normal_executor'}",
            "policy": policy_identity,
            "policy_source": dict(self.policy_source or {}),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": checkpoint_digest,
            "opening": {
                "name": self.opening_name,
                "identity": _opening_identity(self.opening_name),
            },
            "executor_factory": {
                "name": getattr(self._factory(), "name", "executor_v0"),
                "version": getattr(
                    self._factory(), "version", EXECUTOR_FACTORY_VERSION
                ),
                "config": serialize_agent_config(self.executor_config),
            },
            "repository_commit": _repository_commit(),
            "execution_mode": "in_process",
        }

    def _factory(self) -> Any:
        if self._executor_factory is None:
            from rl_manager.executor_factory import make_default_executor_factory

            self._executor_factory = make_default_executor_factory(self.executor_config)
        return self._executor_factory

    def create(
        self,
        *,
        seat: int,
        configuration: Mapping[str, Any],
    ) -> InternalController:
        return InternalController(
            seat=seat,
            policy=self.policy,
            opening_name=self.opening_name,
            executor_factory=self._factory(),
            configuration=configuration,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opening_identity(name: str) -> str:
    from rl_manager.provenance import opening_provenance

    return str(opening_provenance(name)["digest"])


def make_agent_config(
    payload: Mapping[str, Any] | None = None,
    *,
    aggressive_sell_all: bool | None = None,
) -> AgentConfig:
    """Build an AgentConfig from a JSON object without silently dropping keys."""
    # Match the current RL evaluation factory rather than the lower-level
    # ExecutorAgent constructor's permissive unit-test defaults.
    values = {"strict": True, "optional_spare_watering": True}
    values.update(dict(payload or {}))
    if "foreman" in values and not isinstance(values["foreman"], ForemanConfig):
        foreman = dict(values["foreman"])
        if "shed_access_tiles" in foreman:
            foreman["shed_access_tiles"] = tuple(
                tuple(int(item) for item in tile)
                for tile in foreman["shed_access_tiles"]
            )
        values["foreman"] = ForemanConfig(**foreman)
    if aggressive_sell_all is not None:
        values["aggressive_sell_all"] = bool(aggressive_sell_all)
    allowed = {field.name for field in AgentConfig.__dataclass_fields__.values()}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown AgentConfig fields: {unknown}")
    return AgentConfig(**values)


def load_internal_factory(
    kind: str,
    checkpoint_path: str | Path,
    *,
    executor_config: AgentConfig,
    display_name: str | None = None,
    opening_name: str = "standard_mixed",
) -> InternalControllerFactory:
    """Load BC-E or a detached PPO snapshot through existing loaders."""
    path = Path(checkpoint_path)
    if kind == "bc":
        from bc_manager_jax.checkpoint import load_torch_checkpoint
        from bc_manager_jax.model import ManagerConfig
        from rl_manager.policy import JaxEPlanPolicy

        params, metadata = load_torch_checkpoint(path, model_variant="E")
        config = ManagerConfig(**metadata["model_config"])
        policy = JaxEPlanPolicy(
            params,
            config,
            name=display_name or "bc_e",
            version="bc-e-v1",
            model_variant="E",
        )
    elif kind in {"ppo", "ppo-snapshot", "snapshot"}:
        from rl_manager.ppo_checkpoint import load_ppo_snapshot

        policy, metadata = load_ppo_snapshot(path)
        if display_name:
            policy.identity = type(policy.identity)(
                name=display_name,
                version=policy.identity.version,
                fingerprint=policy.identity.fingerprint,
            )
    else:
        raise ValueError(f"unsupported internal controller kind {kind!r}")
    return InternalControllerFactory(
        policy=policy,
        executor_config=executor_config,
        opening_name=opening_name,
        display_name=display_name or getattr(policy.identity, "name", kind),
        checkpoint_path=str(path),
        policy_source={
            "kind": kind,
            "metadata": metadata,
        },
    )


__all__ = [
    "InternalController",
    "InternalControllerFactory",
    "PassControllerFactory",
    "load_internal_factory",
    "make_agent_config",
    "serialize_agent_config",
    "sha256_file",
]
