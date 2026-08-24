"""Official-vs-fast parity seam with first-divergence reports (issue #9 A4).

Runs the identical composition/seed stack on two backends and compares, in
order: opening handoff diagnostics, per-(seat, day) manager input digests,
decoded plans, every primitive joint action, and final banks/statuses. The
first divergence carries episode seed / seat / day / hour / turn / field
path / both values / both actions where practical.

If the official engine dependency (`kaggle_environments`) is unavailable in
the current interpreter, `official_backend_available()` reports False and the
gated test path must skip with an explicit blocker and reproduction command —
never contort the simulator or install anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oracle.canonical import deep_diff

from rl_manager.runner import EpisodeResult, RunnerConfig, SelfPlayRunner


def official_backend_available() -> bool:
    """True only if the official 1.32.7 dependency imports right here."""
    try:
        import kaggle_environments  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure = unavailable
        return False
    return True


OFFICIAL_BLOCKER_COMMAND = (
    "python -m pytest tests/test_rl_manager_parity.py -k official -v "
    "(requires kaggle_environments 1.32.7 in the current interpreter; "
    "do NOT install into this environment)")


@dataclass(frozen=True)
class DivergenceReport:
    """First parity mismatch with enough context to reproduce it."""

    seed: int
    backend_a: str
    backend_b: str
    phase: str  # opening_handoff | manager_inputs | plans | actions | banks
    seat: int | None
    day: int | None
    hour: int | None
    turn_index: int | None
    field_path: str
    value_a: Any
    value_b: Any
    action_a: Any = None
    action_b: Any = None

    def render(self) -> str:
        return (
            f"first rollout divergence phase={self.phase} "
            f"seed={self.seed} backends={self.backend_a}/{self.backend_b} "
            f"seat={self.seat} day={self.day} hour={self.hour} "
            f"turn_index={self.turn_index} path={self.field_path} "
            f"a={self.value_a!r} b={self.value_b!r} "
            f"action_a={self.action_a!r} action_b={self.action_b!r}")


@dataclass
class ParityComparison:
    matched: bool
    checks: int
    report: DivergenceReport | None = None


def compare_rollouts(a: EpisodeResult, b: EpisodeResult) -> ParityComparison:
    """Compare two recorded episode results; first divergence wins."""
    checks = 0

    def diff(phase: str, left: Any, right: Any, *, seat=None, day=None,
             hour=None, turn=None, path="", action_a=None, action_b=None) \
            -> DivergenceReport | None:
        nonlocal checks
        checks += 1
        diffs = deep_diff(left, right, path=path)
        if diffs:
            first = diffs[0]
            return DivergenceReport(
                seed=a.seed, backend_a=a.rollout.backend_name,
                backend_b=b.rollout.backend_name, phase=phase,
                seat=seat, day=day, hour=hour, turn_index=turn,
                field_path=first.path, value_a=first.official_value,
                value_b=first.fast_value, action_a=action_a,
                action_b=action_b)
        return None

    if a.rollout is None or b.rollout is None:
        raise ValueError(
            "compare_rollouts requires RunnerConfig(record_rollout=True)")
    if a.seed != b.seed or a.composition != b.composition:
        raise ValueError("paired comparison requires same seed/composition")

    for seat in range(2):
        report = diff(
            "opening_handoff",
            a.opening_diagnostics[seat], b.opening_diagnostics[seat],
            seat=seat, path=f"opening_diagnostics[{seat}]")
        if report:
            return ParityComparison(False, checks, report)

    keys_a = sorted(a.rollout.manager_input_digests)
    keys_b = sorted(b.rollout.manager_input_digests)
    for key in sorted(set(keys_a) | set(keys_b)):
        seat, day = key
        digest_a = a.rollout.manager_input_digests.get(key)
        digest_b = b.rollout.manager_input_digests.get(key)
        report = diff("manager_inputs", digest_a, digest_b, seat=seat,
                      day=day, hour=0, path=f"manager_inputs[{seat},{day}]")
        if report:
            return ParityComparison(False, checks, report)

    plan_keys = sorted(set(a.rollout.plans) | set(b.rollout.plans))
    for key in plan_keys:
        seat, day = key
        report = diff("plans", a.rollout.plans.get(key),
                      b.rollout.plans.get(key), seat=seat, day=day,
                      path=f"plans[{seat},{day}]")
        if report:
            return ParityComparison(False, checks, report)

    turns = min(len(a.rollout.joint_actions), len(b.rollout.joint_actions))
    for index in range(turns):
        turn_a = a.rollout.joint_actions[index]
        turn_b = b.rollout.joint_actions[index]
        day, hour = turn_a[1], turn_a[2]
        for seat in range(2):
            action_a = turn_a[3 + seat]
            action_b = turn_b[3 + seat]
            report = diff("actions", action_a, action_b, seat=seat, day=day,
                          hour=hour, turn=index,
                          path=f"actions[{index}][{seat}]",
                          action_a=action_a, action_b=action_b)
            if report:
                return ParityComparison(False, checks, report)
    if len(a.rollout.joint_actions) != len(b.rollout.joint_actions):
        return ParityComparison(False, checks + 1, DivergenceReport(
            seed=a.seed, backend_a=a.rollout.backend_name,
            backend_b=b.rollout.backend_name, phase="actions", seat=None,
            day=None, hour=None, turn_index=turns,
            field_path="joint_actions.length",
            value_a=len(a.rollout.joint_actions),
            value_b=len(b.rollout.joint_actions)))

    report = diff("banks", {"banks": a.final_banks, "statuses": a.statuses},
                  {"banks": b.final_banks, "statuses": b.statuses},
                  path="final")
    if report:
        return ParityComparison(False, checks + 1, report)
    return ParityComparison(True, checks + 1, None)


def run_backend_rollout(
    config: RunnerConfig,
    spec_builder,
    specs_args: list[tuple],
) -> list[EpisodeResult]:
    """Run one backend's share of a paired comparison (record_rollout=True)."""
    config = RunnerConfig(
        backend_name=config.backend_name,
        backend_configuration=config.backend_configuration,
        opening=config.opening,
        manager_start_day=config.manager_start_day,
        max_turns=config.max_turns,
        num_envs=config.num_envs,
        record_rollout=True,
    )
    runner = SelfPlayRunner(config)
    specs = [spec_builder(*args) for args in specs_args]
    return runner.run(specs)
