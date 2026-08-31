"""Framework-neutral Stage-A types and protocols (issue #9, architecture req. 1).

Nothing here imports JAX or torch: the trajectory/runner seams must stay
usable with fake policies and fake engines in tiny deterministic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from bc_manager.economics import normalize_e_history_version

# Compositions (issue #9 A3). Seat 0's policy is listed first.
E_VS_E = "e_vs_e"
E_VS_PASS = "e_vs_pass"
CANDIDATE_VS_FROZEN = "candidate_vs_frozen"
FROZEN_VS_CANDIDATE = "frozen_vs_candidate"
CURRENT_VS_CURRENT_ECONOMIC = "current_vs_current_economic"
COMPOSITIONS = (E_VS_E, E_VS_PASS, CANDIDATE_VS_FROZEN,
                 FROZEN_VS_CANDIDATE, CURRENT_VS_CURRENT_ECONOMIC)


@dataclass(frozen=True)
class PolicyIdentity:
    """Immutable opponent/policy snapshot identity.

    `fingerprint` is a parameter digest (or checkpoint digest) — never a
    committed artifact. The optional E history version is part of the
    identity, so corrected and legacy policies cannot silently alias.
    """

    name: str
    version: str
    fingerprint: str
    e_history_version: str | None = None

    def __post_init__(self) -> None:
        if self.e_history_version is not None:
            object.__setattr__(
                self, "e_history_version",
                normalize_e_history_version(self.e_history_version))

    def identity_id(self) -> str:
        base = f"{self.name}@{self.version}:{self.fingerprint[:12]}"
        return (base if self.e_history_version is None else
                f"{base}:e-history={self.e_history_version}")

    def to_json_dict(self) -> dict[str, str]:
        result = {
            "name": self.name,
            "version": self.version,
            "fingerprint": self.fingerprint,
            "identity_id": self.identity_id(),
        }
        if self.e_history_version is not None:
            result["e_history_version"] = self.e_history_version
        return result


@dataclass(frozen=True)
class PolicyOutputs:
    """Batched policy outputs for one policy group/day.

    `action_tensors` holds the exact encoded action components (see
    `rl_manager.decode.ACTION_TENSOR_SHAPES`). The logprob group slots and the
    scalar value slot exist even for deterministic baselines (zeros there) so
    the schema is PPO-ready without a later migration.
    """

    action_tensors: Mapping[str, np.ndarray]
    logprob_groups: Mapping[str, np.ndarray]
    logprob_total: np.ndarray
    value: np.ndarray
    batch_size: int


@runtime_checkable
class BatchedPlanPolicy(Protocol):
    """One policy forward per contiguous request batch — never per env.

    Implementations receive already-stacked model-facing input arrays plus an
    explicit PRNG/seed identifier and return batched actions with PPO-ready
    logprob/value slots and their immutable identity.
    """

    identity: PolicyIdentity

    def plan_batch(
        self,
        inputs: Mapping[str, np.ndarray],
        prng_id: str,
    ) -> PolicyOutputs: ...


def seat_policies(
    composition: str,
    candidate: BatchedPlanPolicy,
    frozen: BatchedPlanPolicy,
    *,
    controlled_seat: int = 0,
) -> tuple[BatchedPlanPolicy, BatchedPlanPolicy]:
    """Resolve a composition name into (seat0_policy, seat1_policy)."""
    if composition == E_VS_E:
        if candidate.identity != frozen.identity:
            raise ValueError(
                "e_vs_e composition requires identical policies at both seats")
        return candidate, frozen
    if composition == E_VS_PASS:
        if controlled_seat not in (0, 1):
            raise ValueError(
                f"controlled_seat must be 0 or 1, got {controlled_seat!r}")
        return ((candidate, frozen) if controlled_seat == 0
                else (frozen, candidate))
    if composition == CANDIDATE_VS_FROZEN:
        return candidate, frozen
    if composition == FROZEN_VS_CANDIDATE:
        return frozen, candidate
    if composition == CURRENT_VS_CURRENT_ECONOMIC:
        return candidate, candidate
    raise ValueError(
        f"unknown composition {composition!r}; expected one of "
        f"{list(COMPOSITIONS)}")
