"""JAX E batched plan policy (issue #9, architecture req. 1/2).

One `bc_manager_jax.forward(..., model_variant="E")` call per contiguous
request batch — the runner groups all (env, seat) requests of one day by
policy identity, so there is never a per-environment JAX call. The wrapper
consumes only own-board E observations (`include_opponent=False` encoding)
with explicit params/config and decodes deterministically exactly like
issue #8. Logprob/value slots are diagnostic zeros in Stage A but exist so
the trajectory schema is PPO-ready.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import jax
import numpy as np

from bc_manager_jax.model import (
    ECONOMIC_CONTEXT_KEY,
    ManagerConfig,
    OWN_INPUT_KEYS,
    forward,
    resolve_model_variant,
)

from rl_manager.decode import (
    LOGPROB_GROUPS,
    decode_outputs_to_action_tensors,
)
from rl_manager.types import PolicyIdentity, PolicyOutputs


def params_fingerprint(params: Mapping) -> str:
    """Stable sha256 over sorted leaf paths + raw bytes of a param pytree.

    Used as the opponent/policy snapshot identity; no checkpoint artifact is
    ever committed or required.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    digest = hashlib.sha256()
    for path, leaf in flat:
        # Path entries vary by node type: DictKey.key, SequenceKey.idx,
        # GetAttrKey.name. str() of each entry renders all of them stably.
        key = "/".join(str(entry) for entry in path)
        digest.update(key.encode("utf-8"))
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class JaxEPlanPolicy:
    """Deterministic batched E policy over explicit `bc_manager_jax` params."""

    def __init__(
        self,
        params: Mapping,
        config: ManagerConfig,
        *,
        name: str = "jax_e",
        version: str = "stage-a-v1",
        model_variant: str = "E",
    ) -> None:
        self._variant = resolve_model_variant(model_variant)
        if self._variant != "E":
            raise ValueError(
                "JaxEPlanPolicy is the own-only E contract; got variant "
                f"{self._variant!r}")
        self._params = params
        self._config = config
        self.identity = PolicyIdentity(
            name=name,
            version=version,
            fingerprint=params_fingerprint(params),
        )
        # Batching proof instrumentation (tests assert on these).
        self.call_count = 0
        self.batch_size_history: list[int] = []

    def plan_batch(
        self,
        inputs: Mapping[str, np.ndarray],
        prng_id: str,
    ) -> PolicyOutputs:
        if not isinstance(prng_id, str) or not prng_id:
            raise ValueError("prng_id must be a non-empty string identifier")
        # Own-only E contract is enforced HERE at the wrapper seam: the
        # issue-#8 `validate_inputs` deliberately accepts opponent arrays as
        # ignored keys for torch-parity, but the RL policy must reject any
        # leaked opponent-public array or metadata key loudly.
        unknown = sorted(set(inputs.keys()) - set(OWN_INPUT_KEYS)
                         - {ECONOMIC_CONTEXT_KEY})
        if unknown:
            raise ValueError(
                f"own-only E contract violated: unknown/leaked input keys "
                f"{unknown}; opponent-public arrays and metadata must never "
                f"reach the E policy")
        outputs = forward(self._params, inputs, self._config,
                          model_variant=self._variant)
        action_tensors = decode_outputs_to_action_tensors(outputs)
        batch_size = int(action_tensors["crop"].shape[0])
        zeros = np.zeros(batch_size, dtype=np.float32)
        self.call_count += 1
        self.batch_size_history.append(batch_size)
        return PolicyOutputs(
            action_tensors=action_tensors,
            logprob_groups={group: zeros.copy() for group in LOGPROB_GROUPS},
            logprob_total=zeros.copy(),
            value=zeros.copy(),
            batch_size=batch_size,
        )
