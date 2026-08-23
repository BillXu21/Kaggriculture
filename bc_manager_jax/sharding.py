"""Replicated data-parallel helpers for the JAX manager (single host).

Design: one logical global batch is sharded along axis 0 across a
`Mesh(devices, ('data',))` axis; parameters and optimizer state stay fully
replicated (`PartitionSpec()`). The same compiled forward/loss/train code
runs unchanged for 1 or N devices: all loss reductions are full-array
`jnp.mean`s over the global batch, so GSPMD inserts the correct
cross-replica reductions. Because the global batch must be divisible by the
device count, every shard is equal-sized and mean-of-shards is exactly the
global mean.

Multi-host (process_count != 1) is explicitly unsupported and rejected.
"""

from __future__ import annotations

from typing import Mapping

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

DATA_AXIS = "data"
REPLICATED = PartitionSpec()
BATCH_LEADING = PartitionSpec(DATA_AXIS)


def create_data_mesh(device_count: int | None = None) -> Mesh:
    """Create a ('data',) mesh over the first `device_count` local devices."""
    if jax.process_count() != 1:
        raise ValueError(
            f"multi-host runs are unsupported (process_count="
            f"{jax.process_count()}); use a single-process addressable "
            f"device mesh")
    devices = jax.devices()
    n = len(devices) if device_count is None else int(device_count)
    if n < 1:
        raise ValueError(f"device_count must be >= 1, got {n}")
    if n > len(devices):
        raise ValueError(
            f"requested device_count={n} but only {len(devices)} addressable "
            f"devices are visible ({[str(d) for d in devices]})")
    return Mesh(np.array(devices[:n]), (DATA_AXIS,))


def check_global_batch(global_batch: int, device_count: int) -> None:
    if global_batch < 1:
        raise ValueError(f"global_batch must be >= 1, got {global_batch}")
    if global_batch % device_count != 0:
        raise ValueError(
            f"global batch {global_batch} is not divisible by device count "
            f"{device_count}; every device must receive an equal shard")


def _batch_sharding(ndim: int, mesh: Mesh) -> NamedSharding:
    if ndim < 1:
        return NamedSharding(mesh, REPLICATED)
    return NamedSharding(
        mesh, PartitionSpec(DATA_AXIS, *((None,) * (ndim - 1))))


def shard_batch(batch: Mapping[str, jax.Array], mesh: Mesh) -> dict:
    """Physically place each array sharded along axis 0 ('data')."""
    return {
        key: jax.device_put(value, _batch_sharding(np.asarray(value).ndim,
                                                   mesh))
        for key, value in batch.items()
    }


def replicate_tree(tree, mesh: Mesh):
    """Physically replicate every leaf (`PartitionSpec()`)."""
    return jax.tree_util.tree_map(
        lambda leaf: jax.device_put(leaf, NamedSharding(mesh, REPLICATED)),
        tree)


def describe_sharding(tree) -> dict[str, str]:
    """Human-readable per-leaf sharding description (tests/CLI metadata)."""
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]

    def key_of(tokens) -> str:
        parts = [str(getattr(entry, "key", None)
                     if getattr(entry, "key", None) is not None
                     else entry.idx) for entry in tokens]
        return "/".join(parts)

    described = {}
    for tokens, leaf in flat:
        sharding = getattr(leaf, "sharding", None)
        spec = getattr(sharding, "spec", None)
        described[key_of(tokens)] = str(spec) if spec is not None \
            else type(sharding).__name__ if sharding is not None \
            else "unsharded"
    return described
