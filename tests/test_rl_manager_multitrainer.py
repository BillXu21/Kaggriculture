"""CPU-only identity and configuration tests for issue #21."""

from __future__ import annotations

import numpy as np
import pytest

from rl_manager.multitrainer import TrainerIdentityRouter
from rl_manager.multitrainer_benchmark import BenchmarkOptions, build_parser, \
    options_from_args
from rl_manager.parallel_protocol import InferenceRequest, policy_row_request_id
from rl_manager.trajectory import TransitionMetadata
from rl_manager.types import PolicyIdentity


def _identity(name: str) -> PolicyIdentity:
    return PolicyIdentity(name, "issue-21", name * 64)


def _request(identity: PolicyIdentity, episode: int) -> InferenceRequest:
    return InferenceRequest(
        request_id=policy_row_request_id(episode, 0, 4, identity),
        worker_id=0, episode_index=episode, seat=0, day=4,
        policy_identity=identity, prng_id="test",
        inputs={"day": np.asarray([[4]], dtype=np.int16)}, queued_at=0.0)


def _metadata(index: int, identity: PolicyIdentity, trainable: bool = True):
    return TransitionMetadata(
        index=index, episode_index=index, seed=index, seat=0, day=4,
        policy_id=identity.identity_id(), policy_version=identity.version,
        policy_fingerprint=identity.fingerprint, opponent_id="opponent",
        trainable=trainable, plan_json={})


def test_router_groups_requests_by_exact_identity_and_sorting():
    first = _identity("first")
    second = _identity("second")
    router = TrainerIdentityRouter({first: "trainer-1", second: "trainer-2"})
    groups = router.group_requests([_request(second, 2), _request(first, 3),
                                   _request(first, 1)])
    assert [(identity, trainer, [request.episode_index for request in requests])
            for identity, trainer, requests in groups] == [
                (first, "trainer-1", [1, 3]),
                (second, "trainer-2", [2]),
            ]


def test_router_rejects_unknown_and_duplicate_requests():
    known = _identity("known")
    router = TrainerIdentityRouter({known: object()})
    request = _request(known, 1)
    with pytest.raises(ValueError, match="duplicate"):
        router.group_requests([request, request])
    with pytest.raises(ValueError, match="no trainer"):
        router.group_requests([_request(_identity("unknown"), 1)])


def test_router_partitions_only_matching_trainable_rows():
    first = _identity("first")
    second = _identity("second")
    router = TrainerIdentityRouter({first: object(), second: object()})
    arrays = {
        "trainable": np.asarray([1, 0, 1], dtype=np.uint8),
        "value": np.asarray([10, 20, 30], dtype=np.float32),
    }
    metadata = [_metadata(0, first), _metadata(1, second, False),
                _metadata(2, second)]
    partitioned = router.partition_trainable_rows(arrays, metadata)
    np.testing.assert_array_equal(partitioned[first]["value"], [10])
    np.testing.assert_array_equal(partitioned[second]["value"], [30])


def test_router_rejects_metadata_identity_or_flag_drift():
    identity = _identity("first")
    router = TrainerIdentityRouter({identity: object()})
    arrays = {"trainable": np.asarray([1], dtype=np.uint8)}
    with pytest.raises(ValueError, match="unknown policy"):
        router.partition_trainable_rows(
            arrays, [_metadata(0, _identity("other"))])
    with pytest.raises(ValueError, match="trainable flag"):
        router.partition_trainable_rows(
            arrays, [_metadata(0, identity, trainable=False)])


def test_benchmark_parser_preserves_one_process_trainer_counts():
    args = build_parser().parse_args([
        "--trainer-counts", "1,2,4", "--model-config", "tiny",
        "--batch-size", "8", "--warmup", "0", "--iterations", "1",
    ])
    options = options_from_args(args)
    assert options == BenchmarkOptions(
        trainer_counts=(1, 2, 4), model_config="tiny", variant="E",
        batch_size=8, warmup=0, iterations=1)
    assert "ONE Python process" in build_parser().format_help()
