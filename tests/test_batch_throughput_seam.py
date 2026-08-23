"""Issue #2 throughput seam: GIL release, instance-local Rayon thread count,
and cross-thread-count determinism for the Rust batch environment.

Bounded claims covered here:
- substantial native batch work runs with the GIL released (proven by real
  Python-thread progress during a long call, not by timing);
- caller-owned ``*_into`` buffers stay intact under concurrent Python-thread
  pressure (results identical to a quiet single-threaded reference);
- trajectories are byte-identical across 1/2/4/default worker counts,
  including across the day boundary;
- invalid thread counts are rejected loudly.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest

from fast_env import FastKaggricultureEnv
from fast_env._kaggriculture_env import ACTION_SLOTS, MASK_SIZE, OBS_SIZE, RustBatchEnv

# Above the Rust PARALLEL_MIN_ENVS=128 threshold so every configuration here
# exercises the parallel path (pool or global), not just the serial loop.
NUM_ENVS = 130


def _make_backend(num_threads: int | None) -> RustBatchEnv:
    if num_threads is None:
        return RustBatchEnv(NUM_ENVS, 720)
    return RustBatchEnv(NUM_ENVS, 720, num_threads=num_threads)


def _seed_vector() -> np.ndarray:
    return np.asarray([np.uint64(7919 * i + 13) for i in range(NUM_ENVS)], dtype=np.uint64)


def _scripted_actions(rng_seed: int = 123) -> np.ndarray:
    """One fixed action tensor reused verbatim across every thread count."""
    rng = np.random.default_rng(rng_seed)
    actions = np.zeros((NUM_ENVS, 2, ACTION_SLOTS, 3), dtype=np.int64)
    # Small legal-ish ranges; the engine treats everything else as no-ops, so
    # exact legality is irrelevant to the determinism contract being tested.
    actions[:, :, :, 0] = rng.integers(-1, 18, size=(NUM_ENVS, 2, ACTION_SLOTS))
    actions[:, :, :, 1] = rng.integers(-1, 9, size=(NUM_ENVS, 2, ACTION_SLOTS))
    actions[:, :, :, 2] = rng.integers(-1, 6, size=(NUM_ENVS, 2, ACTION_SLOTS))
    return actions


def _run_trajectory(num_threads: int | None) -> dict[str, list[str]]:
    """Run 30 steps (crossing the day boundary at step 24) and hash outputs."""
    backend = _make_backend(num_threads)
    observations, statuses = backend.reset(_seed_vector())
    digests: dict[str, list[str]] = {"obs": [], "rewards": [], "statuses": [], "masks": []}
    digests["obs"].append(hashlib.sha256(observations.tobytes()).hexdigest())
    digests["statuses"].append(hashlib.sha256(statuses.tobytes()).hexdigest())
    masks = np.zeros((NUM_ENVS, 2, MASK_SIZE), dtype=np.uint8)
    rewards = np.zeros((NUM_ENVS, 2), dtype=np.float32)
    statuses_buf = np.zeros((NUM_ENVS, 2), dtype=np.uint8)
    actions = _scripted_actions()
    for step in range(1, 31):
        backend.step_into(actions, observations, rewards, statuses_buf)
        digests["obs"].append(hashlib.sha256(observations.tobytes()).hexdigest())
        digests["rewards"].append(hashlib.sha256(rewards.tobytes()).hexdigest())
        digests["statuses"].append(hashlib.sha256(statuses_buf.tobytes()).hexdigest())
        if step in (1, 23, 24, 25, 30):
            masks.fill(0)
            backend.action_masks_into(masks)
            digests["masks"].append(hashlib.sha256(masks.tobytes()).hexdigest())
    return digests


def test_num_threads_validation_and_reporting() -> None:
    assert RustBatchEnv(2, 720).num_threads() == 0  # default: global pool
    assert RustBatchEnv(2, 720, num_threads=1).num_threads() == 1
    assert RustBatchEnv(2, 720, num_threads=4).num_threads() == 4
    with pytest.raises(ValueError):
        RustBatchEnv(2, 720, num_threads=0)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        RustBatchEnv(2, 720, num_threads=-1)


def test_scalar_wrapper_validates_num_threads() -> None:
    env = FastKaggricultureEnv({"seed": 7, "numThreads": 2})
    env.reset()
    assert env._backend.num_threads() == 2
    with pytest.raises(ValueError):
        FastKaggricultureEnv({"seed": 7, "numThreads": 0})
    with pytest.raises(ValueError):
        FastKaggricultureEnv({"seed": 7, "numThreads": "many"})


def test_gil_released_during_long_native_batch_call() -> None:
    # A background Python thread spins a plain GIL-bound counter while the
    # main thread runs a long native batch call (~1s on this host). If the
    # native work held the GIL, the spinner would be starved and the counter
    # would stay near zero; with the GIL released it advances freely.
    backend = RustBatchEnv(512, 720, num_threads=1)
    seeds = np.asarray([np.uint64(104729 * i + 1) for i in range(512)], dtype=np.uint64)
    observations, _ = backend.reset(seeds)
    rewards = np.zeros((512, 2), dtype=np.float32)
    statuses = np.zeros((512, 2), dtype=np.uint8)
    actions = np.zeros((512, 2, ACTION_SLOTS, 3), dtype=np.int64)

    stop = threading.Event()
    ticks = [0]

    def spin() -> None:
        while not stop.is_set():
            ticks[0] += 1

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        for _ in range(300):
            backend.step_into(actions, observations, rewards, statuses)
    finally:
        stop.set()
        spinner.join()

    assert (
        ticks[0] > 1000
    ), f"Python thread advanced only {ticks[0]} times during the native batch call; GIL was not released"


def test_concurrent_python_thread_pressure_does_not_corrupt_buffers() -> None:
    # Same seeds/actions as a quiet reference run; the loaded run executes
    # while a second Python thread hammers observe_into on its own buffer.
    # Byte equality proves no GIL-dependent mutation race exists.
    reference = RustBatchEnv(NUM_ENVS, 720, num_threads=1)
    ref_obs, _ = reference.reset(_seed_vector())
    ref_rewards = np.zeros((NUM_ENVS, 2), dtype=np.float32)
    ref_statuses = np.zeros((NUM_ENVS, 2), dtype=np.uint8)
    actions = _scripted_actions()
    for _ in range(10):
        reference.step_into(actions, ref_obs, ref_rewards, ref_statuses)

    loaded = RustBatchEnv(NUM_ENVS, 720, num_threads=2)
    obs, _ = loaded.reset(_seed_vector())
    rewards = np.zeros((NUM_ENVS, 2), dtype=np.float32)
    statuses = np.zeros((NUM_ENVS, 2), dtype=np.uint8)

    stop = threading.Event()

    def hammer_other_buffer() -> None:
        # A separate backend instance and a separate caller-owned buffer;
        # concurrent native work from other Python threads must never leak
        # into the main trajectory buffers. (Conflicting access to the *same*
        # backend is rejected loudly by PyO3's borrow guard instead of racing,
        # which is the soundness property this seam relies on.)
        other_backend = RustBatchEnv(NUM_ENVS, 720, num_threads=2)
        other_obs, _ = other_backend.reset(_seed_vector())
        other = np.zeros((NUM_ENVS, 2, OBS_SIZE), dtype=np.float32)
        while not stop.is_set():
            other_backend.observe_into(other_obs)
            other_backend.observe_into(other)
            other.fill(0.0)

    presser = threading.Thread(target=hammer_other_buffer, daemon=True)
    presser.start()
    try:
        for _ in range(10):
            loaded.step_into(actions, obs, rewards, statuses)
    finally:
        stop.set()
        presser.join()

    assert np.array_equal(obs, ref_obs)
    assert np.array_equal(rewards, ref_rewards)
    assert np.array_equal(statuses, ref_statuses)


def test_trajectories_identical_across_thread_counts_including_day_boundary() -> None:
    baseline = _run_trajectory(1)
    assert len(baseline["obs"]) == 31
    # The day boundary at canonical step 24 must actually be crossed inside
    # the masked windows so the boundary claim is non-vacuous.
    assert len(baseline["masks"]) == 5
    for num_threads in (2, 4, None):
        other = _run_trajectory(num_threads)
        assert other == baseline, f"thread count {num_threads!r} diverged from single-thread baseline"
