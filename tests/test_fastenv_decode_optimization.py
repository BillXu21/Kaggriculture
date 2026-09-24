"""Parity and aliasing coverage for the optimized FastEnv decoder."""

from __future__ import annotations

import copy

import pytest

from fast_env import BatchedFastEnv, FastKaggricultureEnv
from fast_env._reference import decode_observation_pair
from fast_env.api import PRODUCTS, _inventory, _round


def _action(farmer: tuple[object, ...] = ("PASS",), market=()):
    return {
        "farmer": list(farmer),
        "hands": [],
        "market": [list(order) for order in market],
    }


def _pair(farmer=("PASS",), market=()):
    return [_action(farmer, market), _action(farmer, market)]


def _reachable_actions(turn: int):
    if turn == 0:
        return _pair(
            ("PASS",),
            (
                ["BUY_SEED", "WHEAT", 2],
                ["BUY_SEED", "CARROT", 2],
                ["BUY_ANIMAL", "GOOSE", 1],
                ["HIRE"],
            ),
        )
    return _pair(("PLANT", "WHEAT") if turn == 1 else ("PASS",))


@pytest.mark.parametrize("canonical_farms", [False, True])
def test_optimized_decoder_matches_reference_over_reachable_turns(
    canonical_farms: bool,
) -> None:
    batch = BatchedFastEnv(
        4,
        {"numThreads": 1},
        canonical_observations=canonical_farms,
    )
    batch.reset([7, 19, 42, 123])
    for environment in range(batch.num_envs):
        assert batch.observations(environment) == decode_observation_pair(
            batch.observation_buffer[environment],
            batch.configuration,
            canonical_farms=canonical_farms,
        )

    for turn in range(40):
        batch.step([_reachable_actions(turn) for _ in range(batch.num_envs)])
        for environment in range(batch.num_envs):
            assert batch.observations(environment) == decode_observation_pair(
                batch.observation_buffer[environment],
                batch.configuration,
                canonical_farms=canonical_farms,
            )


def test_scalar_noncanonical_decoder_matches_reference() -> None:
    environment = FastKaggricultureEnv(
        {"seed": 19, "numThreads": 1}, canonical_observations=False
    )
    batch = BatchedFastEnv(
        1, {"seed": 19, "numThreads": 1}, canonical_observations=False
    )
    environment.reset()
    batch.reset([19])
    for turn in range(20):
        actions = _reachable_actions(turn)
        scalar_observations, _, _ = environment.step(actions)
        batch.step([actions])
        expected = decode_observation_pair(
            batch.observation_buffer[0],
            environment.configuration,
            canonical_farms=False,
        )
        assert scalar_observations == expected
    assert environment.state_snapshot()[0]["farms"] is not environment.state_snapshot()[1]["farms"]


def test_populated_reachable_buffer_matches_reference() -> None:
    batch = BatchedFastEnv(
        1,
        {
            "numThreads": 1,
            "startingMoney": 100000,
            "weedSpawnChance": 0.0,
        },
        canonical_observations=True,
    )
    batch.reset([7])
    orders = [
        ["BUY_SEED", crop, 5]
        for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    ]
    orders.extend([["BUY_ANIMAL", "GOOSE", 1], ["HIRE"], ["HIRE"]])
    batch.step([_pair(("BUILD_COOP",), orders)])
    batch.step([_pair(("PICKUP", "GOOSE", 1))])
    batch.step([_pair(("PLACE", "GOOSE", 1))])
    batch.step([_pair(("WEST",))])
    batch.step([_pair(("BUILD_PASTURE",))])
    batch.step([_pair(("NORTH",))])
    batch.step([_pair(("PLANT", "WHEAT"))])
    batch.step([_pair(("WATER",))])
    expected = decode_observation_pair(
        batch.observation_buffer[0], batch.configuration, canonical_farms=True
    )
    assert batch.observations(0) == expected
    assert len(batch.observations(0)[0]["farms"][0]["hands"]) == 2
    assert any(
        isinstance(tile, dict) and tile.get("kind") == "PLANT"
        for row in batch.observations(0)[0]["farms"][0]["tiles"]
        for tile in row
    )


def test_inventory_rounding_matches_scalar_protocol_domain() -> None:
    import numpy as np

    raw = np.zeros(12, dtype=np.float32)
    for quantity in range(-100, 10001):
        raw[0] = np.float32(quantity / 100.0)
        decoded = _inventory(raw, 0)
        assert decoded[PRODUCTS[0]] == _round(raw[0] * 100.0)

    for value in (
        -0.015,
        -0.005,
        0.005,
        0.015,
        float(np.nextafter(np.float32(0.005), np.float32(1.0))),
        float(np.nextafter(np.float32(0.005), np.float32(-1.0))),
    ):
        raw[0] = np.float32(value)
        assert _inventory(raw, 0)[PRODUCTS[0]] == _round(
            raw[0] * 100.0
        )


def test_fixed_seed_rollout_matches_reference_decoder() -> None:
    import fast_env.batch as batch_module

    def run(decoder):
        batch = BatchedFastEnv(
            2, {"numThreads": 1}, canonical_observations=True
        )
        original = batch_module._decode_observation_pair
        batch_module._decode_observation_pair = decoder
        try:
            observations = batch.reset([7, 19])
            trace = [(
                copy.deepcopy(observations),
                batch.status_buffer.tolist(),
            )]
            for turn in range(60):
                actions = [_reachable_actions(turn) for _ in range(2)]
                observations, rewards, statuses = batch.step(actions)
                trace.append((
                    copy.deepcopy(observations),
                    rewards.tolist(),
                    statuses.tolist(),
                ))
            return trace
        finally:
            batch_module._decode_observation_pair = original

    reference_trace = run(decode_observation_pair)
    from fast_env.api import _decode_observation_pair

    optimized_trace = run(_decode_observation_pair)
    assert reference_trace == optimized_trace


def test_canonical_pair_aliasing_and_turn_environment_isolation() -> None:
    batch = BatchedFastEnv(2, {"numThreads": 1}, canonical_observations=True)
    observations = batch.reset([7, 19])
    assert observations[0][0]["farms"] is observations[0][1]["farms"]
    assert observations[0][0]["market"] is observations[0][1]["market"]
    assert observations[0][0]["private"] is not observations[0][1]["private"]
    assert observations[0][0]["farms"] is not observations[1][0]["farms"]
    assert observations[0][0]["market"] is not observations[1][0]["market"]
    assert observations[0][0]["private"] is not observations[1][0]["private"]

    previous = copy.deepcopy(observations[0])
    previous_identity = observations[0]
    batch.step([_pair(), _pair()])
    current = batch.observations(0)
    assert current is not previous_identity
    assert current[0]["farms"] is not previous_identity[0]["farms"]
    assert observations[0] == previous

    observations[0][0]["private"]["shed"]["WHEAT"] = 999
    observations[0][0]["farms"][0]["tiles"][0][0] = "mutated"
    assert observations[0][1]["private"]["shed"]["WHEAT"] != 999
    assert observations[1][0]["private"]["shed"]["WHEAT"] != 999
    assert observations[1][0]["farms"][0]["tiles"][0][0] != "mutated"
    assert current[0]["private"]["shed"]["WHEAT"] != 999
    assert current[0]["farms"][0]["tiles"][0][0] != "mutated"
