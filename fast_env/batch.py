"""Native batched fast environment with reusable NumPy buffers."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ._kaggriculture_env import ACTION_SLOTS, OBS_SIZE, RustBatchEnv
from .api import (
    DEFAULT_CONFIGURATION,
    MARKET_ACTION_START,
    _as_int,
    _decode_observation_pair,
    _market_row,
    _unit_row,
)


class BatchedFastEnv:
    """One native engine owning a fixed set of simultaneous two-seat games.

    Returned observations use the canonical farm tile vocabulary expected by
    the executor. Public farm/market/town objects are shared read-only between
    seat views; private dictionaries are decoded from only that seat's native
    row. Callers that mutate observations must copy their seat view first.
    """

    def __init__(
        self,
        num_envs: int,
        configuration: Mapping[str, Any] | None = None,
        *,
        canonical_observations: bool = False,
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.canonical_observations = bool(canonical_observations)
        self.configuration = dict(DEFAULT_CONFIGURATION)
        if configuration:
            self.configuration.update(configuration)
        if int(self.configuration["boardSize"]) != 10:
            raise ValueError("fast engine supports boardSize=10 only")
        if int(self.configuration["maxMarketOrdersPerTurn"]) != 10:
            raise ValueError("fast engine supports maxMarketOrdersPerTurn=10 only")
        raw_num_threads = self.configuration.get("numThreads")
        num_threads = None if raw_num_threads is None else _as_int(
            raw_num_threads, "numThreads"
        )
        if num_threads is not None and num_threads < 1:
            raise ValueError("numThreads must be a positive integer")
        self._backend = RustBatchEnv(
            self.num_envs,
            int(self.configuration["episodeSteps"]),
            int(self.configuration["turnsPerDay"]),
            float(self.configuration["weedSpawnChance"]),
            int(self.configuration["townCenterSellInterval"]),
            int(self.configuration["townShopSellInterval"]),
            int(self.configuration["townShopUnlockInterval"]),
            float(self.configuration["startingMoney"]),
            10,
            int(self.configuration["shedCapacity"]),
            json.dumps(self.configuration.get("marketParams", {}), sort_keys=True),
            int(self.configuration["farmHandCostMult"]),
            "",
            num_threads,
        )
        self.action_buffer = np.zeros(
            (self.num_envs, 2, ACTION_SLOTS, 3), dtype=np.int64
        )
        self.observation_buffer = np.zeros(
            (self.num_envs, 2, OBS_SIZE), dtype=np.float32
        )
        self.reward_buffer = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.status_buffer = np.zeros((self.num_envs, 2), dtype=np.uint8)
        self._observations: list[list[dict[str, Any]]] = []
        self.last_timing_seconds = {
            "action_encode": 0.0,
            "native_step": 0.0,
            "observation_decode": 0.0,
        }

    def _decode(self) -> list[list[dict[str, Any]]]:
        self._observations = [
            _decode_observation_pair(
                self.observation_buffer[index],
                self.configuration,
                canonical_farms=self.canonical_observations,
            )
            for index in range(self.num_envs)
        ]
        return self._observations

    def reset(self, seeds: Sequence[int]) -> list[list[dict[str, Any]]]:
        seed_values = [int(seed) for seed in seeds]
        if len(seed_values) != self.num_envs:
            raise ValueError(f"seeds must have shape ({self.num_envs},)")
        if any(seed < 0 or seed >= 2**64 for seed in seed_values):
            raise ValueError("seeds must be unsigned 64-bit integers")
        seed_array = np.asarray(seed_values, dtype=np.uint64)
        observations, statuses = self._backend.reset(seed_array)
        np.copyto(self.observation_buffer, observations)
        np.copyto(self.status_buffer, statuses)
        self.reward_buffer.fill(0.0)
        return self._decode()

    def encode_actions_into(
        self,
        action_batch: Sequence[Sequence[Mapping[str, Any]]],
    ) -> np.ndarray:
        if len(action_batch) != self.num_envs:
            raise ValueError(
                f"action batch must contain {self.num_envs} environments"
            )
        encoded = self.action_buffer
        encoded.fill(0)
        for environment, actions in enumerate(action_batch):
            if len(actions) != 2:
                raise ValueError(
                    f"actions[{environment}] must contain exactly two seats"
                )
            for player, raw_action in enumerate(actions):
                action: Mapping[str, Any] = (
                    raw_action if isinstance(raw_action, Mapping) else {}
                )
                encoded[environment, player, 0] = _unit_row(
                    action.get("farmer", ["PASS"])
                )
                for index, hand in enumerate(
                    action.get("hands", [])[: MARKET_ACTION_START - 1], start=1
                ):
                    encoded[environment, player, index] = _unit_row(hand)
                for index, order in enumerate(action.get("market", [])[:10]):
                    encoded[environment, player, MARKET_ACTION_START + index] = (
                        _market_row(order)
                    )
        return encoded

    def step(
        self,
        action_batch: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[list[list[dict[str, Any]]], np.ndarray, np.ndarray]:
        started = time.perf_counter()
        self.encode_actions_into(action_batch)
        encoded = time.perf_counter()
        self._backend.step_into(
            self.action_buffer,
            self.observation_buffer,
            self.reward_buffer,
            self.status_buffer,
        )
        stepped = time.perf_counter()
        observations = self._decode()
        decoded = time.perf_counter()
        self.last_timing_seconds = {
            "action_encode": encoded - started,
            "native_step": stepped - encoded,
            "observation_decode": decoded - stepped,
        }
        return observations, self.reward_buffer, self.status_buffer

    def observations(self, index: int) -> list[dict[str, Any]]:
        if not self._observations:
            raise RuntimeError("batch environment must be reset before observation")
        return self._observations[index]

    def rewards(self, index: int) -> list[float]:
        return [float(value) for value in self.reward_buffer[index]]

    def statuses(self, index: int) -> list[str]:
        return [
            "DONE" if bool(value) else "ACTIVE"
            for value in self.status_buffer[index]
        ]
