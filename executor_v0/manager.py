"""Daily manager wrapper: checkpoint-backed provider, injection seam, caching.

Issue #1 section 2. One small interface (`PlanProvider`), one real
implementation backed by a D-019 checkpoint via the existing
`bc_manager.training.load_model_from_checkpoint`, one fixed/fake
implementation for tests and executor work, and one once-per-day caching
wrapper. No framework.

Caching semantics (documented per issue #1 "hour 0 or once per new day"):

- The first `daily_plan` call for a newly observed day computes exactly once,
  regardless of the hour at which that first call happens (deterministic
  resilience when a new day is first seen after hour 0).
- Every later call on the same day returns the cached plan without invoking
  the underlying provider. The plan is never recomputed within a day.
"""

import math
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import torch

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER, PRODUCT_ORDER
from bc_manager.live import encode_live_inputs
from bc_manager.model import ManagerConfig, predict_counts, predict_land, \
    predict_sells
from bc_manager.training import load_model_from_checkpoint

from .plan import SELL_BIN_ANCHORS, DailyPlan

__all__ = [
    "PlanProvider",
    "FixedPlanProvider",
    "CheckpointPlanProvider",
    "CachingPlanProvider",
    "decode_daily_plan",
]

_SELL_PRESENCE_THRESHOLD = 0.5


@runtime_checkable
class PlanProvider(Protocol):
    """Minimal seam: anything producing one plan per observed day."""

    def daily_plan(
        self,
        obs: Mapping,
        seat: int,
        previous_execution: Mapping[str, int] | None = None,
    ) -> DailyPlan:
        ...


class FixedPlanProvider:
    """Fake/fixed manager for executor and wrapper tests; no model involved."""

    def __init__(self, plan: DailyPlan) -> None:
        if not isinstance(plan, DailyPlan):
            raise ValueError(
                f"FixedPlanProvider requires a DailyPlan, got "
                f"{type(plan).__name__}")
        self.plan = plan

    def daily_plan(self, obs, seat, previous_execution=None) -> DailyPlan:
        return self.plan


def decode_daily_plan(outputs: Mapping[str, torch.Tensor], *,
                      count_max: int) -> DailyPlan:
    """Decode structured model outputs into a validated `DailyPlan`.

    - crop/animal/fertilizer/CARE: argmax over the ``count_max + 1`` count
      classes via `predict_counts` (values are therefore in ``[0, count_max]``).
    - land: `predict_land` argmax + 1, i.e. 1..4.
    - sells: presence is sigmoid > 0.5 (strictly greater; a logit of exactly
      0 means absent); quantity is the `expm1` helper output converted with
      deterministic round-half-up (``floor(q + 0.5)``) and forced to 0 when
      presence is false.
    """
    crops = [int(v) for v in predict_counts(outputs["crop_logits"])[0].tolist()]
    animals = [int(v)
               for v in predict_counts(outputs["animal_logits"])[0].tolist()]
    land = int(predict_land(outputs["land_logits"])[0])
    fertilizer = [int(v)
                  for v in predict_counts(outputs["fertilizer_logits"])[0]
                  .tolist()]
    care = [int(v) for v in predict_counts(outputs["care_logits"])[0].tolist()]

    presence, quantity = predict_sells(outputs["sell_presence_logits"],
                                       outputs["sell_quantity_log1p"])
    presence_rows = presence[0] > _SELL_PRESENCE_THRESHOLD
    quantity_rows = quantity[0]
    sells: list[tuple[int, ...]] = []
    for presence_row, quantity_row in zip(presence_rows.tolist(),
                                          quantity_rows.tolist()):
        cells = []
        for present, raw in zip(presence_row, quantity_row):
            value = int(math.floor(float(raw) + 0.5)) if present else 0
            cells.append(max(0, value))
        sells.append(tuple(cells))

    return DailyPlan.create(
        crop_targets=dict(zip(CROP_ORDER, crops)),
        animal_targets=dict(zip(ANIMAL_ORDER, animals)),
        land_count=land,
        fertilizer_by_crop=dict(zip(CROP_ORDER, fertilizer)),
        care_by_animal=dict(zip(ANIMAL_ORDER, care)),
        sell_quantities={
            product: dict(zip(SELL_BIN_ANCHORS, row))
            for product, row in zip(PRODUCT_ORDER, sells)
        },
    )


class CheckpointPlanProvider:
    """Real D-019 manager loaded from an explicit checkpoint path/device."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu",
                 include_opponent_board: bool | None = None) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"BC manager checkpoint not found: {path}; supply the real "
                f"D-019 best.pt path instead of fabricating one")
        self.model, payload = load_model_from_checkpoint(path, device=device)
        self.model_config = ManagerConfig(**payload["model_config"])
        if include_opponent_board is None:
            include_opponent_board = self.model_config.include_opponent_board
        if include_opponent_board and not self.model_config.include_opponent_board:
            raise ValueError(
                "include_opponent_board=True is incompatible with a model "
                "trained without the opponent public board")
        self.include_opponent_board = bool(include_opponent_board)

    def daily_plan(self, obs, seat, previous_execution=None) -> DailyPlan:
        inputs = encode_live_inputs(
            obs, seat, previous_execution,
            include_opponent=self.include_opponent_board)
        batch = {
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(batch)
        return decode_daily_plan(outputs,
                                 count_max=self.model_config.count_max)


class CachingPlanProvider:
    """Once-per-day cache around any `PlanProvider` (see module docstring)."""

    def __init__(self, provider: PlanProvider) -> None:
        self._provider = provider
        self._cached_day: int | None = None
        self._cached_plan: DailyPlan | None = None

    def daily_plan(self, obs, seat, previous_execution=None) -> DailyPlan:
        day = obs["day"]
        if self._cached_plan is not None and self._cached_day == day:
            return self._cached_plan
        plan = self._provider.daily_plan(obs, seat, previous_execution)
        self._cached_day = day
        self._cached_plan = plan
        return plan
