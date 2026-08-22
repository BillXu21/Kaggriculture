"""Focused tests for the daily-manager tile Transformer and its loss.

Covers: exact output shapes, finite backward gradients, parameter-count
behavior, token layout, nullable-lifecycle safety, metadata-leak rejection,
opponent-public mode, target validation, group-balanced sell loss masking,
adapter-owned >100 repeated sells without clipping, tiny-batch overfit,
interleaved batches, and one bounded real schema-v2 adapter batch smoke.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bc_manager.adapter import load_selected_table, table_to_arrays
from bc_manager.loss import GROUP_NAMES, ManagerLossConfig, manager_loss
from bc_manager.model import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SELL_BIN_COUNT,
    DailyManagerTransformer,
    ManagerConfig,
    TileEncoder,
    predict_counts,
    predict_land,
    predict_sells,
    tiny_manager_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SAMPLE = REPO_ROOT / "data" / "canonical" / "2026-08-20-sample.parquet"


# ------------------------------------------------------------- fixtures


def make_inputs(batch_size: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def ints(shape, low, high, dtype):
        return torch.randint(low, high, shape, generator=g).to(dtype)

    day = ints((batch_size,), 0, 30, torch.int16)
    return {
        "board_kind": ints((batch_size, 100), 0, 8, torch.int16),
        "board_crop": ints((batch_size, 100), 0, 7, torch.int8),
        "board_animal": ints((batch_size, 100), 0, 5, torch.int8),
        "board_numeric":
            (torch.randn((batch_size, 100, 11), generator=g) * 3)
            .to(torch.float32),
        "board_bool":
            torch.rand((batch_size, 100, 8), generator=g) > 0.5,
        "board_mask": ints((batch_size, 100, 4), 0, 2, torch.uint8),
        "scalars":
            (torch.rand((batch_size, 4), generator=g) * 1000 + 10)
            .to(torch.float32),
        "shed_counts": ints((batch_size, 12), 0, 10, torch.int32),
        "seed_counts": ints((batch_size, 5), 0, 10, torch.int32),
        "carried_counts": ints((batch_size, 12), 0, 10, torch.int32),
        "unlocked": ints((batch_size, 4), 0, 2, torch.uint8),
        "market_inventory": ints((batch_size, 9), 0, 20, torch.int32),
        "market_prices":
            (torch.rand((batch_size, 9), generator=g) * 50)
            .to(torch.float32),
        "shop_counts": ints((batch_size, 9), 0, 5, torch.int32),
        "day": day,
        "days_remaining": (29 - day.long()).to(torch.int16),
    }


def make_targets(batch_size: int = 2, seed: int = 0, count_max: int = 100,
                 positive_cells: int = 6) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def counts(shape):
        return torch.randint(0, count_max + 1, shape, generator=g) \
            .to(torch.int32)

    presence = torch.zeros((batch_size, NUM_PRODUCTS, SELL_BIN_COUNT),
                           dtype=torch.bool)
    flat = presence.view(-1)
    flat[:positive_cells] = True
    quantity = torch.log1p(
        (flat.to(torch.float32)
         * torch.randint(1, 60, flat.shape, generator=g).to(torch.float32)))
    return {
        "crop_target": counts((batch_size, NUM_CROPS)),
        "animal_target": counts((batch_size, NUM_ANIMALS)),
        "land_count": torch.randint(1, 5, (batch_size,),
                                    generator=g).to(torch.int32),
        "fertilizer_target": counts((batch_size, NUM_CROPS)),
        "care_target": counts((batch_size, NUM_ANIMALS)),
        "sell_presence": presence.reshape(presence.shape),
        "sell_quantity_log1p": quantity.reshape(presence.shape)
            .to(torch.float32),
    }


def loss_and_backward(model, inputs, targets):
    outputs = model(inputs)
    total, groups = manager_loss(outputs, targets)
    total.backward()
    return outputs, total.detach(), groups


# --------------------------------------------------------------- shapes


def test_tiny_forward_output_shapes():
    config = tiny_manager_config()
    model = DailyManagerTransformer(config)
    model.eval()
    outputs = model(make_inputs(batch_size=3))
    c = config.count_classes
    assert outputs["crop_logits"].shape == (3, NUM_CROPS, c)
    assert outputs["animal_logits"].shape == (3, 3, c)
    assert outputs["land_logits"].shape == (3, 4)
    assert outputs["fertilizer_logits"].shape == (3, NUM_CROPS, c)
    assert outputs["care_logits"].shape == (3, 3, c)
    assert outputs["sell_presence_logits"].shape == (3, NUM_PRODUCTS,
                                                     SELL_BIN_COUNT)
    assert outputs["sell_quantity_log1p"].shape == (3, NUM_PRODUCTS,
                                                    SELL_BIN_COUNT)


def test_backward_pass_finite_gradients():
    torch.manual_seed(7)
    model = DailyManagerTransformer(tiny_manager_config())
    inputs = make_inputs(batch_size=2, seed=1)
    targets = make_targets(batch_size=2, seed=2)
    _, total, groups = loss_and_backward(model, inputs, targets)
    assert set(groups) == set(GROUP_NAMES)
    assert torch.isfinite(total)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_config_changes_parameter_count_and_default_range(capsys):
    default = DailyManagerTransformer(ManagerConfig())
    tiny = DailyManagerTransformer(tiny_manager_config())
    wider = DailyManagerTransformer(tiny_manager_config(d_model=32))
    default_count = default.trainable_parameters
    print(f"default trainable parameters: {default_count}")
    print(f"tiny trainable parameters: {tiny.trainable_parameters}")
    print(f"default token count: {ManagerConfig().token_count}")
    assert 700_000 <= default_count <= 1_500_000
    assert tiny.trainable_parameters < default_count
    assert wider.trainable_parameters > tiny.trainable_parameters


def test_default_token_layout_is_manager_plus_100_tiles_plus_five_globals():
    config = ManagerConfig()
    assert config.token_count == 106  # 1 manager + 100 tiles + 5 globals
    opp_config = ManagerConfig(include_opponent_board=True)
    assert opp_config.token_count == 206

    captured = {}
    handle = None

    def hook(module, args, output):
        captured["seq"] = args[0].shape

    model = DailyManagerTransformer(tiny_manager_config())
    handle = model.encoder.register_forward_hook(hook)
    try:
        model.eval()
        model(make_inputs(batch_size=2))
    finally:
        handle.remove()
    assert captured["seq"][1] == 106
    assert captured["seq"][-1] == 16


def test_nullable_lifecycle_fill_changes_representation_safely():
    torch.manual_seed(11)
    model = DailyManagerTransformer(tiny_manager_config())
    model.eval()
    inputs = make_inputs(batch_size=2, seed=3)
    filled = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    # Present derived struct with null timing -> NaN channels (adapter
    # contract); mark derived present so the NaN path is exercised.
    inputs["board_numeric"][..., 9:] = float("nan")
    inputs["board_mask"][..., 3] = 1
    with torch.no_grad():
        out_nan = model(inputs)
        out_filled = model(filled)
    for name, value in out_nan.items():
        assert torch.isfinite(value).all(), name
    assert not all(
        torch.equal(out_nan[name], out_filled[name]) for name in out_nan)


def test_metadata_result_keys_are_rejected_not_consumed():
    model = DailyManagerTransformer(tiny_manager_config())
    model.eval()
    inputs = make_inputs(batch_size=2, seed=4)
    leaked = dict(inputs)
    leaked.update({
        "names": ["a", "b"],
        "avg_score": np.array([2999.0, 3001.0]),
        "final_bank_self": np.array([100.0, 200.0]),
        "final_bank_opponent": np.array([100.0, 200.0]),
        "partition_date": ["2026-08-20", "2026-08-20"],
        "source_path": ["x.parquet", "x.parquet"],
        "meta": [{}, {}],
    })
    with pytest.raises(ValueError, match="unknown input keys"):
        model(leaked)
    # Without the metadata keys the same batch forwards cleanly.
    with torch.no_grad():
        model(inputs)


def test_disabled_opponent_mode_requires_and_uses_no_opponent_inputs():
    torch.manual_seed(13)
    model = DailyManagerTransformer(tiny_manager_config())
    model.eval()
    inputs = make_inputs(batch_size=2, seed=5)
    with torch.no_grad():
        baseline = model(inputs)
        with_opponent_arrays = model({
            **inputs,
            "opp_board_kind": inputs["board_kind"].clone(),
            "opp_board_crop": inputs["board_crop"].clone(),
            "opp_board_animal": inputs["board_animal"].clone(),
            "opp_board_numeric": inputs["board_numeric"].clone(),
            "opp_board_bool": inputs["board_bool"].clone(),
            "opp_board_mask": inputs["board_mask"].clone(),
            "opp_scalars": torch.rand((2, 2)),
            "opp_unlocked": torch.ones((2, 4), dtype=torch.uint8),
        })
    for name in baseline:
        assert torch.allclose(baseline[name], with_opponent_arrays[name]), \
            f"opponent arrays influenced disabled-opponent output {name}"


def test_enabled_opponent_mode_consumes_public_tiles_with_shared_encoder():
    torch.manual_seed(17)
    config = tiny_manager_config(include_opponent_board=True)
    model = DailyManagerTransformer(config)
    model.eval()
    own = make_inputs(batch_size=2, seed=6)
    opponent = make_inputs(batch_size=2, seed=7)
    opp_inputs = {f"opp_{k}": v for k, v in opponent.items()
                  if k.startswith("board_")}
    opp_inputs["opp_scalars"] = opponent["scalars"][:, :2].contiguous()
    opp_inputs["opp_unlocked"] = opponent["unlocked"]

    with pytest.raises(ValueError, match="missing required input keys"):
        model(dict(own))
    # Opponent-private resource paths are never accepted.
    private = {**own, **opp_inputs, "opp_shed_counts": torch.ones((2, 12))}
    with pytest.raises(ValueError, match="unknown input keys"):
        model(private)

    with torch.no_grad():
        base = model({**own, **opp_inputs})
        changed = model({**own, **{
            key: value.flip(-1) for key, value in opp_inputs.items()}})
    assert any(not torch.equal(base[name], changed[name])
               for name in base)
    # One shared TileEncoder instance encodes both boards; the learned role
    # embedding is what separates own from opponent PUBLIC tokens.
    encoders = [m for m in model.modules() if isinstance(m, TileEncoder)]
    assert len(encoders) == 1
    assert model.role_embedding.weight.shape[0] == 2


# ---------------------------------------------------------------- loss


def test_target_shapes_ranges_and_land_classes_validated():
    model = DailyManagerTransformer(tiny_manager_config())
    inputs = make_inputs(batch_size=2, seed=8)
    outputs = model(inputs)

    bad_crop = make_targets(batch_size=2, seed=9)
    bad_crop["crop_target"][0, 0] = 101  # count_max=100
    with pytest.raises(ValueError, match="crop_target"):
        manager_loss(outputs, bad_crop)

    bad_land_low = make_targets(batch_size=2, seed=9)
    bad_land_low["land_count"][0] = 0
    with pytest.raises(ValueError, match="land_count"):
        manager_loss(outputs, bad_land_low)

    bad_land_high = make_targets(batch_size=2, seed=9)
    bad_land_high["land_count"][1] = 5
    with pytest.raises(ValueError, match="land_count"):
        manager_loss(outputs, bad_land_high)

    missing_care = make_targets(batch_size=2, seed=9)
    del missing_care["care_target"]
    with pytest.raises(ValueError, match="care_target"):
        manager_loss(outputs, missing_care)

    bad_shape = make_targets(batch_size=2, seed=9)
    bad_shape["animal_target"] = bad_shape["animal_target"][:, :2]
    with pytest.raises(ValueError, match="animal_target"):
        manager_loss(outputs, bad_shape)


def test_sell_group_output_shapes_and_weighted_total():
    torch.manual_seed(19)
    weights = ManagerLossConfig(crop_weight=2.0, sell_presence_weight=0.5)
    model = DailyManagerTransformer(tiny_manager_config())
    inputs = make_inputs(batch_size=2, seed=10)
    targets = make_targets(batch_size=2, seed=11)
    outputs = model(inputs)
    total, groups = manager_loss(outputs, targets, weights)
    assert set(groups) == set(GROUP_NAMES)
    expected = sum(weights.weight(name) * groups[name]
                   for name in GROUP_NAMES)
    assert torch.allclose(total, expected)
    assert groups["sell_presence"].shape == ()
    assert groups["sell_quantity"].shape == ()


def test_quantity_loss_masks_negative_cells_and_zero_when_none_positive():
    torch.manual_seed(23)
    model = DailyManagerTransformer(tiny_manager_config())
    model.eval()
    inputs = make_inputs(batch_size=2, seed=12)
    outputs = model(inputs)  # graph kept for the zero-positive grad check

    targets_a = make_targets(batch_size=2, seed=13, positive_cells=4)
    targets_b = {k: v.clone() for k, v in targets_a.items()}
    negative = ~targets_a["sell_presence"]
    targets_b["sell_quantity_log1p"][negative] = 42.0  # ignored garbage
    _, groups_a = manager_loss(outputs, targets_a)
    _, groups_b = manager_loss(outputs, targets_b)
    assert torch.allclose(groups_a["sell_quantity"], groups_b["sell_quantity"])

    none_positive = make_targets(batch_size=2, seed=14, positive_cells=0)
    assert not none_positive["sell_presence"].any()
    quantity_head = model.sell_quantity_head.weight
    total, groups = manager_loss(outputs, none_positive)
    assert groups["sell_quantity"].item() == 0.0
    total.backward()
    assert quantity_head.grad is not None
    assert torch.isfinite(quantity_head.grad).all()


def test_adapter_repeated_sell_events_over_cap_used_without_clipping():
    from bc_manager.adapter import build_targets

    sells = [[{"product": "WHEAT", "hour": 5, "quantity": 60},
              {"product": "WHEAT", "hour": 7, "quantity": 70}]]
    targets = build_targets([
        {"crop_composition_end": {}, "animal_counts_end": {},
         "unlocked_quadrants_end": ["NW"], "fertilizer_by_crop": {},
         "care_by_animal": {}},
    ] * len(sells), sells)
    bounded = targets["sell_quantity_bounded"]
    assert bounded.max() > 100  # repeated same-bin events accumulate
    log1p_targets = {
        "crop_target": torch.from_numpy(targets["crop_target"]),
        "animal_target": torch.from_numpy(targets["animal_target"]),
        "land_count": torch.from_numpy(targets["land_count"]),
        "fertilizer_target": torch.from_numpy(targets["fertilizer_target"]),
        "care_target": torch.from_numpy(targets["care_target"]),
        "sell_presence": torch.from_numpy(targets["sell_presence"]),
        "sell_quantity_log1p":
            torch.from_numpy(targets["sell_quantity_log1p"]),
    }
    torch.manual_seed(29)
    model = DailyManagerTransformer(tiny_manager_config())
    outputs = model(make_inputs(batch_size=len(sells), seed=15))
    total, groups = manager_loss(outputs, log1p_targets)
    assert torch.isfinite(total)
    # The >100 cell is consumed at its true log1p magnitude, never clipped.
    assert log1p_targets["sell_quantity_log1p"].max() > math.log1p(100)


# -------------------------------------------------------------- learning


def test_tiny_fixed_batch_overfits_substantially():
    torch.manual_seed(31)
    batch_size = 8
    config = tiny_manager_config()
    model = DailyManagerTransformer(config)
    # Distinct day per row gives each example a unique identity through the
    # DAY token, making per-row memorization learnable by the tiny model.
    inputs = make_inputs(batch_size=batch_size, seed=16)
    inputs["day"] = torch.arange(batch_size, dtype=torch.int16)
    inputs["days_remaining"] = (29 - inputs["day"].long()).to(torch.int16)
    targets = make_targets(batch_size=batch_size, seed=17)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    initial = None
    final = None
    model.train()
    for step in range(301):
        optimizer.zero_grad()
        outputs = model(inputs)
        total, _ = manager_loss(outputs, targets)
        if step == 0:
            initial = total.item()
        if step == 300:
            final = total.item()
        total.backward()
        optimizer.step()
    print(f"overfit total loss: initial={initial:.4f} final={final:.4f}")
    assert math.isfinite(initial) and math.isfinite(final)
    assert final < 0.25 * initial, (
        f"tiny batch failed to overfit: initial={initial:.4f}, "
        f"final={final:.4f}")
    with torch.no_grad():
        crop_pred = predict_counts(model(inputs)["crop_logits"])
    assert (crop_pred == targets["crop_target"]).float().mean() > 0.5


def test_interleaved_batch_rows_are_independent():
    torch.manual_seed(37)
    model = DailyManagerTransformer(tiny_manager_config())
    model.eval()
    pattern_a = make_inputs(batch_size=2, seed=18)
    pattern_b = make_inputs(batch_size=2, seed=19)
    interleaved = {
        key: torch.stack([pattern_a[key][0], pattern_b[key][0],
                          pattern_a[key][1], pattern_b[key][1]])
        for key in pattern_a
    }
    with torch.no_grad():
        mixed = model(interleaved)
        only_a = model(pattern_a)
        only_b = model(pattern_b)
    for name in mixed:
        # Rows are independent modules-wise; tiny numeric drift comes from
        # batch-shape-dependent GEMM reduction order only.
        assert torch.allclose(mixed[name][0::2], only_a[name], atol=1e-4), \
            name
        assert torch.allclose(mixed[name][1::2], only_b[name], atol=1e-4), \
            name


@pytest.mark.skipif(not REAL_SAMPLE.exists(),
                    reason="local real canonical sample not present")
def test_real_schema_v2_adapter_batch_forward_backward_smoke():
    table, report = load_selected_table(
        REAL_SAMPLE, dates=["2026-08-20"], min_score=0.0)
    assert report["rows_selected"] >= 8
    inputs_np, targets_np, _ = table_to_arrays(table.slice(0, 8))

    def tensor(name):
        return torch.from_numpy(np.ascontiguousarray(inputs_np[name]))

    inputs = {name: tensor(name) for name in inputs_np}
    targets = {key: torch.from_numpy(np.ascontiguousarray(value))
               for key, value in targets_np.items()}
    torch.manual_seed(41)
    model = DailyManagerTransformer(tiny_manager_config(count_max=100))
    _, total, groups = loss_and_backward(model, inputs, targets)
    assert torch.isfinite(total)
    assert all(torch.isfinite(loss) for loss in groups.values())
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# ------------------------------------------------------------ inference


def test_inference_helpers_contract():
    count_logits = torch.randn(2, NUM_CROPS, 101)
    assert predict_counts(count_logits).shape == (2, NUM_CROPS)
    land_logits = torch.randn(2, 4)
    land = predict_land(land_logits)
    assert ((land >= 1) & (land <= 4)).all()
    presence_logits = torch.randn(2, NUM_PRODUCTS, SELL_BIN_COUNT)
    quantity_log1p = torch.tensor([[[2.0, -1.0]]])
    presence, quantity = predict_sells(presence_logits,
                                       quantity_log1p.expand(2, 9, -1))
    assert bool(((presence >= 0) & (presence <= 1)).all())
    assert bool((quantity >= 0).all())  # negative log predictions clamp to 0
