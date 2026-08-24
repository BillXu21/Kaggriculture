"""Stage-1 V1 economics tests (issue #6): exact econ channels, cross-file
batch history grouping, live/batch parity + tracker resets, V0 checkpoint
compatibility, E forward/loss/checkpoint behavior, and coherence diagnostics.

No real corpus or checkpoints required; synthetic canonical records only.
"""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from test_bc_manager import _records  # noqa: E402

from bc_manager.adapter import load_dataset, load_train_val, table_to_arrays  # noqa: E402
from bc_manager.coherence import (  # noqa: E402
    cash_ratio,
    coherence_metrics,
    current_animal_counts,
    current_crop_counts,
    land_expansion_cost,
    lower_bound_acquisition_cost,
)
from bc_manager.economics import (  # noqa: E402
    ANIMAL_COSTS,
    ECONOMIC_CONTEXT_KEY,
    ECONOMIC_DIM,
    SEED_COSTS,
    EconomicHistory,
    affordability,
    cash_linear,
    economic_context,
    normalize_model_variant,
    signed_log_cash,
)
from bc_manager.live import EconomicHistory as LiveEconomicHistory  # noqa: E402
from bc_manager.live import encode_live_inputs  # noqa: E402
from bc_manager.loss import GROUP_NAMES, ManagerLossConfig  # noqa: E402
from bc_manager.model import DailyManagerTransformer, tiny_manager_config  # noqa: E402
from bc_manager.training import (  # noqa: E402
    TrainingConfig,
    arrays_to_tensors,
    evaluate,
    load_checkpoint,
    load_model_from_checkpoint,
    run_training,
)
from bc_manager_jax.checkpoint import expected_torch_state_shapes  # noqa: E402
from replay_daily.constants import (  # noqa: E402
    LAND_PRICES,
    PRODUCTS,
    SCHEMA_VERSION,
    SELL_BIN_ANCHORS,
)
from replay_daily.extractor import empty_events  # noqa: E402
from replay_daily.storage import write_parquet  # noqa: E402

TRAIN_DATES = ("2026-08-17",)
VAL_DATES = ("2026-08-21",)

_METADATA_FIELDS = (
    "episode_id", "source_dataset", "partition_date", "source_path", "seat",
    "player", "opponent", "seed", "module_version", "avg_score", "min_score",
    "max_score", "sum_score", "final_rewards", "final_bank_self",
    "final_bank_opponent",
)


# ------------------------------------------------------------ record builder


def _board10() -> list[list]:
    return [[None] * 10 for _ in range(10)]


def make_record(episode_id, seat, day, *, money=3000.0,
                unlocked=("NW",), partition_date="2026-08-17",
                min_score=2950.0, end_money=None) -> dict:
    """Minimal valid canonical schema-v3 record with controllable economics."""
    def state(m):
        return {
            "money": float(m),
            "board": _board10(),
            "unlocked_quadrants": list(unlocked),
            "farmer": [0, 0],
            "hands": [],
            "hires_today": 0,
        }

    self_state = dict(state(money), shed={}, seeds={}, inventories=[])
    shared = {"market": {"inventory": {}, "prices": {}},
              "town": {"unlocked_shops": [], "shop_counts": {}}}
    metadata = {name: None for name in _METADATA_FIELDS}
    metadata.update(episode_id=episode_id, seat=seat,
                    partition_date=partition_date, min_score=min_score)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "day": day,
        "start": {
            "day": day, "hour": 0,
            "self": self_state,
            "opponent_public": state(money),
            **copy.deepcopy(shared),
            "previous_execution": {"workers_hired": 0, "hire_cost": 0},
        },
        "events": {**empty_events(),
                   "hires": {"submitted": 0,
                             "realized": {"workers_hired": 0,
                                          "hire_cost": 0}}},
        "targets": {
            "crop_composition_end": {},
            "animal_counts_end": {},
            "unlocked_quadrants_end": list(unlocked),
            "land_expansion": {"expanded": False, "new_quadrants": []},
            "fertilizer_by_crop": {},
            "care_by_animal": {"GOOSE": 0, "COW": 0, "SHEEP": 0},
            "sell_quantity": {str(a): {p: 0 for p in PRODUCTS}
                              for a in SELL_BIN_ANCHORS},
        },
        "end": {
            "boundary": "terminal", "day": day, "hour": 23,
            "self": dict(state(end_money if end_money is not None else money),
                         shed={}, seeds={}, inventories=[]),
            "opponent_public": state(money),
            **shared,
        },
    }


def write_records(records, tmp_path, name="synthetic.parquet") -> Path:
    path = tmp_path / name
    write_parquet(records, path)
    return path


def load_econ(path_or_paths, dates=TRAIN_DATES):
    data = load_dataset(path_or_paths, dates=dates, min_score=2950.0,
                        with_economic_context=True)
    order = sorted(
        range(len(data["meta"])),
        key=lambda i: (data["meta"][i]["episode_id"],
                       data["meta"][i]["seat"], data["meta"][i]["day"]))
    return data, order


# --------------------------------------------------- 1. exact channel values


def test_channel_order_and_exact_values():
    m = 46.0
    row = economic_context(m, unlocked_count=1, prev_net_cash=None)
    assert row.shape == (ECONOMIC_DIM,) and row.dtype == np.float32
    assert row[0] == pytest.approx(math.log1p(m * 1e-4), abs=1e-6)
    assert row[1] == pytest.approx(m * 1e-4, abs=1e-6)
    for k, cost in enumerate(SEED_COSTS):
        assert row[2 + k] == pytest.approx(math.log1p(m / cost), abs=1e-6)
    for k, cost in enumerate(ANIMAL_COSTS):
        assert row[7 + k] == pytest.approx(math.log1p(m / cost), abs=1e-6)
    # One locked quadrant: next price is LAND_PRICES[0].
    assert row[10] == pytest.approx(math.log1p(m / LAND_PRICES[0]), abs=1e-6)
    assert row[11] == 1.0
    assert row[12] == 0.0 and row[13] == 0.0  # invalid history


def test_all_unlocked_saturates_land_channel_with_invalid_bit():
    row = economic_context(5000.0, unlocked_count=4, prev_net_cash=None)
    assert row[10] == 8.0
    assert row[11] == 0.0


def test_second_and_third_locked_quadrants_use_ordered_prices():
    assert economic_context(46.0, 2, None)[10] == \
        pytest.approx(math.log1p(46 / LAND_PRICES[1]), abs=1e-6)
    assert economic_context(46.0, 3, None)[10] == \
        pytest.approx(math.log1p(46 / LAND_PRICES[2]), abs=1e-6)


def test_zero_negative_cash_and_prev_delta_channels():
    zero = economic_context(0.0, 1, None)
    assert zero[:2].tolist() == [0.0, 0.0]
    assert np.all(zero[2:11] == 0.0)

    negative = economic_context(-50.0, 1, None)
    assert negative[0] == pytest.approx(-math.log1p(50 * 1e-4), abs=1e-6)
    assert negative[1] == pytest.approx(-50 * 1e-4, abs=1e-6)
    assert np.all(negative[2:11] == 0.0)  # conservative: same as zero cash

    delta = economic_context(250.0, 1, prev_net_cash=150.0)
    assert delta[12] == pytest.approx(signed_log_cash(150.0), abs=1e-6)
    assert delta[13] == 1.0
    invalid = economic_context(250.0, 1, None)
    assert invalid[12] == 0.0 and invalid[13] == 0.0


def test_transform_helpers_and_validation():
    assert signed_log_cash(-3e7) == -8.0  # log1p(3000) > 8 -> clip boundary
    assert signed_log_cash(-8e5) == pytest.approx(-math.log1p(80.0), abs=1e-6)
    assert cash_linear(1e6) == 8.0 and cash_linear(-1e6) == -8.0
    assert affordability(0.0, 10) == 0.0
    assert affordability(-5.0, 10) == 0.0
    assert affordability(1000.0, 10) == pytest.approx(math.log1p(100.0),
                                                      abs=1e-6)
    assert affordability(3e8, 10) == 8.0  # saturated log ratio
    with pytest.raises(ValueError, match="cost"):
        affordability(10.0, 0)
    with pytest.raises(ValueError, match="unlocked_count"):
        economic_context(10.0, 0, None)
    with pytest.raises(ValueError, match="unlocked_count"):
        economic_context(10.0, 5, None)
    with pytest.raises(ValueError, match="money"):
        economic_context(float("nan"), 1, None)
    assert normalize_model_variant("e") == "E"
    assert normalize_model_variant("V0") == "V0"
    with pytest.raises(ValueError, match="model_variant"):
        normalize_model_variant("X")  # J/JE are valid since Stage 2


# ------------------------------------- 2. cross-file batch history grouping


def test_batch_history_cross_file_out_of_order_gaps_and_isolation(tmp_path):
    records = [
        # Episode 1 seat 0: money 100 -> 250 -> 250 -> 400.
        make_record(1, 0, 0, money=100.0),
        make_record(1, 0, 1, money=250.0),
        make_record(1, 0, 2, money=250.0),
        make_record(1, 0, 3, money=400.0),
        # Episode 2 seat 1 interleaved: 1000 -> 900 -> 1200.
        make_record(2, 1, 0, money=1000.0),
        make_record(2, 1, 1, money=900.0),
        make_record(2, 1, 2, money=1200.0),
        # Same episode other seat: never joins across seats.
        make_record(1, 1, 1, money=777.0),
        # Gap: day 2 must NOT join day 0.
        make_record(3, 0, 0, money=50.0),
        make_record(3, 0, 2, money=90.0),
    ]
    # Interleave rows across two files so no episode is contiguous or even
    # wholly contained in one file.
    file_a = write_records([records[2], records[5], records[0],
                            records[9]], tmp_path, "a.parquet")
    file_b = write_records([records[1], records[3], records[4], records[6],
                            records[7], records[8]], tmp_path, "b.parquet")
    data, order = load_econ([file_a, file_b])
    econ = data["inputs"][ECONOMIC_CONTEXT_KEY][order]
    meta = [data["meta"][i] for i in order]

    expected_delta = {
        (1, 0, 0): None, (1, 0, 1): 150.0, (1, 0, 2): 0.0, (1, 0, 3): 150.0,
        (2, 1, 0): None, (2, 1, 1): -100.0, (2, 1, 2): 300.0,
        (1, 1, 1): None,   # seat isolation: no day-0 row for (1, seat 1)
        (3, 0, 0): None, (3, 0, 2): None,  # gap: day 1 missing
    }
    for row, m in zip(econ, meta):
        key = (m["episode_id"], m["seat"], m["day"])
        want = expected_delta[key]
        if want is None:
            assert row[13] == 0.0 and row[12] == 0.0, key
        else:
            assert row[13] == 1.0, key
            assert row[12] == pytest.approx(signed_log_cash(want), abs=1e-6)


def test_batch_history_ignores_intents_and_end_snapshots(tmp_path):
    # End-of-day money and submitted-intent fields differ wildly from the
    # start snapshots; the derived history must depend on starts only.
    plain = make_record(9, 0, 1, money=200.0)
    noisy = make_record(9, 0, 1, money=200.0, end_money=99999.0)
    noisy["events"]["buys"]["seeds"]["WHEAT"] = 500
    noisy["events"]["sells"].append({"product": "WHEAT", "quantity": 500,
                                     "hour": 3})
    day0 = make_record(9, 0, 0, money=100.0)
    results = []
    for second in (plain, noisy):
        path = write_records([day0, second], tmp_path, f"x{id(second)}.parquet")
        data, order = load_econ(path)
        results.append(data["inputs"][ECONOMIC_CONTEXT_KEY][order])
    np.testing.assert_array_equal(results[0], results[1])


def test_batch_duplicate_rows_and_missing_episode_id_fail(tmp_path):
    dup = [make_record(4, 0, 1, money=100.0), make_record(4, 0, 1, money=100.0)]
    path = write_records(dup, tmp_path, "dup.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        load_econ(path)

    anon = make_record(None, 0, 0, money=100.0)
    path = write_records([anon], tmp_path, "anon.parquet")
    with pytest.raises(ValueError, match="episode_id"):
        load_econ(path)


def test_batch_history_derives_per_split_with_gap_default(tmp_path):
    # Real corpus organization gives every episode exactly one
    # partition_date, so a split never slices an episode internally.
    # Derivation therefore runs per split on that split's rows: a day-1 row
    # whose day-0 neighbor was filtered out degrades to the documented
    # 0/invalid gap default rather than reading across splits.
    records = [
        make_record(5, 0, 0, money=100.0, partition_date="2026-08-17"),
        make_record(5, 0, 1, money=300.0, partition_date="2026-08-21"),
        make_record(6, 0, 0, money=100.0, partition_date="2026-08-17"),
        make_record(6, 0, 1, money=300.0, partition_date="2026-08-17"),
    ]
    path = write_records(records, tmp_path)
    train = load_train_val(path, train_dates=("2026-08-17",),
                           val_dates=("2026-08-21",), min_score=2950.0,
                           with_economic_context=True)
    train_meta = train["train"]["meta"]
    train_econ = train["train"]["inputs"][ECONOMIC_CONTEXT_KEY]
    by_day = {m["day"]: i for i, m in enumerate(train_meta)}
    assert train_econ[by_day[0], 13] == 0.0          # episode 6 day 0
    assert train_econ[by_day[1], 13] == 1.0          # joins its own day 0
    assert train_econ[by_day[1], 12] == \
        pytest.approx(signed_log_cash(200.0), abs=1e-6)
    # The val split holds only the stray day-1 row of episode 5: no day-0
    # neighbor inside the split -> invalid, never read across splits.
    val_econ = train["val"]["inputs"][ECONOMIC_CONTEXT_KEY]
    assert val_econ[0, 13] == 0.0 and val_econ[0, 12] == 0.0


# ------------------------------------------------- 3. live parity + tracker


def _live_obs(seat, day, money, unlocked=("NW",)):
    farm = {
        "farmer": [0, 0], "hands": [], "hires_today": 0,
        "money": float(money), "tiles": _board10(),
        "unlocked_quadrants": list(unlocked),
    }
    return {
        "day": day, "hour": 0, "step": day * 24, "player": seat,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def test_live_batch_economic_parity_explicit_prev_start():
    obs = _live_obs(0, 3, 250.0)
    live = encode_live_inputs(obs, 0, economic_prev_start=(2, 100.0))
    record = make_record(31, 0, 3, money=250.0)
    record["metadata"]["partition_date"] = "2026-08-17"
    ref_day0 = make_record(31, 0, 2, money=100.0)
    ref_day0["metadata"]["partition_date"] = "2026-08-17"
    # Direct table path (no temp file needed for the reference).
    from replay_daily.storage import records_to_table
    inputs, _, _ = table_to_arrays(
        records_to_table([ref_day0, record]), with_economic_context=True)
    # Row order follows the input list; our record is second.
    batch_row = inputs[ECONOMIC_CONTEXT_KEY][1]
    np.testing.assert_array_equal(live[ECONOMIC_CONTEXT_KEY][0], batch_row)


def test_live_batch_economic_parity_via_tracker_matches_explicit():
    obs = _live_obs(1, 5, 900.0)
    tracker = LiveEconomicHistory()
    tracker.observe(4, 1000.0)
    via_tracker = encode_live_inputs(obs, 1, economic_history=tracker)
    explicit = encode_live_inputs(obs, 1, economic_prev_start=(4, 1000.0))
    np.testing.assert_array_equal(via_tracker[ECONOMIC_CONTEXT_KEY],
                                  explicit[ECONOMIC_CONTEXT_KEY])


def test_live_v0_default_has_no_economic_key():
    obs = _live_obs(0, 0, 3000.0)
    live = encode_live_inputs(obs, 0)
    assert ECONOMIC_CONTEXT_KEY not in live


def test_live_conflicting_history_arguments_fail():
    obs = _live_obs(0, 1, 100.0)
    with pytest.raises(ValueError, match="not both"):
        encode_live_inputs(obs, 0, economic_history=EconomicHistory(),
                           economic_prev_start=(0, 50.0))


def test_economic_history_tracker_reset_gap_backwards():
    tracker = EconomicHistory()
    delta, valid = tracker.observe(0, 100.0)
    assert (delta, valid) == (0.0, False)  # day 0
    delta, valid = tracker.observe(1, 250.0)
    assert (delta, valid) == (150.0, True)
    delta, valid = tracker.observe(2, 250.0)
    assert (delta, valid) == (0.0, True)
    delta, valid = tracker.observe(5, 400.0)  # gap
    assert (delta, valid) == (0.0, False)
    delta, valid = tracker.observe(4, 380.0)  # backwards
    assert (delta, valid) == (0.0, False)
    tracker.reset()
    delta, valid = tracker.observe(5, 400.0)  # new episode
    assert (delta, valid) == (0.0, False)
    with pytest.raises(ValueError):
        tracker.observe(-1, 10.0)


# --------------------------------------------- 4. V0 compatibility evidence


def test_v0_state_dict_keys_shapes_and_param_counts_unchanged():
    tiny = DailyManagerTransformer(tiny_manager_config())
    default = DailyManagerTransformer()
    assert tiny.trainable_parameters == 37_008
    assert default.trainable_parameters == 1_071_040
    # The JAX V0 converter's expected shape map must keep matching exactly.
    assert expected_torch_state_shapes(tiny_manager_config()) == \
        {k: tuple(v.shape) for k, v in tiny.state_dict().items()}
    assert tuple(tiny.state_dict()["global_encoders.self_resource.0.weight"]
                 .shape) == (16, 35)


def test_v0_model_rejects_economic_input():
    model = DailyManagerTransformer(tiny_manager_config())
    batch = {
        "board_kind": torch.zeros(1, 100, dtype=torch.int16),
        "board_crop": torch.zeros(1, 100, dtype=torch.int8),
        "board_animal": torch.zeros(1, 100, dtype=torch.int8),
        "board_numeric": torch.zeros(1, 100, 11),
        "board_bool": torch.zeros(1, 100, 8, dtype=torch.bool),
        "board_mask": torch.zeros(1, 100, 4, dtype=torch.uint8),
        "scalars": torch.zeros(1, 4),
        "shed_counts": torch.zeros(1, 12, dtype=torch.int32),
        "seed_counts": torch.zeros(1, 5, dtype=torch.int32),
        "carried_counts": torch.zeros(1, 12, dtype=torch.int32),
        "unlocked": torch.zeros(1, 4, dtype=torch.uint8),
        "market_inventory": torch.zeros(1, 9, dtype=torch.int32),
        "market_prices": torch.zeros(1, 9),
        "shop_counts": torch.zeros(1, 9, dtype=torch.int32),
        "day": torch.zeros(1, dtype=torch.int16),
        "days_remaining": torch.zeros(1, dtype=torch.int16),
        ECONOMIC_CONTEXT_KEY: torch.zeros(1, ECONOMIC_DIM),
    }
    with pytest.raises(ValueError, match=ECONOMIC_CONTEXT_KEY):
        model(batch)


def test_old_style_payload_without_variant_loads_as_v0(tmp_path):
    torch.manual_seed(0)
    model = DailyManagerTransformer(tiny_manager_config())
    old_payload = {
        "format": "bc_manager_checkpoint_v1",
        "kind": "best",
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "model_config": {"d_model": 16, "num_layers": 1, "num_heads": 1,
                         "ffn_dim": 32, "dropout": 0.0, "count_max": 100,
                         "include_opponent_board": False},
        "training_config": {},
        "validation_metrics": {},
    }
    path = tmp_path / "old.pt"
    torch.save(old_payload, path)
    loaded, payload = load_model_from_checkpoint(path)
    assert payload.get("model_variant", "V0") == "V0"
    assert loaded.model_variant == "V0"
    # Eval output equivalence against the original weights.
    batch = _v0_dummy_batch(2)
    model.eval()
    with torch.no_grad():
        a = model(batch)
        b = loaded(batch)
    assert set(a) == set(b)
    for k in a:
        torch.testing.assert_close(a[k], b[k])
    # Explicit variant mismatch fails clearly.
    with pytest.raises(ValueError, match="does not match"):
        load_model_from_checkpoint(path, expected_variant="E")


def test_invalid_checkpoint_variant_fails_clearly(tmp_path):
    model = DailyManagerTransformer(tiny_manager_config())
    payload = {
        "format": "bc_manager_checkpoint_v1",
        "kind": "last",
        "epoch": 1,
        "model_variant": "ZZ",  # not a valid variant in any stage
        "model_state_dict": model.state_dict(),
        "model_config": {},
        "training_config": {},
        "validation_metrics": {},
    }
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model_variant"):
        load_model_from_checkpoint(path)


def _v0_dummy_batch(b: int) -> dict:
    return {
        "board_kind": torch.zeros(b, 100, dtype=torch.int16),
        "board_crop": torch.zeros(b, 100, dtype=torch.int8),
        "board_animal": torch.zeros(b, 100, dtype=torch.int8),
        "board_numeric": torch.zeros(b, 100, 11),
        "board_bool": torch.zeros(b, 100, 8, dtype=torch.bool),
        "board_mask": torch.zeros(b, 100, 4, dtype=torch.uint8),
        "scalars": torch.zeros(b, 4),
        "shed_counts": torch.zeros(b, 12, dtype=torch.int32),
        "seed_counts": torch.zeros(b, 5, dtype=torch.int32),
        "carried_counts": torch.zeros(b, 12, dtype=torch.int32),
        "unlocked": torch.ones(b, 4, dtype=torch.uint8),
        "market_inventory": torch.zeros(b, 9, dtype=torch.int32),
        "market_prices": torch.zeros(b, 9),
        "shop_counts": torch.zeros(b, 9, dtype=torch.int32),
        "day": torch.zeros(b, dtype=torch.int16),
        "days_remaining": torch.full((b,), 29, dtype=torch.int16),
    }


# ------------------------------------------ 5. E forward/loss/checkpoint/E2E


def _tiny_e_model() -> DailyManagerTransformer:
    torch.manual_seed(1)
    return DailyManagerTransformer(tiny_manager_config(), model_variant="E")


def _e_batch(b: int = 3) -> dict:
    batch = _v0_dummy_batch(b)
    batch[ECONOMIC_CONTEXT_KEY] = torch.rand(b, ECONOMIC_DIM)
    return batch


def test_e_forward_shapes_match_v0_contract_loss_and_backward():
    e_model = _tiny_e_model()
    outputs = e_model(_e_batch())
    v0_outputs = DailyManagerTransformer(tiny_manager_config())(
        _v0_dummy_batch(3))
    assert set(outputs) == set(v0_outputs)
    for k in outputs:
        assert outputs[k].shape == v0_outputs[k].shape, k

    targets = {
        "crop_target": torch.randint(0, 20, (3, 5)),
        "animal_target": torch.randint(0, 20, (3, 3)),
        "land_count": torch.tensor([1, 2, 4]),
        "fertilizer_target": torch.randint(0, 5, (3, 5)),
        "care_target": torch.randint(0, 3, (3, 3)),
        "sell_presence": (torch.rand(3, 9, 6) > 0.5).float(),
        "sell_quantity_log1p": torch.rand(3, 9, 6) * 4.0,
    }
    loss, groups = ManagerLossConfig(), None
    from bc_manager.loss import manager_loss
    total, groups = manager_loss(outputs, targets, loss)
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in e_model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    # E extends only the self-resource encoder input.
    assert tuple(e_model.state_dict()["global_encoders.self_resource.0.weight"]
                 .shape) == (16, 35 + ECONOMIC_DIM)


def test_e_missing_wrong_or_nonfinite_economic_input_fails_clearly():
    model = _tiny_e_model()
    batch = _v0_dummy_batch(2)
    with pytest.raises(ValueError, match="economic_context"):
        model(batch)  # missing

    bad_shape = _e_batch(2)
    bad_shape[ECONOMIC_CONTEXT_KEY] = torch.zeros(2, ECONOMIC_DIM + 1)
    with pytest.raises(ValueError, match="economic_context"):
        model(bad_shape)

    bad_dtype = _e_batch(2)
    bad_dtype[ECONOMIC_CONTEXT_KEY] = \
        bad_dtype[ECONOMIC_CONTEXT_KEY].double()
    with pytest.raises(ValueError, match="economic_context"):
        model(bad_dtype)

    nonfinite = _e_batch(2)
    nonfinite[ECONOMIC_CONTEXT_KEY][0, 0] = float("inf")
    with pytest.raises(ValueError, match="economic_context"):
        model(nonfinite)


def test_e_checkpoint_round_trip_and_variant_mismatch(tmp_path):
    model = _tiny_e_model()
    ckpt = tmp_path / "e.pt"
    from bc_manager.training import save_checkpoint
    save_checkpoint(ckpt, kind="best", epoch=1, model=model,
                    model_config=tiny_manager_config(),
                    training_config=TrainingConfig(checkpoint_dir=None),
                    validation_metrics={}, model_variant="E")
    payload = load_checkpoint(ckpt)
    assert payload["format"] == "bc_manager_checkpoint_v1"
    assert payload["model_variant"] == "E"
    # Variant stays OUTSIDE model_config (JAX V0 converter compatibility).
    assert "model_variant" not in payload["model_config"]
    loaded, _ = load_model_from_checkpoint(ckpt)
    assert loaded.model_variant == "E"
    batch = _e_batch(2)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch), loaded(batch))
    with pytest.raises(ValueError, match="does not match"):
        load_model_from_checkpoint(ckpt, expected_variant="V0")


def test_run_training_e_end_to_end_smoke_with_checkpoints(tmp_path):
    path = write_records(_records(), tmp_path, "corpus.parquet")
    result = run_training(
        path, model_config=tiny_manager_config(),
        training_config=TrainingConfig(batch_size=2, epochs=1,
                                       checkpoint_dir=str(tmp_path / "ck")),
        train_dates=("2026-08-17", "2026-08-18", "2026-08-19"),
        val_dates=("2026-08-21",), min_score=2950.0, model_variant="E",
        log=lambda _: None)
    assert result["model_variant"] == "E"
    payload = load_checkpoint(tmp_path / "ck" / "best.pt")
    assert payload["model_variant"] == "E"
    loaded, _ = load_model_from_checkpoint(tmp_path / "ck" / "best.pt")
    assert loaded.model_variant == "E"


def test_cli_parser_accepts_variant_flag():
    from bc_manager.cli import build_parser
    args = build_parser().parse_args(
        ["x.parquet", "--variant", "E", "--checkpoint-dir", "d"])
    assert args.variant == "E"
    assert build_parser().parse_args(
        ["x.parquet", "--checkpoint-dir", "d"]).variant == "V0"


# ------------------------------------------------- 6. coherence diagnostics


def test_current_count_helpers_use_board_vocab_ids_only():
    board_crop = np.zeros((2, 100), dtype=np.int8)
    board_crop[0, 0] = 1  # WHEAT
    board_crop[0, 1] = 1  # WHEAT
    board_crop[0, 2] = 5  # MELON
    board_crop[1, 0] = 6  # UNKNOWN id must not count
    counts = current_crop_counts(board_crop)
    assert counts.tolist() == [[2, 0, 0, 0, 1], [0, 0, 0, 0, 0]]

    board_animal = np.zeros((2, 100), dtype=np.int8)
    board_animal[0, 0] = 2  # COW
    board_animal[0, 1] = 2  # COW
    board_animal[0, 2] = 3  # SHEEP
    animals = current_animal_counts(board_animal)
    assert animals.tolist() == [[0, 2, 1], [0, 0, 0]]


def test_land_expansion_cost_ordered_multi_expansion_hand_computed():
    assert land_expansion_cost(1, 1) == 0
    assert land_expansion_cost(1, 2) == 1000
    assert land_expansion_cost(1, 3) == 1000 + 2000   # NE then SW
    assert land_expansion_cost(2, 4) == 2000 + 4000
    assert land_expansion_cost(3, 1) == 0             # regression costs 0 here
    with pytest.raises(ValueError):
        land_expansion_cost(0, 1)
    with pytest.raises(ValueError):
        land_expansion_cost(1, 5)


def test_lower_bound_cost_positive_deltas_only_hand_computed():
    crop_cur = np.array([[2, 0, 0, 0, 0]])   # 2 WHEAT planted
    crop_tgt = np.array([[3, 0, 0, 0, 1]])   # +1 WHEAT (10) +1 MELON (80)
    ani_cur = np.array([[0, 1, 0]])          # 1 COW placed
    ani_tgt = np.array([[0, 2, 1]])          # +1 COW (400) +1 SHEEP (500)
    unlocked = np.array([1])
    land = np.array([3])                     # expansion 1000 + 2000
    cost = lower_bound_acquisition_cost(crop_cur, crop_tgt, ani_cur, ani_tgt,
                                        unlocked, land)
    assert cost[0] == 10 + 80 + 400 + 500 + 3000

    # Negative deltas never reduce the bound; all-unlocked -> no land cost.
    cost_low = lower_bound_acquisition_cost(
        np.array([[5, 0, 0, 0, 0]]), crop_tgt, ani_cur, ani_tgt,
        np.array([4]), np.array([4]))
    assert cost_low[0] == 80 + 400 + 500


def test_cash_ratio_zero_inf_and_finite_rules_documented():
    ratio = cash_ratio(np.array([0.0, 100.0, 100.0]), np.array([-5.0, 0.0, 50.0]))
    assert ratio[0] == 0.0            # zero cost stays 0 even at cash <= 0
    assert np.isinf(ratio[1])         # cost > 0 at cash <= 0 is +inf
    assert ratio[2] == pytest.approx(2.0)


def test_coherence_metrics_namespace_values():
    metrics = coherence_metrics(
        crop_target_counts=np.array([[3, 0, 0, 0, 1]]),
        animal_target_counts=np.array([[0, 2, 1]]),
        land_target=np.array([3]),
        crop_current=np.array([[2, 0, 0, 0, 0]]),
        animal_current=np.array([[0, 1, 0]]),
        unlocked_count=np.array([1]),
        cash=np.array([735.0]),
    )
    assert metrics["lower_bound_cost_mean"] == 3990.0
    assert metrics["ratio_gt_1x_rate"] == 1.0
    assert metrics["ratio_gt_2x_rate"] == 1.0  # 3990/735 > 2
    assert metrics["land_regression_rate"] == 0.0

    broke = coherence_metrics(
        np.zeros((1, 5)), np.zeros((1, 3)), np.array([1]),
        np.zeros((1, 5)), np.zeros((1, 3)), np.array([1]),
        cash=np.array([0.0]))
    assert broke["lower_bound_cost_mean"] == 0.0
    assert broke["ratio_median"] == 0.0  # zero cost, zero cash -> 0
    assert broke["ratio_gt_1x_rate"] == 0.0

    regression = coherence_metrics(
        np.zeros((2, 5)), np.zeros((2, 3)), np.array([1, 1]),
        np.zeros((2, 5)), np.zeros((2, 3)), np.array([2, 1]),
        cash=np.array([10.0, 10.0]))
    assert regression["land_regression_rate"] == 0.5


def test_evaluate_reports_coherence_namespaces(tmp_path):
    path = write_records(_records(), tmp_path, "eval.parquet")
    data = load_train_val(path, train_dates=("2026-08-17", "2026-08-18",
                                             "2026-08-19"),
                          val_dates=("2026-08-21",), min_score=2950.0)
    inputs, targets = arrays_to_tensors(
        data["val"]["inputs"], data["val"]["targets"],
        include_opponent_board=False)
    from bc_manager.training import ManagerTorchDataset
    loader = DataLoader(ManagerTorchDataset(inputs, targets), batch_size=2)
    model = DailyManagerTransformer(tiny_manager_config())
    report = evaluate(model, loader, torch.device("cpu"),
                      ManagerLossConfig())
    assert np.isfinite(report["total"])
    for name in GROUP_NAMES:
        assert f"group.{name}" in report  # losses unchanged
    for namespace in ("pred", "expert"):
        block = report["coherence"][namespace]
        for key in ("lower_bound_cost_mean", "lower_bound_cost_median",
                    "ratio_finite_mean", "ratio_gt_1x_rate",
                    "ratio_gt_2x_rate", "land_regression_rate"):
            assert key in block
            assert np.isfinite(block[key])
