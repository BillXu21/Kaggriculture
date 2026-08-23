"""CPU numerical parity tests: bc_manager (PyTorch) vs bc_manager_jax.

Covers: forward shapes for both board configs, strict checkpoint conversion,
eval parity of all seven output groups for seeded tiny models (own-only and
include_opponent_board) plus one random-initialized default-size model,
exact decoded count/land predictions, exact loss-group parity, native
save/load round-trip, and loud rejection of bad checkpoints/configs.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
import torch

from bc_manager.loss import manager_loss as torch_manager_loss
from bc_manager.model import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SELL_BIN_COUNT,
    DailyManagerTransformer,
    ManagerConfig as TorchManagerConfig,
    predict_counts as torch_predict_counts,
    predict_land as torch_predict_land,
    tiny_manager_config as torch_tiny_config,
)
from bc_manager.training import TrainingConfig, save_checkpoint as \
    torch_save_checkpoint
from bc_manager_jax.checkpoint import (
    NATIVE_CHECKPOINT_FORMAT,
    convert_torch_state_dict,
    expected_torch_state_shapes,
    load_native,
    load_torch_checkpoint,
    save_native,
)
from bc_manager_jax.loss import manager_loss as jax_manager_loss
from bc_manager_jax.model import (
    ManagerConfig as JaxManagerConfig,
    forward as jax_forward,
    init_params,
    predict_counts as jax_predict_counts,
    predict_land as jax_predict_land,
    tiny_manager_config as jax_tiny_config,
)

# Empirically determined float32 CPU tolerances (observed worst-case max
# diff ~9.5e-7 across all groups/modes; see parity assertions below).
MAX_ABS_TOL = 2e-6
MEAN_ABS_TOL = 5e-7


# ------------------------------------------------------------- fixtures


def make_inputs(batch_size: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    """Identical generator logic to tests/test_bc_manager_model.py."""
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


def build_pair(torch_config: TorchManagerConfig,
               jax_config: JaxManagerConfig, seed: int):
    """Seeded torch model + strictly converted identical JAX params."""
    torch.manual_seed(seed)
    model = DailyManagerTransformer(torch_config)
    model.eval()
    params = convert_torch_state_dict(model.state_dict(), jax_config)
    return model, params


def to_numpy(inputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.numpy() for key, value in inputs.items()}


def opponent_inputs(own: dict[str, torch.Tensor],
                    seed: int) -> dict[str, torch.Tensor]:
    opp = make_inputs(batch_size=own["board_kind"].shape[0], seed=seed)
    merged = dict(own)
    for key in list(opp):
        if key.startswith("board_"):
            merged[f"opp_{key}"] = opp[key]
    merged["opp_scalars"] = opp["scalars"][:, :2].contiguous()
    merged["opp_unlocked"] = opp["unlocked"]
    return merged


def assert_parity(t_outputs, j_outputs) -> None:
    assert set(t_outputs.keys()) == set(j_outputs.keys())
    for name in t_outputs:
        t = t_outputs[name].detach().numpy()
        j = np.asarray(j_outputs[name])
        assert t.shape == j.shape, name
        diff = np.abs(t - j)
        print(f"parity {name}: max={diff.max():.3e} mean={diff.mean():.3e}")
        assert diff.max() <= MAX_ABS_TOL, (name, diff.max())
        assert diff.mean() <= MEAN_ABS_TOL, (name, diff.mean())


# --------------------------------------------------------------- shapes


@pytest.mark.parametrize("include_opponent", [False, True])
def test_forward_output_shapes_both_board_configs(include_opponent):
    config = jax_tiny_config(include_opponent_board=include_opponent)
    _, params = build_pair(torch_tiny_config(
        include_opponent_board=include_opponent), config, seed=5)
    inputs = make_inputs(batch_size=3, seed=6)
    if include_opponent:
        inputs = opponent_inputs(inputs, seed=7)
    outputs = jax_forward(params, to_numpy(inputs), config)
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


def test_input_validation_rejects_missing_and_metadata_keys():
    config = jax_tiny_config()
    _, params = build_pair(torch_tiny_config(), config, seed=8)
    inputs = to_numpy(make_inputs(batch_size=2, seed=9))
    leaked = dict(inputs)
    leaked["avg_score"] = np.array([2999.0, 3001.0])
    with pytest.raises(ValueError, match="unknown input keys"):
        jax_forward(params, leaked, config)
    missing = {k: v for k, v in inputs.items() if k != "day"}
    with pytest.raises(ValueError, match="missing required input keys"):
        jax_forward(params, missing, config)
    # Opponent arrays are accepted-but-ignored when the board config
    # disables them (same semantics as the PyTorch model).
    baseline = jax_forward(params, inputs, config)
    ignored = jax_forward(params, opponent_inputs(inputs, seed=10), config)
    for name in baseline:
        assert np.array_equal(np.asarray(baseline[name]),
                              np.asarray(ignored[name])), name


# --------------------------------------------------------------- parity


def test_eval_parity_own_only_tiny():
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=11)
    inputs = to_numpy(make_inputs(batch_size=3, seed=12))
    with torch.no_grad():
        t_outputs = model({k: torch.from_numpy(v) for k, v in inputs.items()})
    j_outputs = jax_forward(params, inputs, jax_config)
    assert_parity(t_outputs, j_outputs)


def test_eval_parity_opponent_mode_random_init_shared_encoder():
    torch_config = torch_tiny_config(include_opponent_board=True)
    jax_config = jax_tiny_config(include_opponent_board=True)
    model, params = build_pair(torch_config, jax_config, seed=13)
    inputs = to_numpy(opponent_inputs(make_inputs(batch_size=2, seed=14),
                                      seed=15))
    with torch.no_grad():
        t_outputs = model({k: torch.from_numpy(v) for k, v in inputs.items()})
    j_outputs = jax_forward(params, inputs, jax_config)
    assert_parity(t_outputs, j_outputs)
    # Shared tile encoder semantics: role embedding is the only difference.
    assert params["role_embedding"].shape == (2, jax_config.d_model)


def test_eval_parity_default_size_own_only():
    torch_config = TorchManagerConfig()
    jax_config = JaxManagerConfig()
    model, params = build_pair(torch_config, jax_config, seed=16)
    inputs = to_numpy(make_inputs(batch_size=2, seed=17))
    with torch.no_grad():
        t_outputs = model({k: torch.from_numpy(v) for k, v in inputs.items()})
    j_outputs = jax_forward(params, inputs, jax_config)
    assert_parity(t_outputs, j_outputs)


def test_decoded_predictions_match_exactly():
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=18)
    inputs = to_numpy(make_inputs(batch_size=4, seed=19))
    with torch.no_grad():
        t_outputs = model({k: torch.from_numpy(v) for k, v in inputs.items()})
    j_outputs = jax_forward(params, inputs, jax_config)
    t_counts = torch_predict_counts(
        t_outputs["crop_logits"]).numpy()
    j_counts = np.asarray(jax_predict_counts(j_outputs["crop_logits"]))
    assert np.array_equal(t_counts, j_counts)
    t_land = torch_predict_land(t_outputs["land_logits"]).numpy()
    j_land = np.asarray(jax_predict_land(j_outputs["land_logits"]))
    assert np.array_equal(t_land, j_land)


# ----------------------------------------------------------------- loss


@pytest.mark.parametrize("positive_cells", [6, 0])
def test_loss_group_parity_same_outputs_and_targets(positive_cells):
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=20)
    inputs_t = make_inputs(batch_size=4, seed=21)
    targets_t = make_targets(batch_size=4, seed=22,
                             positive_cells=positive_cells)
    with torch.no_grad():
        t_outputs = model(inputs_t)
    t_total, t_groups = torch_manager_loss(t_outputs, targets_t)

    j_outputs = jax_forward(params, to_numpy(inputs_t), jax_config)
    j_targets = {key: value.numpy() for key, value in targets_t.items()}
    j_total, j_groups = jax_manager_loss(j_outputs, j_targets)

    assert set(j_groups) == set(t_groups)
    print(f"loss total: torch={float(t_total):.8f} jax={float(j_total):.8f}")
    assert abs(float(t_total) - float(j_total)) <= MAX_ABS_TOL
    for name in t_groups:
        diff = abs(float(t_groups[name]) - float(j_groups[name]))
        print(f"loss group {name}: torch={float(t_groups[name]):.8f} "
              f"jax={float(j_groups[name]):.8f} diff={diff:.3e}")
        assert diff <= MAX_ABS_TOL, (name, diff)


def test_loss_weighted_total_and_validation_errors():
    from bc_manager.loss import GROUP_NAMES, ManagerLossConfig

    weights = ManagerLossConfig(crop_weight=2.0, sell_presence_weight=0.5)
    jax_config = jax_tiny_config()
    _, params = build_pair(torch_tiny_config(), jax_config, seed=23)
    inputs = to_numpy(make_inputs(batch_size=2, seed=24))
    targets = {k: v.numpy()
               for k, v in make_targets(batch_size=2, seed=25).items()}
    outputs = jax_forward(params, inputs, jax_config)
    total, groups = jax_manager_loss(outputs, targets, weights)
    expected = sum(weights.weight(name) * groups[name]
                   for name in GROUP_NAMES)
    assert abs(float(total - expected)) < 1e-6

    bad = dict(targets)
    bad["crop_target"] = targets["crop_target"].copy()
    bad["crop_target"][0, 0] = 101
    with pytest.raises(ValueError, match="crop_target"):
        jax_manager_loss(outputs, bad)
    bad_land = dict(targets)
    bad_land["land_count"] = targets["land_count"].copy()
    bad_land["land_count"][0] = 0
    with pytest.raises(ValueError, match="land_count"):
        jax_manager_loss(outputs, bad_land)
    del bad_land["care_target"]
    with pytest.raises(ValueError, match="care_target"):
        jax_manager_loss(outputs, bad_land)


# ------------------------------------------------------------ conversion


def test_expected_state_shapes_cover_full_torch_state_dict():
    for include_opponent in (False, True):
        torch.manual_seed(26)
        model = DailyManagerTransformer(torch_tiny_config(
            include_opponent_board=include_opponent))
        expected = expected_torch_state_shapes(jax_tiny_config(
            include_opponent_board=include_opponent))
        assert set(expected) == set(model.state_dict().keys())
        for key, shape in expected.items():
            assert tuple(model.state_dict()[key].shape) == shape, key


def test_converter_rejects_missing_unexpected_shape_and_dtype():
    jax_config = jax_tiny_config()
    torch.manual_seed(27)
    state = dict(DailyManagerTransformer(
        torch_tiny_config()).state_dict())

    def expect_failure(mutated, match):
        with pytest.raises(ValueError, match=match):
            convert_torch_state_dict(mutated, jax_config)

    missing = {k: v for k, v in state.items()
               if k != "encoder_norm.weight"}
    expect_failure(missing, "missing keys")
    extra = dict(state)
    extra["mystery_head.weight"] = state["land_head.weight"]
    expect_failure(extra, "unexpected keys")
    bad_shape = dict(state)
    bad_shape["land_head.weight"] = torch.zeros(3, 16)
    expect_failure(bad_shape, "shape mismatch")
    bad_dtype = dict(state)
    bad_dtype["land_head.weight"] = \
        state["land_head.weight"].to(torch.float64)
    expect_failure(bad_dtype, "must be float32")


def test_converter_rejects_incompatible_config():
    jax_config = jax_tiny_config()
    torch.manual_seed(28)
    state = DailyManagerTransformer(torch_tiny_config()).state_dict()
    # Architecture-size mismatch is caught by the shape checks.
    with pytest.raises(ValueError, match="shape mismatch"):
        convert_torch_state_dict(state, jax_tiny_config(d_model=32))
    # NOTE: own-only vs include_opponent_board state dicts are structurally
    # identical (the opponent board adds no parameters), so that
    # incompatibility is enforced at the checkpoint-metadata level in
    # `load_torch_checkpoint` (see the dedicated test below).


def test_real_checkpoint_payload_round_trip(tmp_path: Path):
    """Full pipeline through bc_manager.training.save_checkpoint."""
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    torch.manual_seed(29)
    model = DailyManagerTransformer(torch_config)
    path = tmp_path / "best.pt"
    torch_save_checkpoint(path, kind="best", epoch=3, model=model,
                          model_config=torch_config,
                          training_config=TrainingConfig(),
                          validation_metrics={"total": 12.5})
    params, meta = load_torch_checkpoint(path, jax_config)
    assert meta["format"] == "bc_manager_checkpoint_v1"
    assert meta["epoch"] == 3
    assert meta["kind"] == "best"
    assert meta["model_config"]["d_model"] == 16
    inputs = to_numpy(make_inputs(batch_size=2, seed=30))
    with torch.no_grad():
        t_outputs = model.eval()(make_inputs(batch_size=2, seed=30))
    assert_parity(t_outputs, jax_forward(params, inputs, jax_config))
    # Payload (not just path) also accepted.
    payload_params, _ = load_torch_checkpoint(
        torch.load(path, map_location="cpu", weights_only=True), jax_config)
    flat_a, _ = jax.tree_util.tree_flatten(params)
    flat_b, _ = jax.tree_util.tree_flatten(payload_params)
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(flat_a, flat_b))


def test_load_torch_checkpoint_rejects_bad_format_and_config(tmp_path):
    torch.manual_seed(31)
    model = DailyManagerTransformer(torch_tiny_config())
    path = tmp_path / "best.pt"
    torch_save_checkpoint(path, kind="best", epoch=1, model=model,
                          model_config=torch_tiny_config(),
                          training_config=TrainingConfig(),
                          validation_metrics={})
    # Own-only checkpoint requested with an opponent-board config.
    with pytest.raises(ValueError, match="incompatible"):
        load_torch_checkpoint(path, jax_tiny_config(
            include_opponent_board=True))
    bad_format = tmp_path / "bad.pt"
    torch.save({"format": "something_else"}, bad_format)
    with pytest.raises(ValueError, match="unrecognized checkpoint format"):
        load_torch_checkpoint(bad_format, jax_tiny_config())


# -------------------------------------------------------- native format


def test_native_save_load_round_trip(tmp_path):
    jax_config = jax_tiny_config(include_opponent_board=True)
    params = init_params(jax_config, seed=32)
    path = tmp_path / "native.npz"
    save_native(path, params, jax_config,
                metadata={"epoch": 4, "note": "stage1"})
    loaded, meta = load_native(path, jax_config)
    assert meta["format"] == NATIVE_CHECKPOINT_FORMAT
    assert meta["metadata"]["epoch"] == 4
    flat_a, _ = jax.tree_util.tree_flatten(params)
    flat_b, _ = jax.tree_util.tree_flatten(loaded)
    assert len(flat_a) == len(flat_b)
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(flat_a, flat_b))
    # Round-tripped params produce identical forwards.
    inputs = to_numpy(opponent_inputs(make_inputs(batch_size=2, seed=33),
                                      seed=34))
    out_a = jax_forward(params, inputs, jax_config)
    out_b = jax_forward(loaded, inputs, jax_config)
    for name in out_a:
        assert np.array_equal(np.asarray(out_a[name]),
                              np.asarray(out_b[name])), name
    size_bytes = path.stat().st_size
    print(f"native checkpoint bytes: {size_bytes}")
    assert size_bytes < 50_000_000  # small artifact, no large committed file


def test_native_load_rejects_corruption_and_mismatch(tmp_path):
    jax_config = jax_tiny_config()
    params = init_params(jax_config, seed=35)
    path = tmp_path / "native.npz"
    save_native(path, params, jax_config)

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz archive at all")
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        load_native(corrupt, jax_config)

    with pytest.raises(ValueError, match="incompatible"):
        load_native(path, jax_tiny_config(d_model=32))

    # Tampered param array inside a valid archive must fail loudly (either
    # at npz decode or at the per-param shape check).
    import zipfile
    tampered = tmp_path / "tampered.npz"
    with zipfile.ZipFile(path) as src, \
            zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.startswith("param:") and "heads/land" in \
                    item.filename:
                data = data[:-1]  # truncate one float
            dst.writestr(item, data)
    with pytest.raises(ValueError,
                       match="(corrupt or unreadable|param .* shape)"):
        load_native(tampered, jax_config)


# ------------------------------------------------------- training hook


def test_training_flag_requires_rng_and_changes_outputs_deterministically():
    jax_config = jax_tiny_config(dropout=0.3)
    params = init_params(jax_config, seed=36)
    inputs = to_numpy(make_inputs(batch_size=2, seed=37))
    with pytest.raises(ValueError, match="rng"):
        jax_forward(params, inputs, jax_config, training=True)
    key = jax.random.PRNGKey(0)
    out_a = jax_forward(params, inputs, jax_config, training=True, rng=key)
    out_b = jax_forward(params, inputs, jax_config, training=True, rng=key)
    out_c = jax_forward(params, inputs, jax_config, training=True,
                        rng=jax.random.PRNGKey(1))
    for name in out_a:
        assert np.array_equal(np.asarray(out_a[name]),
                              np.asarray(out_b[name])), name
        assert not np.array_equal(np.asarray(out_a[name]),
                                  np.asarray(out_c[name])), name
