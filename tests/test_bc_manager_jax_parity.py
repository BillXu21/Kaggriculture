"""CPU numerical parity tests: bc_manager (PyTorch) vs bc_manager_jax.

Covers: forward shapes for both board configs, strict checkpoint conversion,
eval parity of all seven output groups for seeded tiny models (own-only and
include_opponent_board) plus one random-initialized default-size model,
exact decoded count/land predictions, exact loss-group parity, native
save/load round-trip, and loud rejection of bad checkpoints/configs.
"""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest
import torch

from bc_manager.economics import (
    ANIMAL_COSTS,
    ECONOMIC_CONTEXT_KEY,
    ECONOMIC_DIM,
    SEED_COSTS,
    EconomicHistory,
    affordability,
    cash_linear,
    economic_context,
    signed_log_cash,
)
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
    predict_sells as torch_predict_sells,
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
    SUPPORTED_MODEL_VARIANTS,
    ManagerConfig as JaxManagerConfig,
    empty_params,
    forward as jax_forward,
    init_params,
    predict_counts as jax_predict_counts,
    predict_land as jax_predict_land,
    predict_sells as jax_predict_sells,
    resolve_model_variant,
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
               jax_config: JaxManagerConfig, seed: int,
               model_variant: str = "V0"):
    """Seeded torch model + strictly converted identical JAX params."""
    torch.manual_seed(seed)
    model = DailyManagerTransformer(torch_config,
                                    model_variant=model_variant)
    model.eval()
    params = convert_torch_state_dict(model.state_dict(), jax_config,
                                      model_variant)
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


# =================================================== issue #8: E variant

E_REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / \
    "artifacts" / "local" / "bc-v1-E" / "best.pt"


def test_supported_variants_and_loud_joint_rejection():
    assert SUPPORTED_MODEL_VARIANTS == ("V0", "E")
    assert resolve_model_variant("v0") == "V0"
    assert resolve_model_variant(" e ") == "E"
    # J/JE are known torch variants that bc_manager_jax deliberately does
    # not port: they must fail with the explicit unsupported-variant
    # message, never a downstream key/shape error.
    for joint in ("J", "JE"):
        with pytest.raises(ValueError,
                           match=f"'{joint}' is not supported by "
                                 f"bc_manager_jax"):
            resolve_model_variant(joint)
        config = jax_tiny_config()
        with pytest.raises(ValueError,
                           match="not supported by bc_manager_jax"):
            empty_params(config, joint)
        with pytest.raises(ValueError,
                           match="not supported by bc_manager_jax"):
            jax_forward({}, config, config, model_variant=joint)
    # A fully unknown name fails at the authoritative normalizer instead.
    with pytest.raises(ValueError, match="unknown model_variant 'X'"):
        resolve_model_variant("X")


def test_parameter_counts_default_and_tiny_both_variants():
    expected = {("default", "V0"): 1_071_040, ("default", "E"): 1_072_832,
                ("tiny", "V0"): 37_008, ("tiny", "E"): 37_232}
    for (label, variant), want in expected.items():
        torch.manual_seed(40)
        if label == "default":
            t_cfg, j_cfg = TorchManagerConfig(), JaxManagerConfig()
        else:
            t_cfg, j_cfg = torch_tiny_config(), jax_tiny_config()
        model = DailyManagerTransformer(t_cfg, model_variant=variant)
        assert model.trainable_parameters == want, (label, variant)
        params = empty_params(j_cfg, variant)
        count = sum(int(np.asarray(leaf).size)
                    for leaf in jax.tree_util.tree_leaves(params))
        assert count == want, (label, variant)


def test_expected_self_resource_shape_widens_for_e():
    config = jax_tiny_config()
    v0 = expected_torch_state_shapes(config, "V0")
    e = expected_torch_state_shapes(config, "E")
    assert v0["global_encoders.self_resource.0.weight"] == (
        config.d_model, 35)
    assert e["global_encoders.self_resource.0.weight"] == (
        config.d_model, 35 + ECONOMIC_DIM)
    # Only the widened first-layer weight differs; every other key/shape
    # is identical between the variants.
    del v0["global_encoders.self_resource.0.weight"]
    del e["global_encoders.self_resource.0.weight"]
    assert v0 == e
    # Full E state-dict coverage against the real torch module.
    torch.manual_seed(41)
    model = DailyManagerTransformer(torch_tiny_config(),
                                    model_variant="E")
    assert set(e) | {"global_encoders.self_resource.0.weight"} == \
        set(model.state_dict().keys())
    for key, shape in expected_torch_state_shapes(config, "E").items():
        assert tuple(model.state_dict()[key].shape) == shape, key


def test_v0_rejects_economic_context_as_unknown_input():
    """V0 must never silently consume E features (torch + JAX)."""
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=42)
    inputs_t = make_inputs(batch_size=2, seed=43)
    inputs_t[ECONOMIC_CONTEXT_KEY] = torch.zeros(
        (2, ECONOMIC_DIM), dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown input keys"):
        model(inputs_t)
    inputs_np = to_numpy(inputs_t)
    with pytest.raises(ValueError, match="unknown input keys"):
        jax_forward(params, inputs_np, jax_config)


def test_e_requires_valid_economic_context():
    _, params = build_pair(torch_tiny_config(), jax_tiny_config(), seed=44,
                           model_variant="E")
    base = make_inputs(batch_size=3, seed=45)
    econ = np.zeros((3, ECONOMIC_DIM), dtype=np.float32)

    missing = {k: v.numpy() for k, v in base.items()}
    with pytest.raises(ValueError, match="missing required input keys"):
        jax_forward(params, missing, jax_tiny_config(), model_variant="E")

    def with_econ(array):
        inputs = {k: v.numpy() for k, v in base.items()}
        inputs[ECONOMIC_CONTEXT_KEY] = array
        return inputs

    with pytest.raises(ValueError, match="economic_context.*shape"):
        jax_forward(params, with_econ(np.zeros((3, 13), np.float32)),
                    jax_tiny_config(), model_variant="E")
    bad = econ.copy()
    bad[1, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        jax_forward(params, with_econ(bad), jax_tiny_config(),
                    model_variant="E")
    bad = econ.copy()
    bad[2, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        jax_forward(params, with_econ(bad), jax_tiny_config(),
                    model_variant="E")


def economic_scenario_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                       list[str]]:
    """Rows covering every audited history case via the AUTHORITATIVE
    `EconomicHistory`/`economic_context` path — no re-derived formulas.

    Returns (econ rows [N, 14], money per row, unlocked count per row,
    case labels).
    """
    cases = [
        # (label, [(day, money, unlocked)]) per episode/seat history
        [("day0-invalid", 0, 500.0, 1)],                       # day 0
        [("adjacent-a", 0, 300.0, 1), ("adjacent-b", 1, 750.0, 1)],
        [("gap-a", 0, 400.0, 2), ("gap-b", 2, 900.0, 2)],      # gap invalid
        [("reset-a", 3, 800.0, 2), ("backward", 1, 600.0, 2)],  # backwards
        [("saturated", 7, 50_000.0, 4)],                        # all land
        [("zero-cash", 4, 0.0, 1), ("negative", 5, -250.0, 1)],
    ]
    rows = []
    moneys = []
    unlocked = []
    labels = []
    for case in cases:
        history = EconomicHistory()
        for label, day, money, unlocked_count in case:
            rows.append(history.context(day, money, unlocked_count))
            moneys.append(money)
            unlocked.append(unlocked_count)
            labels.append(label)
    return (np.stack(rows).astype(np.float32),
            np.asarray(moneys, dtype=np.float64),
            np.asarray(unlocked, dtype=int),
            labels)


def test_economic_feature_cases_match_authoritative_path():
    """The exact arrays fed to JAX equal the authoritative NumPy/PyTorch
    path, across day-0 invalid, adjacent, gap, reset/backwards, all-land
    saturation, zero/negative cash, and the canonical channel order."""
    from replay_daily.constants import LAND_PRICES

    econ, moneys, unlocked_counts, labels = economic_scenario_batch()
    b = econ.shape[0]
    assert econ.shape == (b, ECONOMIC_DIM)

    # Channel order proven against the authoritative scalar helpers.
    i_zero = labels.index("zero-cash")
    row = econ[i_zero]
    assert row[0] == np.float32(signed_log_cash(0.0))
    assert row[1] == np.float32(cash_linear(0.0))
    for k, cost in enumerate(SEED_COSTS):
        assert row[2 + k] == np.float32(affordability(0.0, cost))
    for k, cost in enumerate(ANIMAL_COSTS):
        assert row[7 + k] == np.float32(affordability(0.0, cost))
    # zero cash, one locked quadrant: next-land affordability vs pinned price
    assert row[10] == np.float32(affordability(0.0, LAND_PRICES[0]))
    assert row[11] == 1.0 and row[12] == 0.0 and row[13] == 0.0
    i_sat = labels.index("saturated")
    sat = econ[i_sat]
    assert sat[10] == 8.0 and sat[11] == 0.0  # saturation, flagged invalid
    i_adj = labels.index("adjacent-b")
    adj = econ[i_adj]
    assert adj[12] == np.float32(signed_log_cash(750.0 - 300.0))
    assert adj[13] == 1.0
    for idx in (labels.index("day0-invalid"), labels.index("gap-b"),
                labels.index("backward")):
        assert econ[idx][13] == 0.0 and econ[idx][12] == 0.0

    # Identical authoritative arrays into BOTH frameworks must agree.
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=46,
                               model_variant="E")
    base = make_inputs(batch_size=b, seed=47)
    base["scalars"][:, 0] = torch.from_numpy(
        moneys.astype(np.float32)).contiguous()
    base[ECONOMIC_CONTEXT_KEY] = torch.from_numpy(econ)
    with torch.no_grad():
        t_outputs = model(base)
    j_outputs = jax_forward(params, to_numpy(base), jax_config,
                            model_variant="E")
    assert_parity(t_outputs, j_outputs)


def test_eval_parity_e_variant_tiny_all_groups():
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=48,
                               model_variant="E")
    econ, _, _, _ = economic_scenario_batch()
    b = econ.shape[0]
    inputs = make_inputs(batch_size=b, seed=49)
    inputs[ECONOMIC_CONTEXT_KEY] = torch.from_numpy(econ)
    with torch.no_grad():
        t_outputs = model(inputs)
    j_outputs = jax_forward(params, to_numpy(inputs), jax_config,
                            model_variant="E")
    assert set(t_outputs.keys()) == set(j_outputs.keys())
    worst_max, worst_mean = 0.0, 0.0
    for name in t_outputs:
        diff = np.abs(t_outputs[name].detach().numpy()
                      - np.asarray(j_outputs[name]))
        print(f"E parity {name}: max={diff.max():.3e} "
              f"mean={diff.mean():.3e}")
        worst_max = max(worst_max, float(diff.max()))
        worst_mean = max(worst_mean, float(diff.mean()))
        assert diff.max() <= MAX_ABS_TOL, (name, diff.max())
        assert diff.mean() <= MEAN_ABS_TOL, (name, diff.mean())
    print(f"E parity worst: max={worst_max:.3e} mean={worst_mean:.3e}")


def test_decoded_predictions_match_exactly_e_variant():
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=50,
                               model_variant="E")
    econ, _, _, _ = economic_scenario_batch()
    inputs = make_inputs(batch_size=econ.shape[0], seed=51)
    inputs[ECONOMIC_CONTEXT_KEY] = torch.from_numpy(econ)
    with torch.no_grad():
        t_outputs = model(inputs)
    j_outputs = jax_forward(params, to_numpy(inputs), jax_config,
                            model_variant="E")
    assert np.array_equal(
        torch_predict_counts(t_outputs["crop_logits"]).numpy(),
        np.asarray(jax_predict_counts(j_outputs["crop_logits"])))
    assert np.array_equal(
        torch_predict_counts(t_outputs["animal_logits"]).numpy(),
        np.asarray(jax_predict_counts(j_outputs["animal_logits"])))
    assert np.array_equal(
        torch_predict_land(t_outputs["land_logits"]).numpy(),
        np.asarray(jax_predict_land(j_outputs["land_logits"])))
    t_pres, t_qty = torch_predict_sells(t_outputs["sell_presence_logits"],
                                        t_outputs["sell_quantity_log1p"])
    j_pres, j_qty = jax_predict_sells(j_outputs["sell_presence_logits"],
                                      j_outputs["sell_quantity_log1p"])
    assert np.array_equal(t_pres.numpy() > 0.5,
                          np.asarray(j_pres) > 0.5)
    np.testing.assert_allclose(t_qty.numpy(), np.asarray(j_qty),
                               rtol=0.0, atol=MAX_ABS_TOL)


def test_loss_group_parity_e_variant():
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    model, params = build_pair(torch_config, jax_config, seed=52,
                               model_variant="E")
    econ, _, _, _ = economic_scenario_batch()
    b = econ.shape[0]
    inputs = make_inputs(batch_size=b, seed=53)
    inputs[ECONOMIC_CONTEXT_KEY] = torch.from_numpy(econ)
    targets = make_targets(batch_size=b, seed=54)
    with torch.no_grad():
        t_outputs = model(inputs)
    t_total, t_groups = torch_manager_loss(t_outputs, targets)
    j_outputs = jax_forward(params, to_numpy(inputs), jax_config,
                            model_variant="E")
    j_total, j_groups = jax_manager_loss(
        j_outputs, {k: v.numpy() for k, v in targets.items()})
    assert abs(float(t_total) - float(j_total)) <= MAX_ABS_TOL
    for name in t_groups:
        diff = abs(float(t_groups[name]) - float(j_groups[name]))
        print(f"E loss group {name}: diff={diff:.3e}")
        assert diff <= MAX_ABS_TOL, (name, diff)


def test_torch_checkpoint_e_round_trip_metadata_and_mismatch(tmp_path):
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    torch.manual_seed(55)
    model = DailyManagerTransformer(torch_config, model_variant="E")
    path = tmp_path / "best.pt"
    torch_save_checkpoint(path, kind="best", epoch=6, model=model,
                          model_config=torch_config,
                          training_config=TrainingConfig(),
                          validation_metrics={"total": 3.25},
                          model_variant="E")
    params, meta = load_torch_checkpoint(path, jax_config)
    assert meta["model_variant"] == "E"
    assert meta["model_config"] == {
        f.name: getattr(torch_config, f.name)
        for f in dataclasses.fields(TorchManagerConfig)}
    econ, _, _, _ = economic_scenario_batch()
    inputs = make_inputs(batch_size=econ.shape[0], seed=56)
    inputs[ECONOMIC_CONTEXT_KEY] = torch.from_numpy(econ)
    with torch.no_grad():
        t_outputs = model.eval()(inputs)
    assert_parity(t_outputs,
                  jax_forward(params, to_numpy(inputs), jax_config,
                              model_variant="E"))
    # Strict expected-variant checks, both directions.
    with pytest.raises(ValueError, match="stores model_variant 'E'"):
        load_torch_checkpoint(path, jax_config, model_variant="V0")
    v0_path = tmp_path / "v0.pt"
    torch.manual_seed(57)
    torch_save_checkpoint(v0_path, kind="best", epoch=1,
                          model=DailyManagerTransformer(torch_config),
                          model_config=torch_config,
                          training_config=TrainingConfig(),
                          validation_metrics={})
    with pytest.raises(ValueError, match="stores model_variant 'V0'"):
        load_torch_checkpoint(v0_path, jax_config, model_variant="E")
    # Old payloads without the top-level field still load as V0.
    payload = torch.load(v0_path, map_location="cpu", weights_only=True)
    legacy = {k: v for k, v in payload.items() if k != "model_variant"}
    _, legacy_meta = load_torch_checkpoint(legacy, jax_config)
    assert legacy_meta["model_variant"] == "V0"


def test_joint_variant_checkpoints_rejected_loudly(tmp_path):
    torch_config = torch_tiny_config()
    jax_config = jax_tiny_config()
    torch.manual_seed(58)
    model = DailyManagerTransformer(torch_config, model_variant="J")
    path = tmp_path / "joint.pt"
    torch_save_checkpoint(path, kind="best", epoch=1, model=model,
                          model_config=torch_config,
                          training_config=TrainingConfig(),
                          validation_metrics={}, model_variant="J")
    with pytest.raises(ValueError,
                       match="'J' is not supported by bc_manager_jax"):
        load_torch_checkpoint(path, jax_config)
    with pytest.raises(ValueError,
                       match="'JE' is not supported by bc_manager_jax"):
        state = DailyManagerTransformer(
            torch_config, model_variant="JE").state_dict()
        convert_torch_state_dict(state, jax_config, "JE")


def test_native_roundtrip_e_variant_and_legacy_v0_file(tmp_path):
    jax_config = jax_tiny_config()
    params = init_params(jax_config, seed=59, model_variant="E")
    path = tmp_path / "native_e.npz"
    save_native(path, params, jax_config, metadata={"epoch": 9},
                model_variant="E")
    loaded, meta = load_native(path, jax_config)
    assert meta["model_variant"] == "E"
    flat_a, _ = jax.tree_util.tree_flatten(params)
    flat_b, _ = jax.tree_util.tree_flatten(loaded)
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(flat_a, flat_b))
    with pytest.raises(ValueError, match="stores model_variant 'E'"):
        load_native(path, jax_config, model_variant="V0")

    # A V0 native file round-trips and rejects an E request.
    v0_params = init_params(jax_config, seed=60)
    v0_path = tmp_path / "native_v0.npz"
    save_native(v0_path, v0_params, jax_config)
    _, v0_meta = load_native(v0_path, jax_config)
    assert v0_meta["model_variant"] == "V0"
    with pytest.raises(ValueError, match="stores model_variant 'V0'"):
        load_native(v0_path, jax_config, model_variant="E")

    # A pre-variant native file (no model_variant record) loads as V0.
    legacy_path = tmp_path / "legacy.npz"
    with np.load(v0_path, allow_pickle=False) as archive:
        items = {key: archive[key] for key in archive.files}
    old_meta = json.loads(items["__meta__"].tobytes().decode("utf-8"))
    old_meta.pop("model_variant", None)
    items["__meta__"] = np.frombuffer(
        json.dumps(old_meta, sort_keys=True).encode("utf-8"),
        dtype=np.uint8)
    with open(legacy_path, "wb") as handle:
        np.savez(handle, **items)
    legacy_params, legacy_meta = load_native(legacy_path, jax_config)
    assert legacy_meta["model_variant"] == "V0"
    flat_c, _ = jax.tree_util.tree_flatten(v0_params)
    flat_d, _ = jax.tree_util.tree_flatten(legacy_params)
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(flat_c, flat_d))


@pytest.mark.skipif(not E_REAL_CHECKPOINT.exists(),
                    reason=f"real BC-E checkpoint absent at "
                           f"{E_REAL_CHECKPOINT}; copy it from Kaggle to "
                           f"run this bounded parity smoke")
def test_real_e_checkpoint_parity_if_present():
    """Bounded real-checkpoint smoke (skipped when the artifact is absent).

    Exact rerun command once artifacts/local/bc-v1-E/best.pt exists:
      python -m pytest tests/test_bc_manager_jax_parity.py::test_real_e_checkpoint_parity_if_present -q
    """
    params, meta = load_torch_checkpoint(E_REAL_CHECKPOINT)
    assert meta["model_variant"] == "E"
    config = JaxManagerConfig(**meta["model_config"])
    torch_model = DailyManagerTransformer(
        TorchManagerConfig(**meta["model_config"]),
        model_variant="E").eval()
    econ, _, _, _ = economic_scenario_batch()
    inputs = make_inputs(batch_size=econ.shape[0], seed=61)
    inputs[ECONOMIC_CONTEXT_KEY] = torch.from_numpy(econ)
    with torch.no_grad():
        t_outputs = torch_model(inputs)
    assert_parity(t_outputs,
                  jax_forward(params, to_numpy(inputs), config,
                              model_variant="E"))
