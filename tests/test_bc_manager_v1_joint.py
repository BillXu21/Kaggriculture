"""Stage-2 joint plan decoder tests (issue #6): V0/J/E/JE variant matrix,
state-dict compatibility, non-tautological cross-task coupling evidence,
non-causal single-pass proof, shared sell token, checkpoint round-trips,
and CLI/training plumbing. No real corpus or checkpoints required.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_bc_manager import _records  # noqa: E402
from test_bc_manager_v1_economics import (  # noqa: E402
    _e_batch,
    _v0_dummy_batch,
    write_records,
)

from bc_manager.economics import MODEL_VARIANTS, normalize_model_variant  # noqa: E402
from bc_manager.loss import ManagerLossConfig, manager_loss  # noqa: E402
from bc_manager.model import (  # noqa: E402
    DECISION_SELL_INDEX,
    DECISION_TOKEN_NAMES,
    DailyManagerTransformer,
    predict_counts,
    predict_land,
    predict_sells,
    tiny_manager_config,
)
from bc_manager.training import (  # noqa: E402
    TrainingConfig,
    load_checkpoint,
    load_model_from_checkpoint,
    run_training,
)
from bc_manager_jax.checkpoint import expected_torch_state_shapes  # noqa: E402

JOINT_KEYS = ("decision_tokens", "joint_layer")

# Pinned parameter counts (measured at implementation; regression pins).
TINY_PARAMS = {"V0": 37_008, "J": 39_328, "E": 37_232, "JE": 39_552}
DEFAULT_PARAMS = {"V0": 1_071_040, "J": 1_204_288, "E": 1_072_832,
                  "JE": 1_206_080}


def tiny_variant(variant: str) -> DailyManagerTransformer:
    torch.manual_seed(7)
    return DailyManagerTransformer(tiny_manager_config(),
                                   model_variant=variant)


def batch_for(variant: str, b: int = 2) -> dict:
    return _e_batch(b) if variant in ("E", "JE") else _v0_dummy_batch(b)


# ------------------------------------------------- 1. variant matrix contract


def test_variant_matrix_output_keys_shapes_and_finite_loss():
    reference = None
    for variant in MODEL_VARIANTS:
        model = tiny_variant(variant)
        outputs = model(batch_for(variant))
        keys = set(outputs)
        if reference is None:
            reference = {k: tuple(v.shape) for k, v in outputs.items()}
        assert keys == set(reference), variant
        for k, v in outputs.items():
            assert tuple(v.shape) == reference[k], (variant, k)
            assert torch.isfinite(v).all(), (variant, k)

    # Finite loss + backward through every variant.
    targets = {
        "crop_target": torch.randint(0, 20, (2, 5)),
        "animal_target": torch.randint(0, 20, (2, 3)),
        "land_count": torch.tensor([1, 4]),
        "fertilizer_target": torch.randint(0, 5, (2, 5)),
        "care_target": torch.randint(0, 3, (2, 3)),
        "sell_presence": (torch.rand(2, 9, 6) > 0.5).float(),
        "sell_quantity_log1p": torch.rand(2, 9, 6) * 4.0,
    }
    for variant in MODEL_VARIANTS:
        model = tiny_variant(variant)
        total, groups = manager_loss(model(batch_for(variant)), targets,
                                     ManagerLossConfig())
        assert torch.isfinite(total), variant
        assert set(groups) == set(
            ("crop", "animal", "land", "fertilizer", "care",
             "sell_presence", "sell_quantity"))
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters()), variant


def test_strict_economic_input_rules_per_variant():
    for variant in ("E", "JE"):
        with pytest.raises(ValueError, match="economic_context"):
            tiny_variant(variant)(_v0_dummy_batch(2))  # missing econ input
    for variant in ("V0", "J"):
        with pytest.raises(ValueError, match="unknown input keys"):
            tiny_variant(variant)(_e_batch(2))  # unexpected econ input


def test_decode_helpers_remain_contract_compatible_on_joint_outputs():
    outputs = tiny_variant("JE")(_e_batch(3))
    counts = predict_counts(outputs["crop_logits"])
    land = predict_land(outputs["land_logits"])
    presence, quantity = predict_sells(outputs["sell_presence_logits"],
                                       outputs["sell_quantity_log1p"])
    assert counts.shape == (3, 5) and land.shape == (3,)
    assert presence.shape == (3, 9, 6) and quantity.shape == (3, 9, 6)
    assert bool((land >= 1).all() and (land <= 4).all())
    assert bool((quantity >= 0).all())


# ------------------------------------- 2/3. state dicts and parameter pins


def test_v0_and_e_state_dicts_unchanged_from_stage1():
    v0 = tiny_variant("V0")
    e = tiny_variant("E")
    assert v0.trainable_parameters == TINY_PARAMS["V0"]
    assert e.trainable_parameters == TINY_PARAMS["E"]
    # No joint modules anywhere in V0/E state dicts.
    for model in (v0, e):
        assert not any(word in k for k in model.state_dict()
                       for word in JOINT_KEYS)
    # JAX V0 converter shape map still matches exactly.
    assert expected_torch_state_shapes(tiny_manager_config()) == \
        {k: tuple(v.shape) for k, v in v0.state_dict().items()}
    assert tuple(v0.state_dict()["global_encoders.self_resource.0.weight"]
                 .shape) == (16, 35)
    assert tuple(e.state_dict()["global_encoders.self_resource.0.weight"]
                 .shape) == (16, 49)


def test_default_config_parameter_pins():
    for variant in MODEL_VARIANTS:
        model = DailyManagerTransformer(model_variant=variant)
        assert model.trainable_parameters == DEFAULT_PARAMS[variant], variant


def test_joint_modules_present_only_in_j_je_with_pinned_deltas():
    for variant in ("J", "JE"):
        state = tiny_variant(variant).state_dict()
        assert "decision_tokens" in state
        assert tuple(state["decision_tokens"].shape) == \
            (1, len(DECISION_TOKEN_NAMES), 16)
        assert any(k.startswith("joint_layer.") for k in state)
    j_delta = TINY_PARAMS["J"] - TINY_PARAMS["V0"]
    je_delta = TINY_PARAMS["JE"] - TINY_PARAMS["E"]
    assert j_delta == je_delta == 2_320  # tokens 96 + one encoder layer 2224
    assert (DEFAULT_PARAMS["J"] - DEFAULT_PARAMS["V0"]) == \
        (DEFAULT_PARAMS["JE"] - DEFAULT_PARAMS["E"]) == 133_248


def test_old_style_v0_checkpoint_still_loads_exact(tmp_path):
    model = tiny_variant("V0")
    payload = {
        "format": "bc_manager_checkpoint_v1",
        "kind": "best",
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "model_config": {"d_model": 16, "num_layers": 1, "num_heads": 1,
                         "ffn_dim": 32, "dropout": 0.0, "count_max": 100,
                         "include_opponent_board": False},
        "training_config": {},
        "validation_metrics": {},
    }
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    loaded, loaded_payload = load_model_from_checkpoint(path)
    assert loaded_payload.get("model_variant", "V0") == "V0"
    batch = _v0_dummy_batch(1)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch), loaded(batch))


def test_j_je_checkpoint_round_trip_and_mismatch_errors(tmp_path):
    from bc_manager.training import save_checkpoint
    for variant in ("J", "JE"):
        model = tiny_variant(variant)
        ckpt = tmp_path / f"{variant}.pt"
        save_checkpoint(ckpt, kind="best", epoch=1, model=model,
                        model_config=tiny_manager_config(),
                        training_config=TrainingConfig(checkpoint_dir=None),
                        validation_metrics={}, model_variant=variant)
        payload = load_checkpoint(ckpt)
        assert payload["model_variant"] == variant
        assert "model_variant" not in payload["model_config"]
        loaded, _ = load_model_from_checkpoint(ckpt)
        assert loaded.model_variant == variant
        batch = batch_for(variant)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(batch), loaded(batch))
        # Wrong expected variant fails clearly.
        wrong = "V0" if variant == "J" else "E"
        with pytest.raises(ValueError, match="does not match"):
            load_model_from_checkpoint(ckpt, expected_variant=wrong)


# --------------------------------------------------- 4/5. coupling evidence


@pytest.fixture()
def coupled_model_and_reference():
    # dropout=0 tiny config, eval mode: deterministic coupling probe.
    torch.manual_seed(3)
    model = DailyManagerTransformer(tiny_manager_config(dropout=0.0),
                                    model_variant="J")
    model.eval()
    batch = _v0_dummy_batch(1)
    with torch.no_grad():
        reference = {k: v.clone() for k, v in model(batch).items()}
    return model, batch, reference


def test_perturbing_animal_token_changes_crop_logits_through_attention(
        coupled_model_and_reference):
    model, batch, reference = coupled_model_and_reference
    with torch.no_grad():
        model.decision_tokens[0, 1].add_(1.0)  # animal token only
        perturbed = model(batch)
    assert not torch.equal(perturbed["crop_logits"],
                           reference["crop_logits"])
    assert not torch.equal(perturbed["animal_logits"],
                           reference["animal_logits"])


def test_later_sell_token_changes_earlier_crop_logits_noncausal(
        coupled_model_and_reference):
    model, batch, reference = coupled_model_and_reference
    with torch.no_grad():
        model.decision_tokens[0, DECISION_SELL_INDEX].add_(1.0)
        perturbed = model(batch)
    # Under any causal mask a later token could never influence an earlier
    # one; crop (index 0) changing via sell (last index) proves the decoder
    # is bidirectional in one simultaneous pass.
    assert not torch.equal(perturbed["crop_logits"],
                           reference["crop_logits"])


def test_crop_gradient_flows_into_animal_decision_token(
        coupled_model_and_reference):
    model, batch, _ = coupled_model_and_reference
    model.decision_tokens.grad = None
    model(batch)["crop_logits"].sum().backward()
    grad = model.decision_tokens.grad
    assert grad is not None
    assert float(grad[0, 1].abs().sum()) > 0.0  # animal token got gradient
    assert float(grad[0, DECISION_SELL_INDEX].abs().sum()) > 0.0


def test_joint_layer_receives_all_six_tokens_with_no_mask(
        coupled_model_and_reference):
    model, batch, _ = coupled_model_and_reference
    captured: dict = {}

    def spy(module, inputs, output):
        # Default hook: `inputs` is the positional-arg tuple of the call.
        captured["src"] = inputs[0]
        captured["num_positional_args"] = len(inputs)

    handle = model.joint_layer.register_forward_hook(spy)
    try:
        model(batch)
    finally:
        handle.remove()
    assert captured["src"].shape[1] == len(DECISION_TOKEN_NAMES) == 6
    # Called with the source sequence only: no src_mask/attention-mask
    # arguments are passed at all (the bidirectional default), and the
    # later-token perturbation test above rules out causal behavior.
    assert captured["num_positional_args"] == 1


def test_shared_sell_token_feeds_both_sell_heads_identically(
        coupled_model_and_reference):
    model, batch, _ = coupled_model_and_reference
    seen: dict[str, torch.Tensor] = {}

    def make_spy(name):
        def spy(module, args, kwargs):
            seen[name] = args[0].detach().clone()
        return spy

    h1 = model.sell_presence_head.register_forward_hook(make_spy("presence"))
    h2 = model.sell_quantity_head.register_forward_hook(make_spy("quantity"))
    try:
        model(batch)
    finally:
        h1.remove()
        h2.remove()
    torch.testing.assert_close(seen["presence"], seen["quantity"])


def test_v0_has_no_decision_tokens_attribute():
    assert not hasattr(tiny_variant("V0"), "decision_tokens")
    assert not hasattr(tiny_variant("E"), "decision_tokens")


# ------------------------------------------------------- 7. CLI / training


def test_cli_accepts_all_four_variants():
    from bc_manager.cli import build_parser
    for variant in MODEL_VARIANTS:
        args = build_parser().parse_args(
            ["x.parquet", "--variant", variant, "--checkpoint-dir", "d"])
        assert args.variant == variant
    assert normalize_model_variant("je") == "JE"
    with pytest.raises(ValueError, match="model_variant"):
        normalize_model_variant("X")


@pytest.mark.parametrize("variant", ["J", "JE"])
def test_run_training_smoke_one_epoch_with_checkpoints(tmp_path, variant):
    path = write_records(_records(), tmp_path, "corpus.parquet")
    result = run_training(
        path, model_config=tiny_manager_config(),
        training_config=TrainingConfig(batch_size=2, epochs=1,
                                       checkpoint_dir=str(tmp_path / "ck")),
        train_dates=("2026-08-17", "2026-08-18", "2026-08-19"),
        val_dates=("2026-08-21",), min_score=2950.0, model_variant=variant,
        log=lambda _: None)
    assert result["model_variant"] == variant
    payload = load_checkpoint(tmp_path / "ck" / "best.pt")
    assert payload["model_variant"] == variant
    loaded, _ = load_model_from_checkpoint(tmp_path / "ck" / "best.pt")
    assert loaded.model_variant == variant
