"""One-process multi-trainer JAX/TPU prototype benchmark (issue #21).

This module deliberately does not start rollout workers. One Python process
owns every JAX/libtpu object and assigns trainer ``i`` to
``jax.devices()[i]``. The benchmark uses synthetic manager-day rows and the
same compiled forward/PPO minibatch primitives used by the RL manager.

Example, one process and four independent trainers:

    python -m rl_manager.multitrainer_benchmark \
        --checkpoint /kaggle/working/bc-v1-E/best.pt \
        --trainer-counts 4 --batch-size 256 --warmup 3 --iterations 10 \
        --output-json /kaggle/working/issue21_multitrainer_n4.json

CPU results are placement/plumbing evidence only. This module makes no TPU
scaling claim; the Kaggle run must record the emitted JSON files.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager_jax.benchmark import synthetic_batch
from bc_manager_jax.checkpoint import load_torch_checkpoint
from bc_manager_jax.model import (
    ManagerConfig,
    _forward_eval_with_representation,
    _prepare_inputs,
    init_params,
    resolve_model_variant,
    tiny_manager_config,
)
from rl_manager.ppo import PPOBatch, PPOTrainState, _compiled_update, init_train_state
from rl_manager.ppo_policy import PPOConfig, action_index_tensors
from rl_manager.policy import params_fingerprint
from rl_manager.types import PolicyIdentity


@dataclass(frozen=True)
class BenchmarkOptions:
    """Validated options for one bounded benchmark matrix."""

    trainer_counts: tuple[int, ...] = (1, 2, 4)
    model_config: str = "tiny"
    variant: str = "E"
    checkpoint: str | None = None
    batch_size: int = 8
    warmup: int = 1
    iterations: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.trainer_counts or any(
                isinstance(count, bool) or count < 1
                for count in self.trainer_counts):
            raise ValueError("trainer_counts must contain positive integers")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.warmup < 0 or self.iterations < 1:
            raise ValueError("warmup must be >= 0 and iterations must be >= 1")


@dataclass
class _TrainerRuntime:
    """All device-resident arguments for one independent trainer."""

    trainer_id: int
    identity: PolicyIdentity
    device: Any
    state: PPOTrainState
    inputs: Mapping[str, Any]
    indices: Mapping[str, Any]
    old_logprob: Any
    advantages: Any
    returns: Any
    inference_fn: Any
    update_fn: Any
    last_inference: Any = None

    def infer(self) -> Any:
        self.last_inference = self.inference_fn(self.state.params["base"],
                                                self.inputs)
        return self.last_inference

    def update(self) -> Any:
        params, opt_state, metrics = self.update_fn(
            self.state.params, self.state.opt_state, self.state.frozen_params,
            self.inputs, self.indices, self.old_logprob, self.advantages,
            self.returns)
        self.state = dataclasses.replace(
            self.state, params=params, opt_state=opt_state,
            step=self.state.step + 1)
        return params, opt_state, metrics


def _block(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()


def _device_label(device: Any) -> str:
    return f"{device.platform}:{device.id}:{device.device_kind}"


def _placement(tree: Any) -> dict[str, Any]:
    """Summarize actual host/device placement without materializing arrays."""
    devices: set[str] = set()
    host_leaves = 0
    device_leaves = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf_devices = None
        devices_method = getattr(leaf, "devices", None)
        if callable(devices_method):
            leaf_devices = devices_method()
        elif hasattr(leaf, "device") and hasattr(leaf.device, "platform"):
            leaf_devices = (leaf.device,)
        if leaf_devices:
            device_leaves += 1
            devices.update(_device_label(device) for device in leaf_devices)
        else:
            host_leaves += 1
    return {
        "devices": sorted(devices),
        "device_leaf_count": device_leaves,
        "host_leaf_count": host_leaves,
        "placement": "host" if host_leaves and not devices
        else ("device" if devices and not host_leaves else "mixed"),
    }


def _device_put_tree(tree: Any, device: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf, device),
                                  tree)


def _require_device(tree: Any, device: Any, what: str) -> None:
    expected = _device_label(device)
    placement = _placement(tree)
    if placement["host_leaf_count"] or placement["devices"] != [expected]:
        raise RuntimeError(
            f"{what} was not placed exclusively on {expected}: {placement}")


def _time_stateless(fn: Any, *, warmup: int,
                    iterations: int) -> tuple[float, list[float], Any]:
    started = time.perf_counter()
    result = fn()
    _block(result)
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        _block(fn())
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = fn()
        _block(result)
        samples.append(time.perf_counter() - started)
    return compile_seconds, samples, result


def _time_sequential(trainers: list[_TrainerRuntime], *, warmup: int,
                     iterations: int) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        for trainer in trainers:
            _block(trainer.update())
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        for trainer in trainers:
            _block(trainer.update())
        samples.append(time.perf_counter() - started)
    return samples, [sample / len(trainers) for sample in samples]


def _time_concurrent(trainers: list[_TrainerRuntime], *, warmup: int,
                     iterations: int) -> tuple[list[float], list[float]]:
    def dispatch_and_block() -> None:
        outputs = [trainer.update() for trainer in trainers]
        _block(outputs)

    for _ in range(warmup):
        dispatch_and_block()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        dispatch_and_block()
        samples.append(time.perf_counter() - started)
    return samples, [sample / len(trainers) for sample in samples]


def _timing_summary(samples: list[float], trainer_count: int) -> dict[str, Any]:
    return {
        "round_seconds": samples,
        "mean_round_seconds": float(np.mean(samples)),
        "best_round_seconds": float(np.min(samples)),
        "mean_seconds_per_trainer": float(np.mean(samples) / trainer_count),
        "trainers_per_second": float(trainer_count / np.mean(samples)),
    }


def _synthetic_ppo_batch(config: ManagerConfig, batch_size: int, *,
                        seed: int, variant: str) -> PPOBatch:
    inputs, targets = synthetic_batch(config, batch_size, seed=seed,
                                      model_variant=variant)
    sell_presence = np.asarray(targets["sell_presence"], dtype=np.uint8)
    quantity = np.floor(
        np.expm1(np.clip(targets["sell_quantity_log1p"], 0.0, None)) + 0.5)
    actions = {
        "crop": np.asarray(targets["crop_target"], dtype=np.int16),
        "animal": np.asarray(targets["animal_target"], dtype=np.int16),
        "land": np.asarray(targets["land_count"], dtype=np.int16),
        "fertilizer": np.asarray(targets["fertilizer_target"], dtype=np.int16),
        "care": np.asarray(targets["care_target"], dtype=np.int16),
        "sell_presence": sell_presence,
        "sell_quantity": quantity.astype(np.int16),
    }
    return PPOBatch(
        inputs={key: np.ascontiguousarray(value)
                for key, value in inputs.items()},
        action_tensors={key: np.ascontiguousarray(value)
                        for key, value in actions.items()},
        old_logprob=np.zeros(batch_size, dtype=np.float32),
        advantages=np.linspace(-1.0, 1.0, batch_size, dtype=np.float32),
        returns=np.linspace(-0.5, 0.5, batch_size, dtype=np.float32),
        values=np.zeros(batch_size, dtype=np.float32),
    )


def _environment_metadata() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "platform": platform.platform(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "visible_device_count": len(devices),
        "visible_devices": [
            {"label": _device_label(device), "id": int(device.id),
             "platform": device.platform, "kind": device.device_kind}
            for device in devices
        ],
        "ownership": "one Python process owns JAX/libtpu; no rollout workers",
        "honesty_note": "CPU values are plumbing only; TPU scaling is unmeasured",
    }


def _load_model(options: BenchmarkOptions) -> tuple[ManagerConfig, dict, str,
                                                       str]:
    if options.checkpoint:
        path = Path(options.checkpoint)
        if not path.exists():
            raise ValueError(f"checkpoint does not exist: {path}")
        params, metadata = load_torch_checkpoint(path)
        config = ManagerConfig(**metadata["model_config"])
        variant = resolve_model_variant(metadata.get("model_variant", "V0"))
        return config, params, variant, f"checkpoint:{path}"
    config = (tiny_manager_config() if options.model_config == "tiny"
              else ManagerConfig())
    variant = resolve_model_variant(options.variant)
    params = init_params(config, seed=options.seed, model_variant=variant)
    return config, params, variant, f"random-init:{options.model_config}"


def _make_trainer(config: ManagerConfig, base_params: Mapping, batch: PPOBatch,
                  *, trainer_id: int, device: Any, variant: str,
                  ppo_config: PPOConfig) -> _TrainerRuntime:
    state = init_train_state(base_params, config, seed=17 + trainer_id,
                             ppo_config=ppo_config, model_variant=variant)
    state = dataclasses.replace(
        state,
        params=_device_put_tree(state.params, device),
        opt_state=_device_put_tree(state.opt_state, device),
        frozen_params=_device_put_tree(state.frozen_params, device),
        rng=jax.device_put(state.rng, device))
    inputs = _device_put_tree(
        _prepare_inputs({key: jnp.asarray(value)
                         for key, value in batch.inputs.items()}), device)
    indices = _device_put_tree(
        action_index_tensors(batch.action_tensors, batch.size), device)
    old_logprob = jax.device_put(jnp.asarray(batch.old_logprob), device)
    advantages = jax.device_put(jnp.asarray(batch.advantages), device)
    returns = jax.device_put(jnp.asarray(batch.returns), device)
    _require_device(state.params, device, "trainer params")
    _require_device(state.opt_state, device, "optimizer state")
    _require_device(state.frozen_params, device, "frozen params")
    _require_device(inputs, device, "PPO inputs")
    _require_device(indices, device, "PPO action arrays")

    inference_fn = jax.jit(
        lambda params, prepared: _forward_eval_with_representation(
            params, prepared, config, variant))
    core = _compiled_update(config, ppo_config, variant)
    update_fn = core
    identity = PolicyIdentity(
        name=f"ppo_trainer_{trainer_id}", version="issue-21-prototype",
        fingerprint=params_fingerprint(state.params))
    return _TrainerRuntime(
        trainer_id=trainer_id, identity=identity, device=device, state=state,
        inputs=inputs, indices=indices, old_logprob=old_logprob,
        advantages=advantages, returns=returns, inference_fn=inference_fn,
        update_fn=update_fn)


def _baseline_diagnostic(config: ManagerConfig, base_params: Mapping,
                         batch: PPOBatch, *, variant: str,
                         ppo_config: PPOConfig) -> dict[str, Any]:
    """Show placement of the pre-existing unsharded single-trainer path."""
    state = init_train_state(base_params, config, seed=17,
                             ppo_config=ppo_config, model_variant=variant)
    host_inputs = _prepare_inputs({key: jnp.asarray(value)
                                   for key, value in batch.inputs.items()})
    infer = jax.jit(lambda params, prepared:
                    _forward_eval_with_representation(
                        params, prepared, config, variant))
    started = time.perf_counter()
    output = infer(state.params["base"], host_inputs)
    _block(output)
    return {
        "params": _placement(state.params),
        "optimizer_state": _placement(state.opt_state),
        "rng": _placement(state.rng),
        "inference_inputs": _placement(host_inputs),
        "inference_outputs": _placement(output),
        "ppo_arrays_host": _placement({
            "inputs": batch.inputs, "actions": batch.action_tensors,
            "old_logprob": batch.old_logprob, "advantages": batch.advantages,
            "returns": batch.returns}),
        "first_inference_seconds": time.perf_counter() - started,
        "device_count_used_by_params": len(
            _placement(state.params)["devices"]),
        "interpretation": "default single-trainer path is unsharded unless "
                           "this placement reports otherwise",
    }


def run_case(options: BenchmarkOptions, trainer_count: int) -> dict[str, Any]:
    """Run one N-trainer case, or return an explicit unavailable-device row."""
    if isinstance(trainer_count, bool) or not isinstance(trainer_count, int) \
            or trainer_count < 1:
        raise ValueError(f"trainer_count must be a positive int, got {trainer_count!r}")
    visible = jax.devices()
    row: dict[str, Any] = {
        "trainer_count": trainer_count,
        "batch_size_per_trainer": options.batch_size,
        "total_logical_batch": trainer_count * options.batch_size,
        "status": "ok",
        "reason": None,
        "trainer_devices": [],
        "sequential": None,
        "concurrent_dispatch": None,
    }
    if trainer_count > len(visible):
        row["status"] = "skipped"
        row["reason"] = (
            f"requested {trainer_count} trainers but only {len(visible)} "
            "JAX devices are visible; refusing device reuse")
        return row

    config, base_params, variant, source = _load_model(options)
    ppo_config = PPOConfig(minibatch_size=options.batch_size, epochs=1)
    batch = _synthetic_ppo_batch(
        config, options.batch_size, seed=options.seed + trainer_count,
        variant=variant)
    trainers = [
        _make_trainer(config, base_params, batch, trainer_id=index,
                      device=visible[index], variant=variant,
                      ppo_config=ppo_config)
        for index in range(trainer_count)
    ]
    row.update({
        "source": source,
        "model_variant": variant,
        "token_count": config.token_count,
        "param_count": int(sum(np.asarray(leaf).size for leaf in
                                jax.tree_util.tree_leaves(base_params))),
        "model_config": dataclasses.asdict(config),
        "explicit_assignment": {
            str(trainer.trainer_id): _device_label(trainer.device)
            for trainer in trainers},
        "trainer_devices": [_device_label(trainer.device)
                            for trainer in trainers],
        "trainer_identities": [trainer.identity.to_json_dict()
                               for trainer in trainers],
        "placements": {
            "trainers": [
                {
                    "trainer_id": trainer.trainer_id,
                    "assigned_device": _device_label(trainer.device),
                    "params": _placement(trainer.state.params),
                    "optimizer_state": _placement(trainer.state.opt_state),
                    "rng": _placement(trainer.state.rng),
                    "inference_inputs": _placement(trainer.inputs),
                    "ppo_action_arrays": _placement(trainer.indices),
                    "ppo_scalar_arrays": _placement({
                        "old_logprob": trainer.old_logprob,
                        "advantages": trainer.advantages,
                        "returns": trainer.returns}),
                }
                for trainer in trainers
            ],
        },
    })
    if trainer_count == 1:
        row["default_single_trainer_diagnostic"] = _baseline_diagnostic(
            config, base_params, batch, variant=variant,
            ppo_config=ppo_config)

    inference_compile: list[float] = []
    inference_samples: list[list[float]] = []
    for trainer in trainers:
        compile_seconds, samples, output = _time_stateless(
            trainer.infer, warmup=options.warmup,
            iterations=options.iterations)
        _require_device(output, trainer.device, "inference outputs")
        inference_compile.append(compile_seconds)
        inference_samples.append(samples)
    row["inference"] = {
        "compile_seconds_by_trainer": inference_compile,
        "compile_seconds_max": max(inference_compile),
        "steady_seconds_by_trainer": inference_samples,
        "steady_mean_seconds": float(np.mean(inference_samples)),
        "steady_examples_per_second": float(
            trainer_count * options.batch_size / np.mean(inference_samples)),
        "outputs": _placement(trainers[0].last_inference),
    }

    update_compile: list[float] = []
    for trainer in trainers:
        started = time.perf_counter()
        output = trainer.update()
        _block(output)
        update_compile.append(time.perf_counter() - started)
        _require_device(trainer.state.params, trainer.device,
                        "updated trainer params")
        _require_device(trainer.state.opt_state, trainer.device,
                        "updated optimizer state")
    row["ppo_update"] = {
        "compile_seconds_by_trainer": update_compile,
        "compile_seconds_max": max(update_compile),
        "output_metrics": _placement(output[2]),
        "updated_params": _placement(trainers[0].state.params),
        "updated_optimizer_state": _placement(trainers[0].state.opt_state),
    }

    sequential_rounds, sequential_per_trainer = _time_sequential(
        trainers, warmup=options.warmup, iterations=options.iterations)
    concurrent_rounds, concurrent_per_trainer = _time_concurrent(
        trainers, warmup=options.warmup, iterations=options.iterations)
    row["sequential"] = _timing_summary(sequential_rounds, trainer_count)
    row["sequential"]["steady_seconds_per_trainer"] = sequential_per_trainer
    row["concurrent_dispatch"] = _timing_summary(concurrent_rounds,
                                                  trainer_count)
    row["concurrent_dispatch"]["steady_seconds_per_trainer"] = \
        concurrent_per_trainer
    row["placement_after_timing"] = {
        "params": _placement(trainers[0].state.params),
        "optimizer_state": _placement(trainers[0].state.opt_state),
        "inference_outputs": _placement(trainers[0].last_inference),
    }
    return row


def run_benchmark(options: BenchmarkOptions) -> dict[str, Any]:
    """Run requested trainer counts in this one Python process."""
    metadata = _environment_metadata()
    metadata["requested_trainer_counts"] = list(options.trainer_counts)
    results = []
    for count in options.trainer_counts:
        print(f"[issue21] trainers={count} ...", flush=True)
        result = run_case(options, count)
        results.append(result)
        print(f"    {result['status']}: {result.get('reason') or 'complete'}",
              flush=True)
    return {"metadata": metadata, "results": results}


def _write_json(report: dict[str, Any], path: str | None) -> None:
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[issue21] JSON written to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_manager.multitrainer_benchmark",
        description="One-process independent JAX PPO trainer placement and "
                    "dispatch benchmark; never starts rollout workers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Kaggle TPU v5-8 examples, each command is ONE Python process:

  python -m rl_manager.multitrainer_benchmark --checkpoint \\
      /kaggle/working/bc-v1-E/best.pt --trainer-counts 1 \\
      --batch-size 256 --warmup 3 --iterations 10 \\
      --output-json /kaggle/working/issue21_multitrainer_n1.json
  python -m rl_manager.multitrainer_benchmark --checkpoint \\
      /kaggle/working/bc-v1-E/best.pt --trainer-counts 2 \\
      --batch-size 256 --warmup 3 --iterations 10 \\
      --output-json /kaggle/working/issue21_multitrainer_n2.json
  python -m rl_manager.multitrainer_benchmark --checkpoint \\
      /kaggle/working/bc-v1-E/best.pt --trainer-counts 4 \\
      --batch-size 256 --warmup 3 --iterations 10 \\
      --output-json /kaggle/working/issue21_multitrainer_n4.json
  # Optional only if N=4 is clean and memory remains healthy:
  python -m rl_manager.multitrainer_benchmark --checkpoint \\
      /kaggle/working/bc-v1-E/best.pt --trainer-counts 8 \\
      --batch-size 256 --warmup 3 --iterations 10 \\
      --output-json /kaggle/working/issue21_multitrainer_n8.json

Do not launch these commands concurrently. Do not interpret CPU output as TPU
scaling. Use --trainer-counts 1,2,4 in one invocation for a single-process
matrix when preferred.""")
    parser.add_argument("--trainer-counts", default="1,2,4",
                        help="comma-separated N values (default: %(default)s)")
    parser.add_argument("--model-config", choices=("tiny", "default"),
                        default="tiny")
    parser.add_argument("--variant", choices=("V0", "E"), default="E")
    parser.add_argument("--checkpoint", default=None,
                        help="real PyTorch BC checkpoint; stored config/variant "
                             "are authoritative")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="synthetic manager rows per trainer")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    return parser


def options_from_args(args: argparse.Namespace) -> BenchmarkOptions:
    counts = tuple(int(item.strip()) for item in args.trainer_counts.split(",")
                   if item.strip())
    return BenchmarkOptions(
        trainer_counts=counts, model_config=args.model_config,
        variant=args.variant, checkpoint=args.checkpoint,
        batch_size=args.batch_size, warmup=args.warmup,
        iterations=args.iterations, seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(options_from_args(args))
    report["metadata"]["cli_args"] = vars(args)
    _write_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
