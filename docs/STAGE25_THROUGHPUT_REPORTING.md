# Stage 2.5 throughput reporting

`python -m rl_manager.stage25_ppo_cli` prints one compact report after each
completed update and appends the complete record to `metrics.jsonl` in the
selected output directory. `--json-stdout` additionally writes that complete
record to stdout for consumers that need streaming JSON.

The update wall-clock interval starts immediately before collection and ends
after checkpoint serialization/save. Accounted time is collection, PPO, and
checkpoint time; the reported overhead is the remaining interval, including
the existing second trajectory-to-batch construction in the CLI. Reporting
itself occurs after the wall-clock boundary.

Inference and queue durations are aggregate counters from the parallel
runner. They are labeled as aggregate metrics and must not be interpreted as
update wall time.

JAX compilation is lazy. If `--jax-compilation-cache-dir PATH` is supplied,
the CLI configures the installed persistent compilation-cache API before
state initialization and sets the compile-time and entry-size thresholds to
zero. Update 1 should be treated as warmup/compilation; use updates 2 and
later for steady-state comparisons.
