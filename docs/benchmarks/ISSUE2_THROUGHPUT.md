# Issue #2 throughput benchmarks: official vs fast vs reference

Same-machine A/B on one Windows laptop host; deterministic scripted
actions only (no policy inference). The reference engine is the
unmodified diffmap/kaggicultureRL @ ef8bb3a build (provenance 1.32.6,
old shapes) and is a PERFORMANCE reference only.

## Environment

- os: `Windows 11 build 10.0.26200`
- cpu: `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`
- logical_processors: `20`
- python: `3.13.1`
- numpy: `2.5.2`
- kaggle_environments: `1.32.7`
- repo_commit: `63c8113585575fd6c3edf1417795eb553b44ddae`

## Scalar full episodes (episodeSteps=720)

| engine:trace | median s | min-max s | turns/s | episodes/s | accepted step calls |
|---|---|---|---|---|---|
| official:official:pass_only | 1.320 | 1.318-1.356 | 545.4 | 0.758 | 719 |
| official:official:mixed | 1.302 | 1.286-1.329 | 552.8 | 0.768 | 719 |
| fast:fast_api:pass_only | 0.296 | 0.275-0.307 | 2,428.4 | 3.373 | 719 |
| fast:fast_api:mixed | 0.279 | 0.265-0.287 | 2,580.4 | 3.584 | 719 |
| fast:fast_native:mixed | 0.004 | 0.004-0.008 | 188,365.8 | 261.619 | 719 |
| reference:reference_native:mixed | 0.001 | 0.001-0.002 | 499,393.1 | 693.602 | 719 |

**Scalar speedup vs official (mixed trace, full API incl. dict decode): 4.7x.** Step-call accounting: 719 accepted step calls per 720-step episode (720 competition turns incl. the reset position).

## Batch throughput (env-transitions/sec, steady step_into)

| engine | N | threads | transitions/s median | per-step ms | GIL released |
|---|---|---|---|---|---|
| fast | 1 | 1 | 85,609.1 | 0.01 | yes |
| fast | 1 | 2 | 72,960.7 | 0.01 | yes |
| fast | 1 | 4 | 86,214.3 | 0.01 | yes |
| fast | 1 | default | 98,357.4 | 0.01 | yes |
| fast | 16 | 1 | 85,998.4 | 0.19 | yes |
| fast | 16 | 2 | 87,255.8 | 0.18 | yes |
| fast | 16 | 4 | 87,845.5 | 0.18 | yes |
| fast | 16 | default | 91,258.0 | 0.18 | yes |
| fast | 128 | 1 | 60,320.6 | 2.12 | yes |
| fast | 128 | 2 | 114,482.0 | 1.12 | yes |
| fast | 128 | 4 | 204,295.3 | 0.63 | yes |
| fast | 128 | default | 185,295.6 | 0.69 | yes |
| fast | 512 | 1 | 58,350.1 | 8.77 | yes |
| fast | 512 | 2 | 91,315.9 | 5.61 | yes |
| fast | 512 | 4 | 133,036.7 | 3.85 | yes |
| fast | 512 | default | 167,265.2 | 3.06 | yes |
| fast | 1024 | 1 | 59,615.5 | 17.18 | yes |
| fast | 1024 | 2 | 96,043.8 | 10.66 | yes |
| fast | 1024 | 4 | 132,790.5 | 7.71 | yes |
| fast | 1024 | default | 172,578.8 | 5.93 | yes |
| reference | 1 | default | 256,410.2 | 0.00 | no |
| reference | 16 | default | 529,853.9 | 0.03 | no |
| reference | 128 | default | 240,808.4 | 0.53 | no |
| reference | 512 | default | 310,538.7 | 1.65 | no |
| reference | 1024 | default | 282,802.0 | 3.62 | no |

## Multi-core scaling efficiency (vs 1 thread, same N)

Parallel fan-out engages at N >= 128 (PARALLEL_MIN_ENVS); below that
both engines run a serial loop regardless of pool configuration.

| N | 2T eff | 4T eff | default eff |
|---|---|---|---|
| 1 | 0.85x | 1.01x | 1.15x |
| 16 | 1.01x | 1.02x | 1.06x |
| 128 | 1.90x | 3.39x | 3.07x |
| 512 | 1.56x | 2.28x | 2.87x |
| 1024 | 1.61x | 2.23x | 2.89x |

## Fast vs reference (matched N, closest available modes)

| N | reference (global pool) t/s | fast best t/s | ratio |
|---|---|---|---|
| 1 | 256,410.2 | 98,357.4 | 0.38x |
| 16 | 529,853.9 | 91,258.0 | 0.17x |
| 128 | 240,808.4 | 204,295.3 | 0.85x |
| 512 | 310,538.7 | 167,265.2 | 0.54x |
| 1024 | 282,802.0 | 172,578.8 | 0.61x |

## Phase split at N=512 (transition vs observation construction)

- official: no transition/observation phase seam (pure Python interpreter)
- fast: transition-only 1,163,480 t/s, observe-only 224,597 t/s, observation share 83.9%
- reference: transition-only 1,429,233 t/s, observe-only 388,534 t/s, observation share 75.3%

## Memory

- official: N=1: 0 B/env RSS delta
- fast: theoretical obs buffer 70,128 B/env; action tensor 12,048 B/env; N=128: 107,328 B/env RSS delta; N=512: 65,320 B/env RSS delta
- reference: theoretical obs buffer 45,040 B/env; action tensor 1,296 B/env; N=128: 54,912 B/env RSS delta; N=512: 21,024 B/env RSS delta

RSS deltas include allocator arenas and Rayon pool threads; they
upper-bound the true per-env state cost and are not GameState sizes.

## Profile findings

<details><summary>official scalar cProfile (top cumulative)</summary>

```text
6568779 function calls (5361439 primitive calls) in 3.871 seconds

   Ordered by: cumulative time
   List reduced from 354 to 14 due to restriction <14>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.007    0.007    3.876    3.876 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\scripts\benchmark_engine_throughput.py:565(<lambda>)
        1    0.004    0.004    3.869    3.869 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\scripts\benchmark_engine_throughput.py:297(run_scalar_episode_official)
      719    0.006    0.000    3.699    0.005 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\oracle\official_backend.py:68(step)
      719    0.019    0.000    3.690    0.005 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\kaggle_environments\core.py:256(step)
     1448    0.008    0.000    1.826    0.001 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\kaggle_environments\utils.py:187(process_schema)
1148784/4335    1.097    0.000    1.705    0.000 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\kaggle_environments\utils.py:117(structify)
      721    0.002    0.000    1.684    0.002 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\kaggle_environments\core.py:681(__run_interpreter)
      721    0.019    0.000    1.682    0.002 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\kaggle_environments\core.py:640(__run_interpreter_prod)
      721    0.016    0.000    1.652    0.002 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\kaggle_environments\core.py:621(__loop_through_interpreter)
     1448    0.012    0.000    1.632    0.001 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\jsonschema\validators.py:1265(validate)
     1448    0.006    0.000    1.555    0.001 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\jsonschema\validators.py:306(check_schema)
     2896    0.008    0.000    1.486    0.001 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\jsonschema\validators.py:349(iter_errors)
1584/1448    0.014    0.000    1.448    0.001 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\jsonschema\_keywords.py:332(allOf)
31985/10202    0.129    0.000    1.438    0.000 C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv\Lib\site-packages\jsonschema\validators.py:396(descend)
```
</details>

<details><summary>fast scalar cProfile (top cumulative)</summary>

```text
511984 function calls (509108 primitive calls) in 0.573 seconds

   Ordered by: cumulative time
   List reduced from 46 to 14 due to restriction <14>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.573    0.573 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\scripts\benchmark_engine_throughput.py:567(<lambda>)
        1    0.004    0.004    0.573    0.573 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\scripts\benchmark_engine_throughput.py:311(run_scalar_episode_fast_wrapper)
      719    0.008    0.000    0.557    0.001 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:333(step)
      720    0.006    0.000    0.513    0.001 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:321(_decode)
     1440    0.278    0.000    0.507    0.000 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:227(_decode_observation)
   288000    0.139    0.000    0.139    0.000 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:191(_tile)
    89856    0.040    0.000    0.060    0.000 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:187(_round)
     2880    0.026    0.000    0.048    0.000 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:223(_inventory)
      719    0.010    0.000    0.023    0.000 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:162(_encode_actions)
    89856    0.020    0.000    0.020    0.000 {built-in method builtins.round}
      719    0.014    0.000    0.014    0.000 {method 'step' of 'builtins.RustBatchEnv' objects}
        1    0.008    0.008    0.008    0.008 C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\fast_env\api.py:282(__init__)
6055/3179    0.002    0.000    0.007    0.000 {built-in method builtins.isinstance}
     1438    0.001    0.000    0.005    0.000 C:\Users\liuyi\AppData\Local\Programs\Python\Python313\Lib\typing.py:1374(__instancecheck__)
```
</details>

<details><summary>fast batch N=512 cProfile (top tottime)</summary>

```text
51 function calls in 0.163 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       50    0.163    0.003    0.163    0.003 {method 'step_into' of 'builtins.RustBatchEnv' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```
</details>

<details><summary>reference batch N=512 cProfile (top tottime)</summary>

```text
51 function calls in 0.091 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       50    0.091    0.002    0.091    0.002 {method 'step_into' of 'builtins.RustBatchEnv' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```
</details>

## Findings and optimization decision

- Scalar dict API: 4.7x faster than official per full episode. The native floor is 341x, so Python-side action encoding + observation decoding dominates the wrapper cost.
- Profile finding: writing observations is 84% of steady step_into cost at N=512 (224,597 obs t/s vs 1,163,480 transition-only t/s).
- The unmodified reference core is 2.7x faster per env-transition than ours at N=1. The dominant suspect is the observation/mask writer over the exact-contract MAX_HANDS=240 layout (reference uses the old 16-hand shapes), consistent with the phase split above.
- Decision: no engine change in this stage. Observation-writer cost is the one clear, bounded optimization candidate (e.g. skipping untouched hand blocks or a fused day-step pass); it alters no semantics and belongs to a distinct correction stage. The current engine already beats the official interpreter decisively and scales ~2.9x on the default pool at N>=512, which covers the planned rollout topology; optimizing further now would be vanity tuning.

## Caveats

- Laptop host (hybrid P/E cores, 20 logical processors); absolute
  numbers do not transfer to Kaggle TPU hosts and no TPU claim is made.
- The reference engine pins 1.32.6 semantics with old shapes; ratios
  are performance context, not parity statements.
- Cold-import numbers include interpreter startup; treat them as
  registry/extension-load cost, not per-step cost.
- All medians come from repeated warm runs; cold/warm numbers are
  never mixed in a single figure.
