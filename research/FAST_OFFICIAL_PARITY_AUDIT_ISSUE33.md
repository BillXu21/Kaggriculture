# Issue #33 Fast/Official Parity Audit

Date: 2026-08-30

## Result

The current fast engine has **no observed first semantic divergence** from the
pinned official `kaggle_environments==1.32.7` engine in this bounded audit.
Reset underlying state and policy-visible observations matched for seeds
`7, 17, 42, 123`. Eight seat-swapped current BC-E versus PASS traces and one
current BC-E versus current BC-E trace each ran all 719 accepted post-reset
transitions to canonical terminal step 719 with exact state, observation,
action, reward, and status parity.

The existing legal-ish corpus was rerun after rebuilding the native extension:
seeds `0, 1, 2, 7, 17, 42, 123, 999`, 5,752 action pairs, 33 action families,
29 day boundaries per episode, and zero first divergence. This includes the
market, worker, hiring, overflow, crop, animal, CARE/FERTILIZE, weeds, shops,
RNG/day-refresh, terminal, and malformed/no-op surfaces covered by the
existing generator.

The repository's episode convention is one reset observation plus 719 accepted
steps. The official backend is terminal after that final accepted action, so a
720th post-reset primitive action cannot be submitted; artifacts record this
explicitly rather than pretending a 720th transition was tested.

## Identity

- Branch: `codex/issue-33-fast-official-parity-audit`
- Audit anchor: `57db920689ef80fb373128d4b8129054816d133f`
- Rust source commit: `63c8113585575fd6c3edf1417795eb553b44ddae`
- Native extension: `fast_env/_kaggriculture_env.cp313-win_amd64.pyd`
- Native SHA-256: `c71fc02cd7acbce2c2cc8a950f894311bcc7b2b3385880fe26e7f49f99a64ffa`
- Official package: `kaggle_environments==1.32.7`
- Official upstream commit: `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`
- Official wheel SHA-256: `2a1bb862ad2d6463080f80f6a766f46d94b53fd57168cfeddb9857fc3dbc4c8f`
- Checkpoint: external `bc-v1-E/best.pt`, SHA-256
  `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`.

## Commands

```text
maturin develop --release
python scripts/audit_fast_official_parity.py --checkpoint .../bc-v1-E/best.pt --seed 17 --candidate-seat 0
python scripts/audit_fast_official_parity.py --checkpoint .../bc-v1-E/best.pt --seed 17 --candidate-seat 0 --opponent current_control --output-dir .../active_current_vs_current_seed_17
python scripts/run_parity_corpus.py
python -m pytest tests/test_fast_env.py tests/test_oracle_offline.py tests/test_oracle_replay.py tests/test_oracle_import_isolation.py tests/test_oracle_mechanics.py tests/test_oracle_crops.py tests/test_oracle_animals.py tests/test_oracle_hands.py tests/test_oracle_town_world.py tests/test_fast_official_parity_audit.py -q
```

The audit used the known official interpreter under
`data/temp/official-1327-venv`. No JAX/JAXLIB import is part of the native
parity or corpus commands.

## Conclusion

Machine-readable traces and summaries are local and ignored under
`artifacts/local/fast_official_parity/`. No first-divergence artifact was
produced because no mismatch occurred. No root cause, regression, or
fast-engine fix was identified, so no Rust semantic code was changed and
there is no fix commit.

The fast environment is **provisionally training-safe for the tested paths**,
not full tested parity for every reachable state and not a reason to use fast
score magnitude as promotion evidence.
