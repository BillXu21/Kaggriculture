# Fast engine provenance

## Oracle

- Official source: `Kaggle/kaggle-environments`.
- Pinned commit: `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.
- Package contract: `kaggle-environments==1.32.7`.
- Source: `kaggle_environments/envs/kaggriculture/kaggriculture.py`.
- Protocol/specification: `kaggle_environments/envs/kaggriculture/kaggriculture.json`.
- Verified 1.32.6 parent: `2f22627ff7def9fc23d57bebcab184476153c678`.
- Official delta verified: only `CARROT`, `TOMATO`, and `EGG` scarcity-side
  curves changed to `hinge` (CARROT target `0.20` → `1.00`), plus the
  `HINGE_GAIN=8.0` shape and its schema documentation; package version changed
  from 1.32.6 to 1.32.7.

## Adapted implementation

- Reference repository: `https://github.com/diffmap/kaggicultureRL`.
- Reference branch/commit: `refactor-policy-modes` /
  `ef8bb3a2511231d608742ff3b572971d34e6473f`.
- License: MIT; full notice is retained in
  `third_party/diffmap-kaggicultureRL-LICENSE.txt`.
- Adapted files/components: `rust/kaggriculture_env/src/lib.rs` (scalar state
  transitions, PythonRandom clone, fixed observations, action masks, batch
  shell), `rust/kaggriculture_env/Cargo.toml`/`Cargo.lock`, and the protocol
  generator adapted as `scripts/generate_fast_protocol.py`.
- Local modifications: regenerated protocol/rule tables against the pinned
  1.32.7 source; added Rust `hinge` shape code and exact 1.32.7 market shape;
  exposed the reserved farm-header field as public `hires_today`; added the
  narrow `fast_env` Python JSON API and decoder; removed all normal-use
  dependency on the Kaggle registry/OpenSpiel.

## Generated hashes

`rust/kaggriculture_env/src/generated_protocol.rs` is generated from the
pinned official files and records source, schema, and canonical rule-table
SHA-256 values. Regenerate with the pinned source on `PYTHONPATH`:

```text
$env:PYTHONPATH='C:\path\to\kaggle-environments-1.32.7'
python scripts/generate_fast_protocol.py
```

The generated constants are the authoritative Rust protocol inputs; this
stage does not claim full official differential parity.
