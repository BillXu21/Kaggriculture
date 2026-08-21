"""Parquet physical storage for canonical daily replay records (D-018 logical schema).

The logical record produced by `extractor.extract_replay` remains authoritative.
This module is a pure physical adapter: it maps one logical record to one Parquet
row using native nested Arrow types, and reconstructs logically identical records
on read. Requires `pyarrow` (Zstandard compression is built in).

Physical representation
------------------------
- Top level: one row per (episode, seat, day) with columns
  `schema_version`, `metadata`, `day`, `start`, `events`, `targets`, `end`.
- Metadata/start/end/events/targets are nested Arrow structs; boards are
  `list<list<tile struct>>`; string-keyed dicts use Arrow `map` types;
  nullable canonical fields (BUY_LAND `quadrant`, lifecycle `days_until_*`,
  optional provenance scores) stay nullable — never defaulted.
- Board tiles are heterogeneous in the logical schema (null | "LOCKED" |
  {"kind": ...} dicts with kind-specific fields). Each tile is stored as a
  tagged struct: `tile_kind` plus every known tile field (all nullable) plus a
  `present_mask` bitmask recording exactly which logical keys were present, so
  absent and null-valued keys are distinguished on reconstruction. Bare string
  sentinels (any string tile, e.g. "LOCKED" or "WEED") are flagged with
  `bare_string=True` and reconstruct as the identical string.
- Conformance guards: every hand-enumerated struct boundary (record,
  metadata, start/end sections, tile/derived fields, fixed-shape event and
  target entries) rejects unknown keys with a ValueError naming the path and
  key, because Arrow structs would otherwise silently drop them. Dynamic map
  keys (products/crops/op names/sell bins) remain unrestricted.
- Localized encoding (the only non-native field): `events.market_events_ordered`
  holds mixed-type order lists ([op, *args, hour]); each order is stored as one
  JSON string. Every other field is directly readable Arrow data.

Reading without this module::

    import pyarrow.parquet as pq

    table = pq.read_table("out.parquet")          # one row per daily record
    meta = table.column("metadata").to_pylist()   # episode/seat/scores/provenance
    start = table.column("start").to_pylist()[0]
    sell_bins = table.column("targets").to_pylist()[0]["sell_quantity"]
    tile = start["self"]["board"][0][0]           # tagged tile struct

Or, to get logical records identical to extractor output::

    from replay_daily.storage import read_parquet
    records = read_parquet("out.parquet")
"""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"

# --------------------------------------------------------------------------
# Arrow schema
# --------------------------------------------------------------------------

_INT = pa.int64()
_FLOAT = pa.float64()
_STR = pa.string()
_BOOL = pa.bool_()

_MAP_STR_INT = pa.map_(_STR, _INT)

# Tile fields that can appear on {"kind": ...} dict tiles. Order defines the
# present_mask bit assignment together with _DERIVED_FIELDS below.
_TILE_FIELDS = (
    "crop", "animal", "planted_day", "placed_day", "yield_units",
    "max_lifespan_step", "fertilized_until_day", "consecutive_unwatered",
    "watered_today", "fed_today", "cared_today", "consecutive_unfed",
    "fertilizer_available", "pending_care_bonus",
)
_DERIVED_FIELDS = (
    "age_days", "currently_harvestable", "days_until_next_harvest",
    "days_until_next_product", "fertilizer_active", "past_lifespan",
    "starving",
)
_DERIVED_BIT = len(_TILE_FIELDS)  # bit index of the "derived present" flag

def _derived_field_type(name: str) -> pa.DataType:
    if name in ("currently_harvestable", "fertilizer_active", "past_lifespan",
                "starving"):
        return _BOOL
    return _INT


_DERIVED_STRUCT = pa.struct(
    [(name, _derived_field_type(name)) for name in _DERIVED_FIELDS]
)


def _build_tile_struct() -> pa.StructType:
    fields: list[tuple[str, pa.DataType]] = [
        ("tile_kind", _STR),
        # True when the logical tile was a bare string sentinel (e.g. "LOCKED",
        # "WEED") rather than a {"kind": ...} dict; disambiguates reconstruction.
        ("bare_string", _BOOL),
        ("present_mask", _INT),
    ]
    for name in _TILE_FIELDS:
        if name in ("watered_today", "fed_today", "cared_today",
                    "fertilizer_available"):
            fields.append((name, _BOOL))
        elif name in ("crop", "animal"):
            fields.append((name, _STR))
        else:
            fields.append((name, _INT))
    fields.append(("derived", _DERIVED_STRUCT))
    return pa.struct(fields)


TILE_STRUCT = _build_tile_struct()

_BOARD = pa.list_(pa.list_(TILE_STRUCT))

_SELF_STATE = pa.struct([
    ("money", _FLOAT),
    ("board", _BOARD),
    ("unlocked_quadrants", pa.list_(_STR)),
    ("farmer", pa.list_(_INT)),
    ("hands", pa.list_(pa.list_(_INT))),
    ("hires_today", _INT),
    ("shed", _MAP_STR_INT),
    ("seeds", _MAP_STR_INT),
    ("inventories", pa.list_(_MAP_STR_INT)),
])

_PUBLIC_STATE = pa.struct([
    ("money", _FLOAT),
    ("board", _BOARD),
    ("unlocked_quadrants", pa.list_(_STR)),
    ("farmer", pa.list_(_INT)),
    ("hands", pa.list_(pa.list_(_INT))),
    ("hires_today", _INT),
])

_SHARED = pa.struct([
    ("market", pa.struct([
        ("inventory", _MAP_STR_INT),
        ("prices", _MAP_STR_INT),
    ])),
    ("town", pa.struct([
        ("unlocked_shops", pa.list_(_STR)),
        ("shop_counts", _MAP_STR_INT),
    ])),
])

_START = pa.struct([
    ("day", _INT),
    ("hour", _INT),
    ("self", _SELF_STATE),
    ("opponent_public", _PUBLIC_STATE),
    *_SHARED,
    ("previous_execution", pa.struct([
        ("workers_hired", _INT),
        ("hire_cost", _INT),
    ])),
])

_END = pa.struct([
    ("boundary", _STR),
    ("day", _INT),
    ("hour", _INT),
    ("self", _SELF_STATE),
    ("opponent_public", _PUBLIC_STATE),
    *_SHARED,
])

_LEDGER_ENTRY_BASE = [("tile", pa.list_(_INT)), ("hour", _INT)]

_EVENTS = pa.struct([
    ("plants", _MAP_STR_INT),
    ("digs", pa.struct([
        ("total", _INT),
        ("replaced", _MAP_STR_INT),
    ])),
    ("fertilizer_applications", pa.struct([
        ("by_crop", _MAP_STR_INT),
        ("entries", pa.list_(pa.struct(
            [*_LEDGER_ENTRY_BASE, ("crop", _STR)]))),
    ])),
    ("harvests", pa.struct([
        ("by_item", _MAP_STR_INT),
        ("entries", pa.list_(pa.struct(
            [*_LEDGER_ENTRY_BASE, ("item", _STR)]))),
    ])),
    ("buys", pa.struct([
        ("seeds", _MAP_STR_INT),
        ("products", _MAP_STR_INT),
        ("animals", _MAP_STR_INT),
    ])),
    ("land_purchases", pa.list_(pa.struct([
        ("quadrant", _STR),
        ("hour", _INT),
    ]))),
    ("hires", pa.struct([
        ("submitted", _INT),
        ("realized", pa.struct([
            ("workers_hired", _INT),
            ("hire_cost", _INT),
        ])),
    ])),
    ("sells", pa.list_(pa.struct([
        ("product", _STR),
        ("quantity", _INT),
        ("hour", _INT),
    ]))),
    # Localized encoding: each mixed-type [op, *args, hour] order becomes one
    # JSON string. See module docstring.
    ("market_events_ordered", pa.list_(_STR)),
    ("worker_ops_other", _MAP_STR_INT),
])

_TARGETS = pa.struct([
    ("crop_composition_end", _MAP_STR_INT),
    ("animal_counts_end", _MAP_STR_INT),
    ("unlocked_quadrants_end", pa.list_(_STR)),
    ("land_expansion", pa.struct([
        ("expanded", _BOOL),
        ("new_quadrants", pa.list_(_STR)),
    ])),
    ("fertilizer_by_crop", _MAP_STR_INT),
    ("sell_quantity", pa.map_(_STR, pa.map_(_STR, _INT))),
])

_METADATA = pa.struct([
    ("episode_id", _INT),
    ("source_dataset", _STR),
    ("partition_date", _STR),
    ("source_path", _STR),
    ("seat", _INT),
    ("player", _STR),
    ("opponent", _STR),
    ("seed", _INT),
    ("module_version", _STR),
    ("avg_score", _FLOAT),
    ("min_score", _FLOAT),
    ("max_score", _FLOAT),
    ("sum_score", _FLOAT),
    ("final_rewards", pa.list_(_FLOAT)),
    ("final_bank_self", _FLOAT),
    ("final_bank_opponent", _FLOAT),
])

RECORD_SCHEMA = pa.schema([
    ("schema_version", _INT),
    ("metadata", _METADATA),
    ("day", _INT),
    ("start", _START),
    ("events", _EVENTS),
    ("targets", _TARGETS),
    ("end", _END),
])


# --------------------------------------------------------------------------
# Normalization (logical record -> schema-shaped rows)
# --------------------------------------------------------------------------


def _pairs(mapping: dict[str, Any]) -> list[tuple[Any, Any]]:
    return [(k, v) for k, v in mapping.items()]


# Expected keys of every hand-enumerated canonical struct. Arrow struct
# columns silently ignore row-dict keys outside the declared schema, so each
# normalization boundary checks against these and fails loudly on drift.
# Dynamic map keys (products/crops/op names/sell bins) are never restricted.
_RECORD_KEYS = ("schema_version", "metadata", "day", "start", "events",
                "targets", "end")
_SELF_KEYS = ("money", "board", "unlocked_quadrants", "farmer", "hands",
              "hires_today", "shed", "seeds", "inventories")
_PUBLIC_KEYS = ("money", "board", "unlocked_quadrants", "farmer", "hands",
                "hires_today")
_START_KEYS = ("day", "hour", "self", "opponent_public", "market", "town",
               "previous_execution")
_END_KEYS = ("boundary", "day", "hour", "self", "opponent_public", "market",
             "town")
_MARKET_KEYS = ("inventory", "prices")
_TOWN_KEYS = ("unlocked_shops", "shop_counts")
_PREVIOUS_EXECUTION_KEYS = ("workers_hired", "hire_cost")
_EVENTS_KEYS = ("plants", "digs", "fertilizer_applications", "harvests",
                "buys", "land_purchases", "hires", "sells",
                "market_events_ordered", "worker_ops_other")
_DIGS_KEYS = ("total", "replaced")
_FERTILIZER_APP_KEYS = ("by_crop", "entries")
_HARVESTS_KEYS = ("by_item", "entries")
_BUYS_KEYS = ("seeds", "products", "animals")
_HIRES_KEYS = ("submitted", "realized")
_FERTILIZER_ENTRY_KEYS = ("tile", "crop", "hour")
_HARVEST_ENTRY_KEYS = ("tile", "item", "hour")
_LAND_PURCHASE_KEYS = ("quadrant", "hour")
_SELL_KEYS = ("product", "quantity", "hour")
_TARGETS_KEYS = ("crop_composition_end", "animal_counts_end",
                 "unlocked_quadrants_end", "land_expansion",
                 "fertilizer_by_crop", "sell_quantity")
_LAND_EXPANSION_KEYS = ("expanded", "new_quadrants")

_TILE_ALLOWED_KEYS = ("kind", "derived") + _TILE_FIELDS


def _require_known_keys(expected: tuple[str, ...], got: dict[str, Any],
                        path: str) -> None:
    """Raise if a canonical struct carries keys outside the storage schema.

    Without this guard such keys would be silently discarded when the row is
    coerced into the Arrow struct column.
    """
    unknown = sorted(set(got) - set(expected))
    if unknown:
        raise ValueError(
            f"unsupported canonical key(s) at {path}: {unknown}; "
            f"expected only {sorted(set(expected))}"
        )


def _norm_tile(tile: Any, path: str = "board tile") -> dict[str, Any]:
    if tile is None:
        return {"tile_kind": "EMPTY", "bare_string": False, "present_mask": 0}
    if isinstance(tile, str):
        return {"tile_kind": tile, "bare_string": True, "present_mask": 0}
    _require_known_keys(_TILE_ALLOWED_KEYS, tile, path)
    mask = 0
    out: dict[str, Any] = {"tile_kind": tile["kind"], "bare_string": False}
    for i, name in enumerate(_TILE_FIELDS):
        if name in tile:
            mask |= 1 << i
            out[name] = tile[name]
    derived = tile.get("derived")
    if isinstance(derived, dict):
        _require_known_keys(_DERIVED_FIELDS, derived, f"{path}.derived")
        mask |= 1 << _DERIVED_BIT
        for j, name in enumerate(_DERIVED_FIELDS):
            if name in derived:
                mask |= 1 << (_DERIVED_BIT + 1 + j)
        out["derived"] = {name: derived.get(name) for name in _DERIVED_FIELDS}
    else:
        out["derived"] = None
    out["present_mask"] = mask
    return out


def _norm_self(state: dict[str, Any], path: str = "self") -> dict[str, Any]:
    _require_known_keys(_SELF_KEYS, state, path)
    return {
        "money": state["money"],
        "board": [
            [_norm_tile(t, f"{path}.board[{y}][{x}]")
             for x, t in enumerate(row)]
            for y, row in enumerate(state["board"])
        ],
        "unlocked_quadrants": list(state["unlocked_quadrants"]),
        "farmer": list(state["farmer"]),
        "hands": [list(h) for h in state["hands"]],
        "hires_today": state["hires_today"],
        "shed": _pairs(state["shed"]),
        "seeds": _pairs(state["seeds"]),
        "inventories": [_pairs(inv) for inv in state["inventories"]],
    }


def _norm_public(state: dict[str, Any],
                 path: str = "opponent_public") -> dict[str, Any]:
    _require_known_keys(_PUBLIC_KEYS, state, path)
    return {
        "money": state["money"],
        "board": [
            [_norm_tile(t, f"{path}.board[{y}][{x}]")
             for x, t in enumerate(row)]
            for y, row in enumerate(state["board"])
        ],
        "unlocked_quadrants": list(state["unlocked_quadrants"]),
        "farmer": list(state["farmer"]),
        "hands": [list(h) for h in state["hands"]],
        "hires_today": state["hires_today"],
    }


def _norm_shared(state: dict[str, Any], path: str) -> dict[str, Any]:
    _require_known_keys(_MARKET_KEYS, state["market"], f"{path}.market")
    _require_known_keys(_TOWN_KEYS, state["town"], f"{path}.town")
    return {
        "market": {
            "inventory": _pairs(state["market"]["inventory"]),
            "prices": _pairs(state["market"]["prices"]),
        },
        "town": {
            "unlocked_shops": list(state["town"]["unlocked_shops"]),
            "shop_counts": _pairs(state["town"]["shop_counts"]),
        },
    }


def _norm_entry(entry: dict[str, Any], expected: tuple[str, ...],
                path: str) -> dict[str, Any]:
    """Guard and copy one fixed-shape ledger/target entry."""
    _require_known_keys(expected, entry, path)
    return {name: entry.get(name) for name in expected}


def _norm_events(ev: dict[str, Any]) -> dict[str, Any]:
    _require_known_keys(_EVENTS_KEYS, ev, "events")
    _require_known_keys(_DIGS_KEYS, ev["digs"], "events.digs")
    _require_known_keys(
        _FERTILIZER_APP_KEYS, ev["fertilizer_applications"],
        "events.fertilizer_applications")
    _require_known_keys(_HARVESTS_KEYS, ev["harvests"], "events.harvests")
    _require_known_keys(_BUYS_KEYS, ev["buys"], "events.buys")
    _require_known_keys(_HIRES_KEYS, ev["hires"], "events.hires")
    _require_known_keys(
        _PREVIOUS_EXECUTION_KEYS, ev["hires"]["realized"],
        "events.hires.realized")
    return {
        "plants": _pairs(ev["plants"]),
        "digs": {
            "total": ev["digs"]["total"],
            "replaced": _pairs(ev["digs"]["replaced"]),
        },
        "fertilizer_applications": {
            "by_crop": _pairs(ev["fertilizer_applications"]["by_crop"]),
            "entries": [
                _norm_entry(e, _FERTILIZER_ENTRY_KEYS,
                            f"events.fertilizer_applications.entries[{i}]")
                for i, e in enumerate(ev["fertilizer_applications"]["entries"])
            ],
        },
        "harvests": {
            "by_item": _pairs(ev["harvests"]["by_item"]),
            "entries": [
                _norm_entry(e, _HARVEST_ENTRY_KEYS,
                            f"events.harvests.entries[{i}]")
                for i, e in enumerate(ev["harvests"]["entries"])
            ],
        },
        "buys": {
            "seeds": _pairs(ev["buys"]["seeds"]),
            "products": _pairs(ev["buys"]["products"]),
            "animals": _pairs(ev["buys"]["animals"]),
        },
        "land_purchases": [
            _norm_entry(p, _LAND_PURCHASE_KEYS, f"events.land_purchases[{i}]")
            for i, p in enumerate(ev["land_purchases"])
        ],
        "hires": {
            "submitted": ev["hires"]["submitted"],
            "realized": {
                "workers_hired": ev["hires"]["realized"]["workers_hired"],
                "hire_cost": ev["hires"]["realized"]["hire_cost"],
            },
        },
        "sells": [
            _norm_entry(s, _SELL_KEYS, f"events.sells[{i}]")
            for i, s in enumerate(ev["sells"])
        ],
        "market_events_ordered": [
            json.dumps(order, separators=(",", ":"), ensure_ascii=False)
            for order in ev["market_events_ordered"]
        ],
        "worker_ops_other": _pairs(ev["worker_ops_other"]),
    }


def _norm_targets(targets: dict[str, Any]) -> dict[str, Any]:
    _require_known_keys(_TARGETS_KEYS, targets, "targets")
    _require_known_keys(
        _LAND_EXPANSION_KEYS, targets["land_expansion"],
        "targets.land_expansion")
    return {
        "crop_composition_end": _pairs(targets["crop_composition_end"]),
        "animal_counts_end": _pairs(targets["animal_counts_end"]),
        "unlocked_quadrants_end": list(targets["unlocked_quadrants_end"]),
        "land_expansion": {
            "expanded": targets["land_expansion"]["expanded"],
            "new_quadrants": list(targets["land_expansion"]["new_quadrants"]),
        },
        "fertilizer_by_crop": _pairs(targets["fertilizer_by_crop"]),
        "sell_quantity": [
            (anchor, _pairs(products))
            for anchor, products in targets["sell_quantity"].items()
        ],
    }


def record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one logical canonical record into a RECORD_SCHEMA row."""
    _require_known_keys(_RECORD_KEYS, record, "record")
    _require_known_keys(
        tuple(field.name for field in _METADATA), record["metadata"],
        "metadata")
    start, end = record["start"], record["end"]
    _require_known_keys(_START_KEYS, start, "start")
    _require_known_keys(_END_KEYS, end, "end")
    _require_known_keys(
        _PREVIOUS_EXECUTION_KEYS, start["previous_execution"],
        "start.previous_execution")
    return {
        "schema_version": record["schema_version"],
        "metadata": dict(record["metadata"]),
        "day": record["day"],
        "start": {
            "day": start["day"],
            "hour": start["hour"],
            "self": _norm_self(start["self"], "start.self"),
            "opponent_public": _norm_public(
                start["opponent_public"], "start.opponent_public"),
            **_norm_shared(start, "start"),
            "previous_execution": {
                "workers_hired": start["previous_execution"]["workers_hired"],
                "hire_cost": start["previous_execution"]["hire_cost"],
            },
        },
        "events": _norm_events(record["events"]),
        "targets": _norm_targets(record["targets"]),
        "end": {
            "boundary": end["boundary"],
            "day": end["day"],
            "hour": end["hour"],
            "self": _norm_self(end["self"], "end.self"),
            "opponent_public": _norm_public(
                end["opponent_public"], "end.opponent_public"),
            **_norm_shared(end, "end"),
        },
    }


# --------------------------------------------------------------------------
# Denormalization (schema-shaped rows -> logical records)
# --------------------------------------------------------------------------


def _denorm_tile(tile: Any) -> Any:
    if tile is None:
        return None
    if tile["bare_string"]:
        return tile["tile_kind"]
    kind = tile["tile_kind"]
    if kind == "EMPTY":
        return None
    mask = tile["present_mask"]
    out: dict[str, Any] = {"kind": kind}
    for i, name in enumerate(_TILE_FIELDS):
        if mask >> i & 1:
            out[name] = tile.get(name)
    if mask >> _DERIVED_BIT & 1:
        derived_raw = tile.get("derived") or {}
        out["derived"] = {
            name: derived_raw.get(name)
            for j, name in enumerate(_DERIVED_FIELDS)
            if mask >> (_DERIVED_BIT + 1 + j) & 1
        }
    else:
        out["derived"] = None
    return out


def _denorm_board(board: Any) -> list[list[Any]]:
    return [[_denorm_tile(t) for t in row] for row in board]


def _denorm_map(pairs: Any) -> dict[str, Any]:
    return {k: v for k, v in pairs}


def _denorm_self(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "money": state["money"],
        "board": _denorm_board(state["board"]),
        "unlocked_quadrants": list(state["unlocked_quadrants"]),
        "farmer": list(state["farmer"]),
        "hands": [list(h) for h in state["hands"]],
        "hires_today": state["hires_today"],
        "shed": _denorm_map(state["shed"]),
        "seeds": _denorm_map(state["seeds"]),
        "inventories": [_denorm_map(inv) for inv in state["inventories"]],
    }


def _denorm_public(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "money": state["money"],
        "board": _denorm_board(state["board"]),
        "unlocked_quadrants": list(state["unlocked_quadrants"]),
        "farmer": list(state["farmer"]),
        "hands": [list(h) for h in state["hands"]],
        "hires_today": state["hires_today"],
    }


def _denorm_shared(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "market": {
            "inventory": _denorm_map(state["market"]["inventory"]),
            "prices": _denorm_map(state["market"]["prices"]),
        },
        "town": {
            "unlocked_shops": list(state["town"]["unlocked_shops"]),
            "shop_counts": _denorm_map(state["town"]["shop_counts"]),
        },
    }


def _denorm_events(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "plants": _denorm_map(ev["plants"]),
        "digs": {
            "total": ev["digs"]["total"],
            "replaced": _denorm_map(ev["digs"]["replaced"]),
        },
        "fertilizer_applications": {
            "by_crop": _denorm_map(ev["fertilizer_applications"]["by_crop"]),
            "entries": [
                {"tile": list(e["tile"]), "crop": e["crop"], "hour": e["hour"]}
                for e in ev["fertilizer_applications"]["entries"]
            ],
        },
        "harvests": {
            "by_item": _denorm_map(ev["harvests"]["by_item"]),
            "entries": [
                {"tile": list(e["tile"]), "item": e["item"], "hour": e["hour"]}
                for e in ev["harvests"]["entries"]
            ],
        },
        "buys": {
            "seeds": _denorm_map(ev["buys"]["seeds"]),
            "products": _denorm_map(ev["buys"]["products"]),
            "animals": _denorm_map(ev["buys"]["animals"]),
        },
        "land_purchases": [
            {"quadrant": p["quadrant"], "hour": p["hour"]}
            for p in ev["land_purchases"]
        ],
        "hires": {
            "submitted": ev["hires"]["submitted"],
            "realized": {
                "workers_hired": ev["hires"]["realized"]["workers_hired"],
                "hire_cost": ev["hires"]["realized"]["hire_cost"],
            },
        },
        "sells": [
            {"product": s["product"], "quantity": s["quantity"], "hour": s["hour"]}
            for s in ev["sells"]
        ],
        "market_events_ordered": [json.loads(s) for s in ev["market_events_ordered"]],
        "worker_ops_other": _denorm_map(ev["worker_ops_other"]),
    }


def _denorm_targets(targets: dict[str, Any]) -> dict[str, Any]:
    return {
        "crop_composition_end": _denorm_map(targets["crop_composition_end"]),
        "animal_counts_end": _denorm_map(targets["animal_counts_end"]),
        "unlocked_quadrants_end": list(targets["unlocked_quadrants_end"]),
        "land_expansion": {
            "expanded": targets["land_expansion"]["expanded"],
            "new_quadrants": list(targets["land_expansion"]["new_quadrants"]),
        },
        "fertilizer_by_crop": _denorm_map(targets["fertilizer_by_crop"]),
        "sell_quantity": {
            anchor: _denorm_map(products)
            for anchor, products in targets["sell_quantity"]
        },
    }


def row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct one logical canonical record from a RECORD_SCHEMA row."""
    return {
        "schema_version": row["schema_version"],
        "metadata": dict(row["metadata"]),
        "day": row["day"],
        "start": {
            "day": row["start"]["day"],
            "hour": row["start"]["hour"],
            "self": _denorm_self(row["start"]["self"]),
            "opponent_public": _denorm_public(row["start"]["opponent_public"]),
            **_denorm_shared(row["start"]),
            "previous_execution": {
                "workers_hired": row["start"]["previous_execution"]["workers_hired"],
                "hire_cost": row["start"]["previous_execution"]["hire_cost"],
            },
        },
        "events": _denorm_events(row["events"]),
        "targets": _denorm_targets(row["targets"]),
        "end": {
            "boundary": row["end"]["boundary"],
            "day": row["end"]["day"],
            "hour": row["end"]["hour"],
            "self": _denorm_self(row["end"]["self"]),
            "opponent_public": _denorm_public(row["end"]["opponent_public"]),
            **_denorm_shared(row["end"]),
        },
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def records_to_table(records: list[dict[str, Any]]) -> pa.Table:
    """Build a RECORD_SCHEMA Arrow table from logical canonical records."""
    return pa.Table.from_pylist(
        [record_to_row(r) for r in records], schema=RECORD_SCHEMA,
    )


def table_to_records(table: pa.Table) -> list[dict[str, Any]]:
    """Reconstruct logical canonical records from a RECORD_SCHEMA table."""
    return [row_to_record(row) for row in table.to_pylist()]


def write_parquet(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write logical canonical records to a Zstandard-compressed Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(records_to_table(records), path, compression=PARQUET_COMPRESSION)


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Read a Parquet file written by `write_parquet` back into logical records."""
    return table_to_records(pq.read_table(path))
