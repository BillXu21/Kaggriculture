"""Schema-v2 Parquet -> compact NumPy arrays BC adapter.

Reads canonical daily-record Parquet directly with PyArrow (nested Arrow
columns, dotted-path projection; no `replay_daily.read_parquet` logical
reconstruction), verifies `schema_version == 2` on every row of every
selected file, filters once by partition date allowlist and the equal
`min_score` cutoff, and converts the selected rows exactly once into compact
NumPy arrays resident in RAM. Never a random row split: split membership is
defined exclusively by `metadata.partition_date`.

Model-facing input arrays (all current-state features, stable order):

- `board_kind`        int16   [N, 100]  tile-kind id (TILE_KIND_IDS)
- `board_crop`        int8    [N, 100]  0 absent / 1..5 crop / 6 UNKNOWN
- `board_animal`      int8    [N, 100]  0 absent / 1..3 animal / 4 UNKNOWN
- `board_numeric`     float32 [N,100,11] BOARD_NUMERIC_FIELDS; fill 0.0,
                                        nullable derived timing NaN when the
                                        derived struct is present but null
- `board_bool`        bool    [N, 100, 8] BOARD_BOOL_FIELDS
- `board_mask`        uint8   [N, 100, 4] tile/plant/animal/derived present
- `opp_board_*`       same six arrays for the opponent PUBLIC board
                                (only with include_opponent=True; public
                                state only — opponent shed/seeds/
                                inventories/private are never read)
- `scalars`           float32 [N, 4] money, hires_today,
                                prev workers_hired, prev hire_cost
- `shed_counts`       int32   [N, 12] RESOURCE_ORDER
- `seed_counts`       int32   [N, 5]  CROP_ORDER
- `carried_counts`    int32   [N, 12] worker inventories summed per item
- `unlocked`          uint8   [N, 4]  NW/NE/SW/SE flags
- `market_inventory`  int32   [N, 9]  PRODUCT_ORDER
- `market_prices`     float32 [N, 9]  PRODUCT_ORDER
- `shop_counts`       int32   [N, 9]  SHOPS + UNKNOWN channel
- `day`, `days_remaining` int16 [N]

Target arrays:

- `crop_target`       int32 [N,5] WHEAT/CARROT/TOMATO/STRAWBERRY/MELON
- `animal_target`     int32 [N,3] GOOSE/COW/SHEEP
- `land_count`        int32 [N]   resulting unlocked quadrant count 1..4
- `fertilizer_target` int32 [N,5] by crop (canonical "unknown" key dropped)
- `care_target`       int32 [N,3] from targets.care_by_animal (v2 required;
                              missing CARE is an error, never fabricated)
- `sell_presence`         bool    [N,9,6]
- `sell_quantity_log1p`   float32 [N,9,6]
- `sell_quantity_bounded` int32   [N,9,6]  per-cell bounded aggregates from
                                           events.sells (NOT targets.sell_
                                           quantity); repeated same-bin
                                           events may exceed the 100 cap

Names, scores, final banks, partition identity, source paths and other
result metadata are returned only in the separate `meta` list of dicts for
filtering/evaluation; they are never part of `inputs`.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from replay_daily.constants import SCHEMA_VERSION

from .constants import (
    ANIMAL_IDS,
    ANIMAL_ORDER,
    ANIMAL_UNKNOWN_ID,
    BOARD_BOOL_FIELDS,
    BOARD_DERIVED_FIELD_ORDER,
    BOARD_DERIVED_PRESENT_BIT,
    BOARD_MASK_FIELDS,
    BOARD_NULLABLE_TIMING,
    BOARD_NUMERIC_FILL,
    BOARD_NUMERIC_FIELDS,
    BOARD_SIZE,
    CROP_IDS,
    CROP_ORDER,
    CROP_UNKNOWN_ID,
    MAX_HANDS,
    MIN_SCORE_DEFAULT,
    PRODUCT_ORDER,
    QUADRANT_ORDER,
    RESOURCE_ORDER,
    SELL_BIN_COUNT,
    SHOPS,
    SHOP_VOCAB,
    TILE_EMPTY,
    TILE_KIND_IDS,
    TOTAL_DAYS,
    TRAIN_DATES_DEFAULT,
    VAL_DATES_DEFAULT,
    bound_sell_quantity,
    board_field_present,
    sell_bin_index,
)

# Dotted-path projection: PyArrow names each selected nested leaf column by
# its last path component ("events.sells" -> "sells").
_READ_COLUMNS = [
    "schema_version", "metadata", "day", "start", "events.sells", "targets",
]


class SchemaVersionError(ValueError):
    """Raised on v1/mixed/missing schema_version in processed Parquet."""


def _read_selected_table(path: str | Path) -> pa.Table:
    try:
        return pq.read_table(path, columns=_READ_COLUMNS)
    except (KeyError, ValueError, pa.ArrowInvalid) as exc:
        raise ValueError(
            f"{path}: not a canonical daily-record Parquet file ({exc}); "
            f"expected columns {_READ_COLUMNS}"
        ) from exc


def _require_schema_v2(table: pa.Table, path: str | Path) -> None:
    if "schema_version" not in table.column_names:
        raise SchemaVersionError(f"{path}: schema_version column missing")
    raw_versions = table.column("schema_version").to_pylist()
    # A date-filtered split may legitimately be empty; the source table still
    # carries the schema-v2 column and there are no foreign rows to accept.
    if not raw_versions:
        return
    if any(v is None for v in raw_versions):
        raise SchemaVersionError(f"{path}: schema_version contains null")
    versions = sorted({int(v) for v in raw_versions})
    if versions != [SCHEMA_VERSION]:
        raise SchemaVersionError(
            f"{path}: unsupported schema_version value(s) {versions}; "
            f"expected only {SCHEMA_VERSION}. v1/mixed processed data is "
            f"rejected; regenerate it from raw replays."
        )


def _row_filter_mask(meta_rows: Sequence[Mapping[str, Any]],
                     dates: Sequence[str], min_score: float) -> list[bool]:
    allowed_dates = set(dates)
    keep = []
    for m in meta_rows:
        if m is None:
            keep.append(False)
            continue
        score = m.get("min_score")
        keep.append(
            m.get("partition_date") in allowed_dates
            and score is not None
            and float(score) >= min_score
        )
    return keep


def load_selected_table(
    paths: str | Path | Sequence[str | Path],
    *,
    dates: Sequence[str],
    min_score: float = MIN_SCORE_DEFAULT,
) -> tuple[pa.Table, dict[str, int]]:
    """Read, version-check, and filter rows once; returns (table, report).

    The report counts rows read, selected by the date allowlist + equal
    min_score cutoff, and excluded. Selection never shuffles or samples.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("at least one Parquet path is required")
    parts: list[pa.Table] = []
    empty_template: pa.Table | None = None
    report = {"rows_read": 0, "rows_selected": 0, "rows_excluded": 0}
    for path in paths:
        table = _read_selected_table(path)
        if empty_template is None:
            empty_template = table
        _require_schema_v2(table, path)
        meta_rows = table.column("metadata").to_pylist()
        mask = _row_filter_mask(meta_rows, dates, min_score)
        report["rows_read"] += table.num_rows
        report["rows_selected"] += int(sum(mask))
        report["rows_excluded"] += int(len(mask) - sum(mask))
        if any(mask):
            parts.append(table.filter(pa.array(mask)))
    if not parts:
        assert empty_template is not None
        return empty_template.slice(0, 0), report
    return pa.concat_tables(parts), report


# ------------------------------------------------------- array conversion


def _id_lookup(value: Any, ids: Mapping[str, int], unknown_id: int,
               what: str) -> int:
    if value is None:
        return 0
    got = ids.get(value)
    if got is None:
        return unknown_id  # explicit UNKNOWN id, never invented identity
    return got


def _board_arrays(board_rows: Sequence[Any]) -> dict[str, np.ndarray]:
    n = len(board_rows)
    kind = np.full((n, BOARD_SIZE), TILE_KIND_IDS[TILE_EMPTY], dtype=np.int16)
    crop = np.zeros((n, BOARD_SIZE), dtype=np.int8)
    animal = np.zeros((n, BOARD_SIZE), dtype=np.int8)
    numeric = np.full(
        (n, BOARD_SIZE, len(BOARD_NUMERIC_FIELDS)), BOARD_NUMERIC_FILL,
        dtype=np.float32)
    boolean = np.zeros((n, BOARD_SIZE, len(BOARD_BOOL_FIELDS)), dtype=bool)
    mask = np.zeros((n, BOARD_SIZE, len(BOARD_MASK_FIELDS)), dtype=np.uint8)

    num_idx = {name: i for i, name in enumerate(BOARD_NUMERIC_FIELDS)}
    bool_idx = {name: i for i, name in enumerate(BOARD_BOOL_FIELDS)}

    for i, board in enumerate(board_rows):
        tiles = [tile for row in board for tile in row]
        if len(tiles) != BOARD_SIZE:
            raise ValueError(
                f"record {i}: expected {BOARD_SIZE} board tiles, "
                f"got {len(tiles)}")
        for j, tile in enumerate(tiles):
            tk = tile["tile_kind"]
            if tile.get("bare_string"):
                # Bare string sentinels: known strings keep their kind id,
                # anything else is BARE_OTHER.
                kind[i, j] = TILE_KIND_IDS.get(tk, TILE_KIND_IDS["BARE_OTHER"])
            else:
                kind[i, j] = TILE_KIND_IDS.get(tk, TILE_KIND_IDS["UNKNOWN"])
            if tk == TILE_EMPTY and not tile.get("bare_string"):
                continue

            present_mask = int(tile.get("present_mask") or 0)
            mask[i, j, 0] = 1  # non-empty tagged or bare tile
            crop_present = board_field_present(present_mask, "crop")
            animal_present = board_field_present(present_mask, "animal")
            crop[i, j] = _id_lookup(
                tile.get("crop") if crop_present else None,
                CROP_IDS, CROP_UNKNOWN_ID, "crop")
            animal[i, j] = _id_lookup(
                tile.get("animal") if animal_present else None,
                ANIMAL_IDS, ANIMAL_UNKNOWN_ID, "animal")
            mask[i, j, 1] = int(crop_present)
            mask[i, j, 2] = int(animal_present)

            derived = tile.get("derived")
            derived_present = bool(
                present_mask & (1 << BOARD_DERIVED_PRESENT_BIT))
            mask[i, j, 3] = int(derived_present)
            for name in BOARD_NUMERIC_FIELDS:
                field_present = board_field_present(present_mask, name)
                if name in BOARD_DERIVED_FIELD_ORDER:
                    raw = derived.get(name) if derived is not None and field_present else None
                    if raw is None and name in BOARD_NULLABLE_TIMING and field_present:
                        value = np.nan  # present derived key, null timing
                    else:
                        value = BOARD_NUMERIC_FILL if raw is None else float(raw)
                else:
                    raw = tile.get(name) if field_present else None
                    value = BOARD_NUMERIC_FILL if raw is None else float(raw)
                numeric[i, j, num_idx[name]] = value
            for name in BOARD_BOOL_FIELDS:
                field_present = board_field_present(present_mask, name)
                if name in BOARD_DERIVED_FIELD_ORDER:
                    raw = derived.get(name) if derived is not None and field_present else None
                else:
                    raw = tile.get(name) if field_present else None
                boolean[i, j, bool_idx[name]] = bool(raw) \
                    if raw is not None else False
    return {
        "kind": kind, "crop": crop, "animal": animal,
        "numeric": numeric, "bool": boolean, "mask": mask,
    }


def _map_vector(pairs: Sequence[tuple[str, Any]], vocab: Sequence[str],
                what: str, unknown_channel: bool = False) -> np.ndarray:
    vec = np.zeros(len(vocab) + (1 if unknown_channel else 0), dtype=np.int32)
    known = set(vocab)
    if isinstance(pairs, Mapping):
        pairs = pairs.items()
    for key, value in pairs:
        if key in known:
            vec[vocab.index(key)] += int(value)
        elif unknown_channel:
            vec[-1] += int(value)
        else:
            raise ValueError(
                f"unknown {what} key {key!r}; expected one of {sorted(known)}. "
                f"Engine resource keys are a closed set."
            )
    return vec


def _as_mapping(value: Any) -> dict[str, Any]:
    """Normalize Arrow map output (pairs) and ordinary Python mappings."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {key: item for key, item in value}


def _column_rows(table: pa.Table, name: str) -> list[Any]:
    """Read a projected leaf or its parent struct without logical rebuilding."""
    if name in table.column_names:
        return table.column(name).to_pylist()
    if name == "sells" and "events" in table.column_names:
        return [((event or {}).get("sells") or [])
                for event in table.column("events").to_pylist()]
    raise ValueError(f"Arrow table is missing selected column {name!r}")


def _worker_side_arrays(state: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    unlocked = np.zeros(len(QUADRANT_ORDER), dtype=np.uint8)
    for q in state["unlocked_quadrants"]:
        if q not in QUADRANT_ORDER:
            raise ValueError(f"unknown quadrant {q!r}")
        unlocked[QUADRANT_ORDER.index(q)] = 1
    hands = state.get("hands") or []
    if len(hands) > MAX_HANDS:
        raise ValueError(
            f"{len(hands)} hired hands exceed MAX_HANDS={MAX_HANDS}")
    return unlocked, hands


def table_to_arrays(
    table: pa.Table, *, include_opponent: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Convert a filtered schema-v2 table into compact NumPy arrays."""
    _require_schema_v2(table, "in-memory Arrow table")
    n = table.num_rows
    start_rows = _column_rows(table, "start")
    target_rows = _column_rows(table, "targets")
    sells_rows = _column_rows(table, "sells")
    day_col = _column_rows(table, "day")
    meta_rows = _column_rows(table, "metadata")

    inputs: dict[str, np.ndarray] = {}
    boards = _board_arrays([s["self"]["board"] for s in start_rows])
    inputs["board_kind"] = boards["kind"]
    inputs["board_crop"] = boards["crop"]
    inputs["board_animal"] = boards["animal"]
    inputs["board_numeric"] = boards["numeric"]
    inputs["board_bool"] = boards["bool"]
    inputs["board_mask"] = boards["mask"]

    scalars = np.zeros((n, 4), dtype=np.float32)
    shed = np.zeros((n, len(RESOURCE_ORDER)), dtype=np.int32)
    seeds = np.zeros((n, len(CROP_ORDER)), dtype=np.int32)
    carried = np.zeros((n, len(RESOURCE_ORDER)), dtype=np.int32)
    unlocked = np.zeros((n, len(QUADRANT_ORDER)), dtype=np.uint8)
    market_inv = np.zeros((n, len(PRODUCT_ORDER)), dtype=np.int32)
    market_prices = np.zeros((n, len(PRODUCT_ORDER)), dtype=np.float32)
    shops = np.zeros((n, len(SHOP_VOCAB)), dtype=np.int32)
    day = np.asarray(day_col, dtype=np.int16)

    for i, s in enumerate(start_rows):
        self_state = s["self"]
        scalars[i] = (
            float(self_state["money"]), float(self_state["hires_today"]),
            float(s["previous_execution"]["workers_hired"]),
            float(s["previous_execution"]["hire_cost"]),
        )
        shed[i] = _map_vector(self_state["shed"], RESOURCE_ORDER, "shed item")
        seeds[i] = _map_vector(self_state["seeds"], CROP_ORDER, "seed")
        for inventory in self_state["inventories"]:
            carried[i] += _map_vector(inventory, RESOURCE_ORDER,
                                      "carried item")
        unlocked[i], _ = _worker_side_arrays(self_state)
        market_inv[i] = _map_vector(s["market"]["inventory"],
                                    PRODUCT_ORDER, "market item")
        prices = _map_vector(s["market"]["prices"], PRODUCT_ORDER,
                             "market price")
        market_prices[i] = prices.astype(np.float32)
        shops[i] = _map_vector(s["town"]["shop_counts"], SHOPS,
                               "town shop", unknown_channel=True)

    inputs["scalars"] = scalars
    inputs["shed_counts"] = shed
    inputs["seed_counts"] = seeds
    inputs["carried_counts"] = carried
    inputs["unlocked"] = unlocked
    inputs["market_inventory"] = market_inv
    inputs["market_prices"] = market_prices
    inputs["shop_counts"] = shops
    inputs["day"] = day
    inputs["days_remaining"] = np.full(n, TOTAL_DAYS - 1, dtype=np.int16) - day

    if include_opponent:
        opp_boards = _board_arrays(
            [s["opponent_public"]["board"] for s in start_rows])
        inputs["opp_board_kind"] = opp_boards["kind"]
        inputs["opp_board_crop"] = opp_boards["crop"]
        inputs["opp_board_animal"] = opp_boards["animal"]
        inputs["opp_board_numeric"] = opp_boards["numeric"]
        inputs["opp_board_bool"] = opp_boards["bool"]
        inputs["opp_board_mask"] = opp_boards["mask"]
        opp_scalars = np.zeros((n, 2), dtype=np.float32)
        opp_unlocked = np.zeros((n, len(QUADRANT_ORDER)), dtype=np.uint8)
        for i, s in enumerate(start_rows):
            opp = s["opponent_public"]
            opp_scalars[i] = (float(opp["money"]), float(opp["hires_today"]))
            opp_unlocked[i], _ = _worker_side_arrays(opp)
        inputs["opp_scalars"] = opp_scalars
        inputs["opp_unlocked"] = opp_unlocked

    targets = build_targets(target_rows, sells_rows)
    meta = [_eval_metadata(m, d) for m, d in zip(meta_rows, day_col)]
    return inputs, targets, meta


def _eval_metadata(m: Mapping[str, Any], day: int) -> dict[str, Any]:
    """Filtering/evaluation metadata only — never model input."""
    return {
        "episode_id": m.get("episode_id"),
        "seat": m.get("seat"),
        "day": day,
        "partition_date": m.get("partition_date"),
        "player": m.get("player"),
        "opponent": m.get("opponent"),
        "avg_score": m.get("avg_score"),
        "min_score": m.get("min_score"),
        "max_score": m.get("max_score"),
        "sum_score": m.get("sum_score"),
        "final_bank_self": m.get("final_bank_self"),
        "final_bank_opponent": m.get("final_bank_opponent"),
        "source_path": m.get("source_path"),
    }


def aggregate_sells(sells: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Bound, bin, and aggregate primitive sell events -> [9, 6] int32.

    Per event: quantity bounded to [0, 100]; bin = floor(hour/4). Repeated
    same-bin events accumulate and may legitimately exceed 100.
    """
    agg = np.zeros((len(PRODUCT_ORDER), SELL_BIN_COUNT), dtype=np.int32)
    for event in sells:
        product = event["product"]
        if product not in PRODUCT_ORDER:
            raise ValueError(
                f"unknown sell product {product!r}; expected one of "
                f"{PRODUCT_ORDER}")
        agg[PRODUCT_ORDER.index(product), sell_bin_index(event["hour"])] += \
            bound_sell_quantity(event["quantity"])
    return agg


def build_targets(
    target_rows: Sequence[Mapping[str, Any]],
    sells_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, np.ndarray]:
    n = len(target_rows)
    if len(sells_rows) != n:
        raise ValueError(
            f"target and events.sells row counts differ: {n} vs {len(sells_rows)}")
    crop = np.zeros((n, len(CROP_ORDER)), dtype=np.int32)
    animal = np.zeros((n, len(ANIMAL_ORDER)), dtype=np.int32)
    land = np.zeros(n, dtype=np.int32)
    fertilizer = np.zeros((n, len(CROP_ORDER)), dtype=np.int32)
    care = np.zeros((n, len(ANIMAL_ORDER)), dtype=np.int32)
    presence = np.zeros((n, len(PRODUCT_ORDER), SELL_BIN_COUNT), dtype=bool)
    qty_bounded = np.zeros((n, len(PRODUCT_ORDER), SELL_BIN_COUNT),
                           dtype=np.int32)

    for i, (t, sells) in enumerate(zip(target_rows, sells_rows)):
        if t is None:
            raise ValueError(f"record {i}: targets struct missing")
        comp = _as_mapping(t["crop_composition_end"])
        for k, name in enumerate(CROP_ORDER):
            crop[i, k] = int(comp.get(name, 0))
        if any(key not in CROP_ORDER and int(value) != 0
               for key, value in comp.items()):
            raise ValueError(f"record {i}: unknown crop target key")
        animals_end = _as_mapping(t["animal_counts_end"])
        for k, name in enumerate(ANIMAL_ORDER):
            animal[i, k] = int(animals_end.get(name, 0))
        if any(key not in ANIMAL_ORDER and int(value) != 0
               for key, value in animals_end.items()):
            raise ValueError(f"record {i}: unknown animal target key")
        if t.get("unlocked_quadrants_end") is None:
            raise ValueError(f"record {i}: unlocked_quadrants_end missing")
        land[i] = len(t["unlocked_quadrants_end"])
        for key, count in _as_mapping(t["fertilizer_by_crop"]).items():
            if int(count) < 0:
                raise ValueError(f"negative fertilizer target value for {key!r}")
            if key in CROP_ORDER:
                fertilizer[i, CROP_ORDER.index(key)] += int(count)
            # Canonical "unknown"-crop applications have no target channel
            # and are intentionally excluded from the [N, 5] vector.
            elif key != "unknown" and int(count) != 0:
                raise ValueError(f"record {i}: unknown fertilizer crop {key!r}")
        care_by_animal = t.get("care_by_animal")
        if care_by_animal is None:
            raise ValueError(
                f"record {i}: targets.care_by_animal missing; this is a "
                f"schema-v2 field. Refusing to fabricate CARE targets."
            )
        for k, name in enumerate(ANIMAL_ORDER):
            care[i, k] = int(_as_mapping(care_by_animal).get(name, 0))

        agg = aggregate_sells(sells)
        qty_bounded[i] = agg
        presence[i] = agg > 0

    for name, arr in (("crop_target", crop), ("animal_target", animal),
                      ("fertilizer_target", fertilizer),
                      ("care_target", care), ("land_count", land)):
        if (arr < 0).any():
            raise ValueError(f"negative values in target {name}")
    if (land < 1).any() or (land > len(QUADRANT_ORDER)).any():
        raise ValueError("land_count target must be in [1, 4]")
    return {
        "crop_target": crop,
        "animal_target": animal,
        "land_count": land,
        "fertilizer_target": fertilizer,
        "care_target": care,
        "sell_presence": presence,
        "sell_quantity_bounded": qty_bounded,
        "sell_quantity_log1p": np.log1p(qty_bounded.astype(np.float64))
        .astype(np.float32),
    }


# ------------------------------------------------------------ public API


def load_dataset(
    paths: str | Path | Sequence[str | Path],
    *,
    dates: Sequence[str],
    min_score: float = MIN_SCORE_DEFAULT,
    include_opponent: bool = False,
) -> dict[str, Any]:
    """Load one date-filtered dataset split as compact arrays."""
    table, report = load_selected_table(paths, dates=dates,
                                        min_score=min_score)
    inputs, targets, meta = table_to_arrays(table,
                                            include_opponent=include_opponent)
    return {"inputs": inputs, "targets": targets, "meta": meta,
            "report": report}


def load_train_val(
    paths: str | Path | Sequence[str | Path],
    *,
    train_dates: Sequence[str] = TRAIN_DATES_DEFAULT,
    val_dates: Sequence[str] = VAL_DATES_DEFAULT,
    min_score: float = MIN_SCORE_DEFAULT,
    include_opponent: bool = False,
) -> dict[str, Any]:
    """Date-held-out train/validation split with equal min_score filtering.

    Split membership comes exclusively from metadata.partition_date; there is
    no random row split anywhere. Rows whose partition_date is in neither
    allowlist are excluded and counted.
    """
    overlap = set(train_dates) & set(val_dates)
    if overlap:
        raise ValueError(f"train/val date lists overlap: {sorted(overlap)}")
    table, read_report = load_selected_table(
        paths, dates=tuple(train_dates) + tuple(val_dates), min_score=min_score)
    meta_rows = table.column("metadata").to_pylist()
    dates_col = [m.get("partition_date") for m in meta_rows]

    def _split(allowed: Sequence[str]) -> dict[str, Any]:
        mask = pa.array([d in set(allowed) for d in dates_col])
        split_table = table.filter(mask)
        inputs, targets, meta = table_to_arrays(
            split_table, include_opponent=include_opponent)
        return {"inputs": inputs, "targets": targets, "meta": meta,
                "report": {"rows_read": read_report["rows_read"],
                           "rows_selected": len(meta),
                           "rows_excluded": read_report["rows_excluded"]}}

    train = _split(train_dates)
    val = _split(val_dates)
    return {
        "train": train,
        "val": val,
        "report": {
            "train_rows": len(train["meta"]),
            "val_rows": len(val["meta"]),
            "rows_read": read_report["rows_read"],
            "rows_selected": read_report["rows_selected"],
            "rows_excluded": read_report["rows_excluded"],
            "min_score": min_score,
            "train_dates": list(train_dates),
            "val_dates": list(val_dates),
        },
    }
