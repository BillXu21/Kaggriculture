"""Focused tests for Parquet physical storage of canonical daily records.

The logical extractor schema (D-018) stays authoritative; these tests verify
that `replay_daily.storage` round-trips logical records with semantic equality,
preserves nulls and nested lifecycle state, and that the CLI selects exactly
one output format. Synthetic fixtures reuse the replay builders from
test_replay_daily; the real-replay smoke runs against the ignored local sample.
"""

import copy
import json
from pathlib import Path

import pyarrow as pa
import pytest

from replay_daily.constants import SCHEMA_VERSION
from replay_daily.storage import (
    PARQUET_COMPRESSION,
    RECORD_SCHEMA,
    read_parquet,
    records_to_table,
    write_parquet,
)

from test_replay_daily import MANIFEST, SAMPLE, make_farm, make_replay

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- fixtures


def _plant(crop: str, planted_day: int, yield_units: int, **overrides) -> dict:
    tile = {
        "kind": "PLANT", "crop": crop, "planted_day": planted_day,
        "yield_units": yield_units, "max_lifespan_step": 312,
        "fertilized_until_day": -1, "consecutive_unwatered": 0,
        "watered_today": True,
    }
    tile.update(overrides)
    return tile


def _animal(animal: str, **overrides) -> dict:
    tile = {
        "kind": "PASTURE", "animal": animal, "placed_day": 0, "yield_units": 0,
        "fed_today": False, "cared_today": False, "consecutive_unfed": 0,
        "fertilizer_available": False, "pending_care_bonus": 0,
    }
    tile.update(overrides)
    return tile


def rich_specs() -> list[dict]:
    """One replay exercising every storage-sensitive canonical feature."""
    start_tiles = [
        [_plant("WHEAT", 2, 1), None],
        ["LOCKED", {"kind": "WEED"}],
    ]
    end_tiles = [
        [
            # Ongoing crop watered today -> derived int days_until_next_harvest.
            _plant("STRAWBERRY", 0, 0),
            # Dry non-ongoing crop -> derived present but days_until_next_harvest null.
            _plant("CARROT", 0, 0, watered_today=False),
        ],
        [
            # Animal structure: derived has only the animal-specific keys.
            _animal("COW"),
            {"kind": "WEED"},
        ],
    ]
    sells = [(1, "WHEAT", 2), (23, "EGG", 1), (23, "MELON", 3)]
    specs = [
        {
            "day": 0, "hour": 0,
            "town": {"unlocked_shops": ["PIZZA_SHOP", "PIZZA_SHOP",
                                        "ICE_CREAM_SHOP"]},
            "farms0": make_farm(tiles=start_tiles),
            "private0": {"shed": {"WHEAT": 3}, "seeds": {"CARROT": 2},
                         "inventories": [{"FERTILIZER": 1}, {}]},
        },
        # Acts on obs hour 0: failed BUY_LAND (quadrant must stay null).
        {
            "day": 0, "hour": 1,
            "action0": {"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]},
        },
        # Acts on obs hour 1.
        {
            "day": 0, "hour": 2,
            "action0": {"farmer": ["PASS"], "hands": [],
                        "market": [["SELL", "WHEAT", 2]]},
        },
        # Hour-23 observation carries a COW under the farmer and a hand on a
        # LOCKED tile; the terminal-boundary action below CAREs both actors,
        # so entries are attributed back to day 0 at exact hour 23.
        {"day": 0, "hour": 23,
         "farms0": make_farm(tiles=[[_animal("COW"), None], ["LOCKED", None]],
                             hands=[[1, 1]])},
        # Terminal day boundary; acts on obs hour 23. Harvest + fertilizer +
        # seed buy ledger entries come from the end-tile board state.
        {
            "day": 1, "hour": 0,
            "farms0": make_farm(tiles=end_tiles),
            "action0": {"farmer": ["CARE"], "hands": [["CARE"]],
                        "market": [["SELL", "EGG", 1], ["SELL", "MELON", 3],
                                   ["BUY_SEED", "WHEAT", 5]]},
        },
    ]
    return specs


@pytest.fixture(scope="module")
def rich_records() -> list[dict]:
    from replay_daily.extractor import extract_replay

    return extract_replay(make_replay(rich_specs()))


# ------------------------------------------------- round-trip equality


def test_parquet_round_trip_semantic_equality(rich_records, tmp_path):
    out = tmp_path / "records.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    assert read_parquet(out) == rich_records


def test_arrow_table_direct_access_without_replay_daily_reader(rich_records):
    table = records_to_table(rich_records)
    assert table.schema.equals(RECORD_SCHEMA)
    assert table.column_names == [
        "schema_version", "metadata", "day", "start", "events", "targets", "end",
    ]
    row = table.to_pylist()[0]
    assert row["metadata"]["episode_id"] == 42
    assert row["start"]["self"]["money"] == row["start"]["self"]["money"]
    assert isinstance(row["targets"]["sell_quantity"], list)  # Arrow map pairs


def test_zstd_compression_setting(tmp_path, rich_records):
    import pyarrow.parquet as pq

    out = tmp_path / "c.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    meta = pq.read_metadata(out).row_group(0).column(0)
    assert meta.compression.upper() == PARQUET_COMPRESSION.upper() == "ZSTD"


# ------------------------------------------------- null preservation


def test_buy_land_quadrant_null_preserved(rich_records, tmp_path):
    out = tmp_path / "n.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    rec = read_parquet(out)[0]
    assert rec["events"]["land_purchases"] == [{"quadrant": None, "hour": 0}]


def test_lifecycle_nulls_not_coerced_to_zero(rich_records, tmp_path):
    out = tmp_path / "n.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    rec = read_parquet(out)[0]
    board = rec["end"]["self"]["board"]
    dry_carrot = board[0][1]
    # Present-but-null derived timing must survive as None, not become 0.
    assert dry_carrot["derived"]["days_until_next_harvest"] is None
    # Empty and locked tiles stay distinguishable non-dict sentinels.
    assert board[1][1] == {"kind": "WEED", "derived": None}
    assert rec["start"]["self"]["board"][1][0] == "LOCKED"
    assert rec["start"]["self"]["board"][0][1] is None


def test_absent_vs_null_tile_keys_distinguished(rich_records, tmp_path):
    """Animal tiles never carry plant keys; reconstruction must not invent them."""
    out = tmp_path / "n.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    cow = read_parquet(out)[0]["end"]["self"]["board"][1][0]
    assert set(cow) == set(rich_records[0]["end"]["self"]["board"][1][0])
    assert "crop" not in cow and "planted_day" not in cow
    # Animal derived carries only its own three mechanical keys.
    assert set(cow["derived"]) == {"currently_harvestable",
                                   "days_until_next_product", "starving"}


# ------------------------------------------------- nested state fidelity


def test_nested_crop_animal_raw_and_derived_state(rich_records, tmp_path):
    out = tmp_path / "nested.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    board = read_parquet(out)[0]["end"]["self"]["board"]

    strawberry = board[0][0]
    assert strawberry["crop"] == "STRAWBERRY" and strawberry["planted_day"] == 0
    assert strawberry["derived"]["days_until_next_harvest"] is not None
    assert strawberry["derived"]["age_days"] == 1

    cow = board[1][0]
    assert cow["animal"] == "COW" and cow["placed_day"] == 0
    assert cow["fed_today"] is False and cow["pending_care_bonus"] == 0
    assert cow["derived"]["starving"] is False


def test_six_sell_bins_and_exact_event_hours(rich_records, tmp_path):
    out = tmp_path / "bins.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    rec = read_parquet(out)[0]
    bins = rec["targets"]["sell_quantity"]
    assert sorted(bins) == ["0", "12", "16", "20", "4", "8"]
    assert bins["0"]["WHEAT"] == 2
    assert bins["20"]["EGG"] == 1 and bins["20"]["MELON"] == 3
    assert sum(sum(p.values()) for p in bins.values()) == 6
    hours = sorted(s["hour"] for s in rec["events"]["sells"])
    assert hours == [1, 23, 23]  # exact primitive sale hours retained
    assert ["SELL", "WHEAT", 2, 1] in rec["events"]["market_events_ordered"]


def test_town_duplicate_multiplicity(rich_records, tmp_path):
    out = tmp_path / "town.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    town = read_parquet(out)[0]["start"]["town"]
    assert town["unlocked_shops"] == ["PIZZA_SHOP", "PIZZA_SHOP",
                                      "ICE_CREAM_SHOP"]
    assert town["shop_counts"] == {"PIZZA_SHOP": 2, "ICE_CREAM_SHOP": 1}


def test_both_seats_and_final_day_rows(tmp_path):
    from replay_daily.extractor import extract_replay

    specs = [{"day": 28, "hour": 23}, {"day": 29, "hour": 0},
             {"day": 29, "hour": 23}]
    records = extract_replay(make_replay(specs))
    out = tmp_path / "seats.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(records, out)
    got = read_parquet(out)
    assert len(got) == len(records) == 2
    assert {(r["metadata"]["seat"], r["day"]) for r in got} == \
        {(0, 29), (1, 29)}
    for rec in got:
        assert rec["end"]["boundary"] == "terminal"
    assert got == records


def test_private_banks_and_provenance_round_trip(rich_records, tmp_path):
    out = tmp_path / "meta.parquet"
    from replay_daily.storage import write_parquet

    write_parquet(rich_records, out)
    rec = read_parquet(out)[0]
    meta = rec["metadata"]
    assert meta["final_rewards"] == [100.0, 200.0]
    assert meta["final_bank_self"] == 100.0
    assert meta["final_bank_opponent"] == 200.0
    assert meta["seed"] == 12345 and meta["module_version"] == "1.32.7"
    assert rec["start"]["self"]["shed"] == {"WHEAT": 3}
    assert rec["start"]["self"]["inventories"] == [{"FERTILIZER": 1}, {}]


# ------------------------------------------------- determinism


def test_deterministic_logical_regeneration_and_order(rich_records, tmp_path):
    from replay_daily.storage import write_parquet

    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_parquet(rich_records, first)
    write_parquet(rich_records, second)
    reread_a, reread_b = read_parquet(first), read_parquet(second)
    assert reread_a == reread_b == rich_records
    # Row order is stable across writes and reads.
    keys_a = [(r["metadata"]["episode_id"], r["metadata"]["seat"], r["day"])
              for r in reread_a]
    keys_b = [(r["metadata"]["episode_id"], r["metadata"]["seat"], r["day"])
              for r in reread_b]
    assert keys_a == keys_b


# ------------------------------------------------- CARE ledger (schema v2)


def test_care_round_trip_exact_including_null_animal(rich_records, tmp_path):
    out = tmp_path / "care.parquet"
    write_parquet(rich_records, out)
    rec = read_parquet(out)[0]
    assert rec["schema_version"] == SCHEMA_VERSION == 2
    assert rec["events"]["care"] == {
        "by_animal": {"GOOSE": 0, "COW": 1, "SHEEP": 0},
        "entries": [
            {"tile": [0, 0], "animal": "COW", "hour": 23},
            {"tile": [1, 1], "animal": None, "hour": 23},
        ],
    }
    assert rec["targets"]["care_by_animal"] == {"GOOSE": 0, "COW": 1, "SHEEP": 0}


def test_care_absent_from_worker_ops_other_after_round_trip(
        rich_records, tmp_path):
    out = tmp_path / "no_double_count.parquet"
    write_parquet(rich_records, out)
    rec = read_parquet(out)[0]
    assert "CARE" not in rec["events"]["worker_ops_other"]
    assert sum(rec["events"]["care"]["by_animal"].values()) == \
        len(rec["events"]["care"]["entries"]) - 1  # one unknown entry


# ------------------------------------------------- schema version policy


def test_writer_rejects_v1_logical_records(rich_records):
    records = copy.deepcopy(rich_records)
    records[0]["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version.*1"):
        records_to_table(records)


def _write_rows_with_version(rows: list[dict], path: Path) -> None:
    import pyarrow.parquet as pq

    pq.write_table(
        pa.Table.from_pylist(rows, schema=RECORD_SCHEMA), path,
        compression=PARQUET_COMPRESSION,
    )


def test_reader_rejects_v1_and_mixed_version_parquet(rich_records, tmp_path):
    rows = records_to_table(rich_records).to_pylist()

    v1_rows = copy.deepcopy(rows)
    for row in v1_rows:
        row["schema_version"] = 1
    mixed_rows = copy.deepcopy(rows)
    mixed_rows[0]["schema_version"] = 1

    for name, bad_rows in (("v1.parquet", v1_rows), ("mixed.parquet", mixed_rows)):
        out = tmp_path / name
        _write_rows_with_version(bad_rows, out)
        with pytest.raises(ValueError, match="schema_version"):
            read_parquet(out)


def test_cli_inspect_rejects_v1_jsonl(tmp_path):
    from replay_daily.cli import main as cli_main_fn

    out = tmp_path / "v1.jsonl"
    out.write_text('{"schema_version": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        cli_main_fn(["inspect", str(out)])


# ------------------------------------------------- CLI format selection


def _cli_extract(cli_main, tmp_path, fmt_args, name):
    from replay_daily.cli import main as cli_main_fn

    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(make_replay(rich_specs())), encoding="utf-8")
    out = tmp_path / name
    rc = cli_main_fn(["extract", "--input", str(replay), "--output", str(out),
                      *fmt_args])
    return rc, out


def test_cli_default_format_is_parquet_with_no_jsonl_duplicate(tmp_path):
    from replay_daily.cli import main as cli_main_fn

    rc, out = _cli_extract(cli_main_fn, tmp_path, [], "default.parquet")
    assert rc == 0
    assert out.exists()
    assert not out.with_suffix(".jsonl").exists()
    assert len(read_parquet(out)) == 4


def test_cli_explicit_format_selection_writes_exactly_one_output(tmp_path):
    from replay_daily.cli import main as cli_main_fn

    rc_pq, pq_out = _cli_extract(cli_main_fn, tmp_path,
                                 ["--format", "parquet"], "explicit.parquet")
    rc_jl, jl_out = _cli_extract(cli_main_fn, tmp_path,
                                 ["--format", "jsonl"], "debug.jsonl")
    assert rc_pq == 0 and rc_jl == 0
    assert pq_out.exists() and not pq_out.with_suffix(".jsonl").exists()
    assert jl_out.exists() and not jl_out.with_suffix(".parquet").exists()

    jsonl_records = [json.loads(line)
                     for line in jl_out.read_text(encoding="utf-8").splitlines()]
    assert jsonl_records == read_parquet(pq_out)


def test_cli_rejects_mismatched_output_extension(tmp_path):
    from replay_daily.cli import main as cli_main_fn

    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(make_replay(rich_specs())), encoding="utf-8")

    bad_jsonl = tmp_path / "wrong.jsonl"
    rc = cli_main_fn(["extract", "--input", str(replay), "--output", str(bad_jsonl),
                      "--format", "parquet"])
    assert rc == 2

    bad_parquet = tmp_path / "wrong.parquet"
    rc = cli_main_fn(["extract", "--input", str(replay), "--output", str(bad_parquet),
                      "--format", "jsonl"])
    assert rc == 2
    assert not bad_jsonl.exists() and not bad_parquet.exists()


def test_cli_inspect_reads_both_formats(tmp_path, capsys):
    from replay_daily.cli import main as cli_main_fn

    rc_pq, pq_out = _cli_extract(cli_main_fn, tmp_path,
                                 ["--format", "parquet"], "inspect.parquet")
    rc_jl, jl_out = _cli_extract(cli_main_fn, tmp_path,
                                 ["--format", "jsonl"], "inspect.jsonl")
    assert rc_pq == 0 and rc_jl == 0
    capsys.readouterr()  # discard extract stdout before inspecting
    for path in (pq_out, jl_out):
        rc = cli_main_fn(["inspect", str(path), "--seat", "0", "--day", "0"])
        assert rc == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["metadata"]["seat"] == 0 and printed["day"] == 0


# ------------------------------------------------- schema-drift guards


def test_unknown_top_level_record_key_fails_loudly(rich_records):
    records = copy.deepcopy(rich_records)
    records[0]["future_section"] = {}
    with pytest.raises(ValueError, match="record.*future_section"):
        records_to_table(records)


def test_unknown_metadata_key_fails_loudly(rich_records):
    records = copy.deepcopy(rich_records)
    records[0]["metadata"]["new_provenance"] = "x"
    with pytest.raises(ValueError, match="metadata.*new_provenance"):
        records_to_table(records)


def test_unknown_tile_and_derived_keys_fail_loudly(rich_records):
    records = copy.deepcopy(rich_records)
    strawberry = records[0]["end"]["self"]["board"][0][0]
    assert strawberry["kind"] == "PLANT"
    strawberry["engine_added_field"] = 7
    with pytest.raises(ValueError,
                       match=r"board\[0\]\[0\].*engine_added_field"):
        records_to_table(records)

    records = copy.deepcopy(rich_records)
    records[0]["end"]["self"]["board"][0][0]["derived"]["guessed_days"] = 3
    with pytest.raises(ValueError,
                       match=r"board\[0\]\[0\]\.derived.*guessed_days"):
        records_to_table(records)


def test_unknown_fixed_event_entry_keys_fail_loudly(rich_records):
    records = copy.deepcopy(rich_records)
    records[0]["events"]["sells"][0]["fee"] = 1
    with pytest.raises(ValueError, match=r"events\.sells\[0\].*fee"):
        records_to_table(records)

    records = copy.deepcopy(rich_records)
    records[0]["events"]["land_purchases"][0]["price"] = 100
    with pytest.raises(ValueError,
                       match=r"land_purchases\[0\].*price"):
        records_to_table(records)


def test_dynamic_map_keys_round_trip(rich_records, tmp_path):
    """Product/crop/op-name map keys are open-ended and must never be guarded."""
    records = copy.deepcopy(rich_records)
    records[0]["events"]["plants"]["ALIEN_CROP"] = 2
    records[0]["events"]["worker_ops_other"]["SOMETHING_NEW"] = 1
    records[0]["start"]["self"]["shed"]["MYSTERY_ITEM"] = 9
    records[0]["targets"]["sell_quantity"]["0"]["PLASMA"] = 4
    out = tmp_path / "dynamic.parquet"
    write_parquet(records, out)
    assert read_parquet(out) == records


# ------------------------------------------------- string sentinels


def test_bare_string_sentinels_round_trip_exactly(rich_records, tmp_path):
    records = copy.deepcopy(rich_records)
    board = records[0]["start"]["self"]["board"]
    assert board[0][1] is None
    board[0][1] = "WEED"  # bare string sentinel, distinct from dict WEED
    out = tmp_path / "sentinels.parquet"
    write_parquet(records, out)
    got = read_parquet(out)[0]["start"]["self"]["board"]
    assert got[0][1] == "WEED" and isinstance(got[0][1], str)
    assert got[1][0] == "LOCKED" and isinstance(got[1][0], str)
    # Dict WEED stays a dict with its derived key.
    end_board = read_parquet(out)[0]["end"]["self"]["board"]
    assert end_board[1][1] == {"kind": "WEED", "derived": None}


# ------------------------------------------------- real replay smoke


@pytest.mark.skipif(not SAMPLE.exists(), reason="local real sample not present")
def test_real_replay_parquet_smoke_semantic_parity(tmp_path):
    from replay_daily.cli import main as cli_main_fn

    pq_out = tmp_path / "real.parquet"
    jl_out = tmp_path / "real.jsonl"
    common = ["--input", str(SAMPLE), "--manifest", str(MANIFEST),
              "--source-dataset", "kaggle/kaggriculture-episodes-2026-08-20",
              "--partition-date", "2026-08-20"]
    assert cli_main_fn(["extract", *common, "--output", str(pq_out)]) == 0
    assert cli_main_fn(["extract", *common, "--format", "jsonl",
                        "--output", str(jl_out)]) == 0

    parquet_records = read_parquet(pq_out)
    jsonl_records = [json.loads(line)
                     for line in jl_out.read_text(encoding="utf-8").splitlines()]

    # 60 logical records: both seats, days 0..29, stable order.
    assert len(parquet_records) == 60
    keys = [(r["metadata"]["seat"], r["day"]) for r in parquet_records]
    assert keys == [(seat, day) for seat in (0, 1) for day in range(30)]

    # Semantic parity against independent JSONL extraction of the same replays.
    assert parquet_records == jsonl_records

    final_days = [r for r in parquet_records if r["day"] == 29]
    assert {r["metadata"]["seat"] for r in final_days} == {0, 1}
    for rec in final_days:
        assert rec["end"]["boundary"] == "terminal"

    def no_opponent_private(node) -> bool:
        if isinstance(node, dict):
            return all(k not in ("shed", "seeds", "inventories", "private")
                       and no_opponent_private(v) for k, v in node.items())
        if isinstance(node, list):
            return all(no_opponent_private(v) for v in node)
        return True

    for rec in parquet_records:
        assert no_opponent_private(rec["start"]["opponent_public"])
        assert no_opponent_private(rec["end"]["opponent_public"])
