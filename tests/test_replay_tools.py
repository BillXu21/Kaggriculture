"""Focused tests for tools/replay_io.py, tools/replay_manifest.py, and
tools/expert_plan.py.

Unit tests run against a synthetic minimal replay dict built in-file.
One integration test exercises the real primary samples directory
(outside this worktree) and skips when it is absent.
"""

from pathlib import Path

import pytest

from executor_v0.plan import DailyPlan
from replay_daily.constants import PRODUCTS
from tools.expert_plan import (
    board_counts,
    boundary_observation,
    collect_day_events,
    end_of_day_observation,
    extract_daily_plan,
)
from tools.replay_io import (
    episode_configuration,
    episode_id,
    load_replay,
)
from tools.replay_manifest import build_manifest, classify, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\samples"
)


# --------------------------------------------------------- synthetic replay


def _farm() -> dict:
    tiles = [["EMPTY"] * 4 for _ in range(4)]
    tiles[1][2] = {"kind": "COOP", "animal": "GOOSE"}  # pos [x=2, y=1]
    tiles[2][1] = {"kind": "PLANT", "crop": "WHEAT"}   # pos [x=1, y=2]
    tiles[3][0] = "WEED"                               # bare string weed
    tiles[0][3] = {"kind": "WEED"}                     # dict-form weed
    tiles[3][3] = {"kind": "PASTURE"}                  # empty structure
    return {
        "tiles": tiles,
        "farmer": [2, 1],      # standing on the GOOSE coop
        "hands": [[2, 1], [1, 2]],  # on GOOSE coop / on WHEAT plant
        "unlocked_quadrants": ["NW"],
        "money": 1000,
        "hires_today": 0,
    }


def _opponent_farm() -> dict:
    return {
        "tiles": [["EMPTY"] * 4 for _ in range(4)],
        "farmer": [0, 0],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "money": 1000,
        "hires_today": 0,
    }


def _obs(day: int, hour: int) -> dict:
    return {
        "day": day,
        "hour": hour,
        "farms": [_farm(), _opponent_farm()],
    }


NOOP = {"farmer": ["PASS"], "hands": [], "market": []}


def _step(day: int, hour: int, action0: dict, action1: dict) -> dict:
    return [
        {"action": action0, "observation": _obs(day, hour),
         "reward": 0, "status": "ACTIVE"},
        {"action": action1, "observation": _obs(day, hour),
         "reward": 0, "status": "ACTIVE"},
    ]


def make_replay() -> dict:
    """Minimal 2-player replay with known attributed events.

    Step timeline for seat 0 (each entry's observation is the POST state;
    the action at step i is attributed to the PRE observation at i-1):
      i=1: pre day0 h10 -> SELL WHEAT x3 (bin anchor 8), HIRE
      i=2: pre day0 h21 -> SELL CARROT x2 (bin anchor 20); post obs day1 h0
      i=3: pre day1 h0  -> farmer CARE (GOOSE), hands CARE + FERTILIZE (WHEAT)
    """
    sell_morning = {
        "farmer": ["PASS"], "hands": [],
        "market": [["SELL", "WHEAT", 3], ["HIRE"]],
    }
    sell_evening = {
        "farmer": ["PASS"], "hands": [],
        "market": [["SELL", "CARROT", 2]],
    }
    care_and_fert = {
        "farmer": ["CARE"],
        "hands": [["CARE"], ["FERTILIZE"]],
        "market": [],
    }
    return {
        "configuration": {"turnsPerDay": 24, "episodeSteps": 720, "seed": 5},
        "info": {
            "EpisodeId": 12345,
            "seed": 99,
            "TeamNames": ["Alpha", "Beta"],
        },
        "rewards": [50000.0, 42000.0],
        "steps": [
            _step(0, 10, NOOP, NOOP),
            _step(0, 21, sell_morning, NOOP),
            _step(1, 0, sell_evening, NOOP),   # post obs crosses into day 1
            _step(1, 3, care_and_fert, NOOP),
        ],
    }


@pytest.fixture()
def replay() -> dict:
    return make_replay()


# ------------------------------------------------------------ replay_io


class TestEpisodeConfiguration:
    def test_seed_override_from_info(self, replay):
        config = episode_configuration(replay)
        assert config["seed"] == 99  # info.seed overrides configuration.seed

    def test_seed_fallback_to_configuration(self, replay):
        del replay["info"]["seed"]
        assert episode_configuration(replay)["seed"] == 5

    def test_turns_per_day_validation_raises(self, replay):
        replay["configuration"]["turnsPerDay"] = 20
        with pytest.raises(ValueError, match="turnsPerDay"):
            episode_configuration(replay)

    def test_episode_steps_validation_raises(self, replay):
        replay["configuration"]["episodeSteps"] = 360
        with pytest.raises(ValueError, match="episodeSteps"):
            episode_configuration(replay)

    def test_episode_id(self, replay):
        assert episode_id(replay) == 12345

    def test_load_replay_errors(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_replay(str(bad))
        with pytest.raises(ValueError, match="cannot read"):
            load_replay(str(tmp_path / "missing.json"))


# ---------------------------------------------------------- classify


class TestClassify:
    def test_failure_specimen_by_episode_id(self):
        assert classify(98178196, [0.0, 0.0]) == "failure_specimen"

    def test_high_reward(self):
        assert classify(111, [50000.0, 40000.0]) == "high_reward"

    def test_unknown_low_reward(self):
        assert classify(111, [39999.9, 80000.0]) == "unknown"


# ------------------------------------------------- collect_day_events


class TestCollectDayEvents:
    def test_attribution_to_previous_observation_day_hour(self, replay):
        events = collect_day_events(replay, 0, 0)
        # The CARROT sale at i=2 has a post observation already at day 1 h0;
        # it must be attributed to day 0 hour 21 of the PRE observation.
        assert events["sells"] == [(10, "WHEAT", 3), (21, "CARROT", 2)]
        assert events["hires_submitted"] == 1
        assert events["land_buys"] == 0

    def test_care_and_fertilizer_via_tile_under_position(self, replay):
        events = collect_day_events(replay, 0, 1)
        # Farmer [2,1] -> tiles[1][2] GOOSE coop; hand0 [2,1] CARE also GOOSE;
        # hand1 [1,2] FERTILIZE -> tiles[2][1] PLANT WHEAT.
        assert events["care_counts"] == {"GOOSE": 2}
        assert events["fert_counts"] == {"WHEAT": 1}
        assert events["sells"] == []

    def test_no_events_for_other_seat(self, replay):
        for day in (0, 1):
            events = collect_day_events(replay, 1, day)
            assert events["sells"] == []
            assert events["care_counts"] == {}
            assert events["fert_counts"] == {}


# ------------------------------------------------------ board boundaries


class TestBoardCountsAndBoundaries:
    def test_board_counts_both_weed_forms(self, replay):
        crops, animals, weeds, empty_structures = board_counts(
            _obs(0, 0), seat=0
        )
        assert crops["WHEAT"] == 1
        assert animals["GOOSE"] == 1
        assert weeds == 2          # "WEED" string + {"kind": "WEED"}
        assert empty_structures == 1  # PASTURE without animal

    def test_boundary_observation(self, replay):
        start = boundary_observation(replay, 0, 1)
        assert start["day"] == 1 and start["hour"] == 0
        with pytest.raises(ValueError, match="start-of-day"):
            boundary_observation(replay, 0, 7)

    def test_end_of_day_next_day_start_vs_terminal(self, replay):
        assert end_of_day_observation(replay, 0, 0)["day"] == 1  # day1 h0
        terminal = end_of_day_observation(replay, 0, 1)  # no day 2
        assert terminal is replay["steps"][-1][0]["observation"]


# ------------------------------------------------------ extract_daily_plan


class TestExtractDailyPlan:
    def test_sell_bin_grouping(self, replay):
        plan = extract_daily_plan(replay, 0, 0)
        assert isinstance(plan, DailyPlan)
        rows = dict(zip(PRODUCTS, plan.sell_quantities))
        assert all(len(row) == 6 for row in plan.sell_quantities)
        assert rows["WHEAT"] == (0, 0, 3, 0, 0, 0)   # hour 10 -> bin anchor 8
        assert rows["CARROT"] == (0, 0, 0, 0, 0, 2)  # hour 21 -> bin anchor 20
        assert rows["MILK"] == (0, 0, 0, 0, 0, 0)

    def test_targets_land_and_caps(self, replay):
        plan = extract_daily_plan(replay, 0, 1)
        # End-of-day state (terminal obs): WHEAT plant + GOOSE.
        assert plan.crop_targets_dict["WHEAT"] == 1
        assert plan.animal_targets_dict["GOOSE"] == 1
        assert plan.land_count == 1
        # care GOOSE submitted twice but capped at end-of-day count 1;
        # fertilizer WHEAT capped at end-of-day plant count 1.
        assert plan.care_by_animal_dict["GOOSE"] == 1
        assert plan.fertilizer_by_crop_dict["WHEAT"] == 1

    def test_all_fields_nonnegative_ints(self, replay):
        for day in (0, 1):
            plan = extract_daily_plan(replay, 0, day)
            for vector in (
                plan.crop_targets, plan.animal_targets,
                plan.fertilizer_by_crop, plan.care_by_animal,
            ):
                assert all(isinstance(v, int) and v >= 0 for v in vector)
            assert isinstance(plan.land_count, int) and plan.land_count >= 0
            assert all(
                isinstance(q, int) and q >= 0
                for row in plan.sell_quantities for q in row
            )


# ------------------------------------------------------------- manifest


class TestManifest:
    def test_write_manifest_roundtrip(self, tmp_path, replay):
        manifest = [{"file": "a.json", "episode_id": 1}]
        out = tmp_path / "m.json"
        write_manifest(manifest, out)
        text = out.read_text(encoding="utf-8")
        import json

        assert json.loads(text) == manifest
        assert json.dumps(manifest, indent=2, sort_keys=True) in text

    def test_unreadable_file_yields_error_entry(self, tmp_path):
        (tmp_path / "broken.json").write_text("{oops", encoding="utf-8")
        manifest = build_manifest(tmp_path)
        assert len(manifest) == 1
        assert manifest[0]["file"] == "broken.json"
        assert "error" in manifest[0]


# ------------------------------------------------------- real-sample tests


@pytest.mark.skipif(
    not SAMPLES_DIR.is_dir(), reason="primary samples dir not available"
)
class TestRealSamples:
    def test_manifest_over_real_samples_dir(self):
        manifest = build_manifest(SAMPLES_DIR)
        assert len(manifest) == 19
        by_episode = {
            entry["episode_id"]: entry
            for entry in manifest
            if "episode_id" in entry
        }
        specimen = by_episode[98178196]
        assert specimen["classification"] == "failure_specimen"
        assert specimen["file"] == "datasamplesbc_e_mirror_98178196.json"

    def test_extract_daily_plan_real_replay(self):
        replay = load_replay(str(SAMPLES_DIR / "98093786.json"))
        plan = extract_daily_plan(replay, 0, 3)
        assert isinstance(plan, DailyPlan)
        vectors = (
            plan.crop_targets, plan.animal_targets,
            plan.fertilizer_by_crop, plan.care_by_animal,
        )
        assert all(
            isinstance(v, int) and v >= 0
            for vector in vectors for v in vector
        )
        assert 1 <= plan.land_count <= 4
        assert len(plan.sell_quantities) == len(PRODUCTS)
        assert all(len(row) == 6 for row in plan.sell_quantities)
        assert all(
            isinstance(q, int) and q >= 0
            for row in plan.sell_quantities for q in row
        )
