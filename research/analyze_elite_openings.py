"""Memory-bounded elite-opening analysis for Kaggriculture 1.32.7 (issue #3).

Selects the strongest locally available both-seats-strong episodes from the
canonical schema-v3 sample Parquet by descending ``min(final_bank_seat0,
final_bank_seat1)``, optionally fetches one remote episode replay fully
in memory (never written to disk), and emits:

- the selected cohort table;
- compact per-(seat, day) opening summaries for the first few days;
- ordered day-0..N market-event sequences with cross-replay clustering;
- representative exact hour traces (full submitted primitive actions);
- internal consistency assertions.

Memory contract (deliberate, see issue #3):
- The Parquet is read with column projection (``metadata`` only); the corpus
  is never converted to Python rows via ``to_pylist()`` over all columns.
- Raw replays (~30 MB JSON each) are loaded ONE AT A TIME, processed, and
  released before the next is loaded. No full-corpus materialization.
- The remote replay is kept as in-memory bytes only.

Usage:
    python research/analyze_elite_openings.py \
        [--parquet data/canonical/2026-08-20-sample.parquet] \
        [--samples-dir data/samples/2026-08-20] \
        [--top 10] [--days 4] [--remote 95531759] \
        [--trace SEAT:DAY:HOUR ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_daily.constants import ENGINE_VERSION  # noqa: E402
from replay_daily.extractor import extract_replay  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Cohort selection (Arrow projection only; no corpus-wide Python rows)
# ---------------------------------------------------------------------------

def select_cohort(parquet: Path, top: int) -> list[dict]:
    table = pq.read_table(str(parquet), columns=["metadata"])
    col = table.column("metadata").combine_chunks()
    episodes: dict[int, dict] = {}
    for i in range(len(col)):
        r = col[i]
        eid = r["episode_id"].as_py()
        ep = episodes.setdefault(
            eid,
            {"episode_id": eid, "seed": r["seed"].as_py(),
             "module_version": r["module_version"].as_py(),
             "partition_date": r["partition_date"].as_py(),
             "players": {}, "banks": {}},
        )
        seat = r["seat"].as_py()
        ep["players"][seat] = r["player"].as_py()
        ep["banks"][seat] = r["final_bank_self"].as_py()
    rows = []
    for ep in episodes.values():
        banks = [ep["banks"][s] for s in sorted(ep["banks"])]
        if len(banks) != 2 or any(b is None for b in banks):
            raise AssertionError(f"episode {ep['episode_id']}: missing final bank")
        ep["min_bank"] = min(banks)
        ep["banks_sorted"] = banks
        rows.append(ep)
    rows.sort(key=lambda e: (-e["min_bank"], e["episode_id"]))
    return rows[:top]


def local_replay_path(samples_dir: Path, episode_id: int) -> Path | None:
    p = samples_dir / f"{episode_id}.json"
    return p if p.exists() else None


def load_local_replay(path: Path) -> dict:
    """Load exactly one raw replay; caller must release before the next."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_remote_replay(episode_id: int) -> dict:
    """Fetch one episode replay fully in memory (kaggle API, no disk write)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import (
        ApiGetEpisodeReplayRequest,
    )

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiGetEpisodeReplayRequest()
        request.episode_id = episode_id
        response = kaggle.competitions.competition_api_client.get_episode_replay(request)
        raw = response.content
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Hour-index helpers over one replay
# ---------------------------------------------------------------------------

def seat_hour_index(replay: dict, seat: int) -> dict[tuple[int, int], int]:
    """Map (day, hour) -> step index whose observation starts that hour."""
    idx: dict[tuple[int, int], int] = {}
    for i, step in enumerate(replay["steps"]):
        obs = step[seat].get("observation") or {}
        key = (obs.get("day"), obs.get("hour"))
        if key not in idx:
            idx[key] = i
    return idx


def hourly_money(replay: dict, seat: int, max_day: int):
    out = []
    for step in replay["steps"]:
        obs = step[seat].get("observation") or {}
        if obs.get("day", 99) > max_day:
            continue
        farm = obs["farms"][seat]
        out.append((obs["day"], obs["hour"], farm["money"],
                    len(farm.get("hands") or []), farm.get("hires_today", 0)))
    return out


def town_unlock_first(replay: dict, max_day: int):
    seen: set[str] = set()
    for step in replay["steps"][0:]:
        obs = step[0].get("observation") or {}
        if obs.get("day", 99) > max_day:
            continue
        shops = tuple(sorted((obs.get("town") or {}).get("unlocked_shops") or []))
        if shops and shops not in seen:
            seen.add(shops)
            yield obs["day"], obs["hour"], list(shops)


# ---------------------------------------------------------------------------
# Compact summaries
# ---------------------------------------------------------------------------

def day_summary(records: list[dict], max_day: int) -> list[dict]:
    rows = []
    for rec in records:
        if rec["day"] > max_day:
            continue
        ev = rec["events"]
        rows.append({
            "day": rec["day"],
            "money_start": rec["start"]["self"]["money"],
            "money_end": rec["end"]["self"]["money"],
            "hands_end": len(rec["end"]["self"]["hands"]),
            "hires_submitted": ev["hires"]["submitted"],
            "hires_realized": ev["hires"]["realized"]["workers_hired"],
            "hire_cost": ev["hires"]["realized"]["hire_cost"],
            "buy_seeds": ev["buys"]["seeds"],
            "buy_products": ev["buys"]["products"],
            "buy_animals": ev["buys"]["animals"],
            "sells": [(s["product"], s["quantity"], s["hour"]) for s in ev["sells"]],
            "plants": ev["plants"],
            "harvests": ev["harvests"]["by_item"],
            "first_harvest_hour": min(
                (e["hour"] for e in ev["harvests"]["entries"]), default=None),
            "land": [(l["quadrant"], l["hour"]) for l in ev["land_purchases"]],
            "animals_end": {k: v for k, v in rec["targets"]["animal_counts_end"].items() if v},
            "crops_end": {k: v for k, v in rec["targets"]["crop_composition_end"].items() if v},
            "fertilizer": ev["fertilizer_applications"]["by_crop"],
            "care": {k: v for k, v in ev["care"]["by_animal"].items() if v},
            "worker_ops_other": ev["worker_ops_other"],
            "market_events": [tuple(m) for m in ev["market_events_ordered"]],
        })
    return rows


def market_signature(rows: list[dict], day: int):
    """Ordered (op, *args) market events of `day`, hours dropped."""
    for r in rows:
        if r["day"] == day:
            return tuple(tuple(m[:-1]) for m in r["market_events"])
    return tuple()


def exact_hour_trace(replay: dict, seat: int, day: int, hour: int) -> dict | None:
    """Full submitted action acting on the (day, hour) observation."""
    idx = seat_hour_index(replay, seat)
    i = idx.get((day, hour))
    if i is None or i + 1 >= len(replay["steps"]):
        return None
    pre = replay["steps"][i][seat]["observation"]
    post = replay["steps"][i + 1][seat].get("observation") or {}
    action = replay["steps"][i + 1][seat].get("action")
    market = pre.get("market") or {}
    return {
        "seat": seat, "day": day, "hour": hour,
        "money_before": pre["farms"][seat]["money"],
        "money_after": (post.get("farms") or [None])[seat].get("money")
                       if post else None,
        "market_inventory_wheat": (market.get("inventory") or {}).get("WHEAT"),
        "market_price_wheat": (market.get("prices") or {}).get("WHEAT"),
        "action": action,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="data/canonical/2026-08-20-sample.parquet")
    ap.add_argument("--samples-dir", default="data/samples/2026-08-20")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--remote", type=int, nargs="*", default=[95531759])
    ap.add_argument("--trace", nargs="*", default=[],
                    help="exact hour traces SEAT:DAY:HOUR (applies to every replay)")
    args = ap.parse_args()

    cohort = select_cohort(REPO / args.parquet, args.top)
    selected = []
    for ep in cohort:
        p = local_replay_path(REPO / args.samples_dir, ep["episode_id"])
        selected.append({**ep, "path": p})
    for rid in args.remote:
        selected.append({"episode_id": rid, "path": None, "remote": True})

    # ---- assertions: cohort shape -----------------------------------------
    assert 8 <= len(selected) <= 12, f"cohort size {len(selected)} outside 8..12"
    ids = [e["episode_id"] for e in selected]
    assert len(set(ids)) == len(ids), "duplicate selected episodes"
    assert 95531759 in ids, "95531759 missing"

    print("# Cohort")
    for e in selected:
        src = e["path"].name if e.get("path") else "REMOTE(in-memory)"
        players = e.get("players") or {}
        banks = e.get("banks_sorted") or "(remote: see replay)"
        print(f"- {e['episode_id']} seed={e.get('seed')} module={e.get('module_version')} "
              f"banks={banks} players={players} src={src}")

    sigs_by_day: dict[int, list[tuple[str, tuple]]] = collections.defaultdict(list)
    traces_req = []
    for t in args.trace:
        s, d, h = (int(x) for x in t.split(":"))
        traces_req.append((s, d, h))

    for e in selected:
        replay = (load_local_replay(e["path"]) if e.get("path")
                  else fetch_remote_replay(e["episode_id"]))
        try:
            mv = replay.get("module_version")
            assert mv == ENGINE_VERSION, f"{e['episode_id']}: module {mv}"
            info = replay.get("info") or {}
            seed = info.get("seed")
            if e.get("seed") is not None:
                assert seed == e["seed"], f"{e['episode_id']}: seed {seed} != {e['seed']}"
            rewards = replay.get("rewards")
            assert rewards and all(r is not None for r in rewards[:2]), \
                f"{e['episode_id']}: missing final banks"
            e.setdefault("players", {})
            e["players"] = {i: n for i, n in enumerate(info.get("TeamNames") or [])}
            e["banks_sorted"] = sorted(rewards[:2], reverse=True)

            records = extract_replay(replay)
            print(f"\n## Episode {e['episode_id']} seed={seed} "
                  f"teams={info.get('TeamNames')} rewards={rewards}")
            for seat in (0, 1):
                rows = day_summary([r for r in records if r["metadata"]["seat"] == seat],
                                   args.days)
                hm = {(d, h): (m, hands, ht) for d, h, m, hands, ht in hourly_money(replay, seat, args.days)}
                min_cash = min(v[0] for v in hm.values()) if hm else None
                print(f"\n### seat {seat} ({e['players'].get(seat)}) min_cash_d0-{args.days}={min_cash}")
                for r in rows:
                    print(f"- d{r['day']}: money {r['money_start']}->{r['money_end']} "
                          f"hands_end={r['hands_end']} hires(sub/real/cost)="
                          f"{r['hires_submitted']}/{r['hires_realized']}/{r['hire_cost']} "
                          f"seeds={r['buy_seeds']} products={r['buy_products']} "
                          f"animals={r['buy_animals']} sells={r['sells']}")
                    print(f"    plants={r['plants']} harvests={r['harvests']} "
                          f"first_harv_h={r['first_harvest_hour']} land={r['land']}")
                    print(f"    crops_end={r['crops_end']} animals_end={r['animals_end']} "
                          f"fert={r['fertilizer']} care={r['care']}")
                    print(f"    other_ops={r['worker_ops_other']}")
                    print(f"    market: {r['market_events']}")
                    # monotonic day/hour ordering check
                    hours = [m[-1] for m in r["market_events"]]
                    assert hours == sorted(hours), f"non-monotonic hours d{r['day']}"
                sigs_by_day[0].append(
                    (f"{e['episode_id']}#s{seat}", market_signature(rows, 0)))
                for d in range(1, args.days + 1):
                    sigs_by_day[d].append(
                        (f"{e['episode_id']}#s{seat}", market_signature(rows, d)))

            for (s, d, h) in traces_req:
                tr = exact_hour_trace(replay, s, d, h)
                if tr:
                    print(f"\n### TRACE ep={e['episode_id']} seat={s} d{d}h{h}\n"
                          f"{json.dumps(tr, separators=(',', ':'))}")

            # day-0 impossible-harvest guard: WHEAT cannot be harvested on day 0
            for rec in records:
                if rec["metadata"]["seat"] in (0, 1) and rec["day"] == 0:
                    assert "WHEAT" not in rec["events"]["harvests"]["by_item"], \
                        "impossible day-0 WHEAT harvest"

            unlocks = list(town_unlock_first(replay, args.days))
            print(f"\n### town unlocks (seat-0 view): {unlocks[:4]}")

            # Focused issue-#3 assertions for episode 95531759 (wheat trade):
            # BUY_PRODUCT WHEAT 6 at d0h0, partial recovery SELL WHEAT 3 at
            # d0h1 (+84 cash), JIT feed top-ups 2@d0h6 and 1@d0h12.
            if e["episode_id"] == 95531759:
                for seat in (0, 1):
                    rows_s = day_summary(
                        [r for r in records if r["metadata"]["seat"] == seat], 0)
                    ev0 = rows_s[0]["market_events"]
                    assert ("BUY_PRODUCT", "WHEAT", 6, 0) in [
                        tuple(m) for m in ev0], "missing d0h0 WHEAT 6 buy"
                    assert ("SELL", "WHEAT", 3, 1) in [
                        tuple(m) for m in ev0], "missing d0h1 WHEAT 3 sell"
                    assert ("BUY_PRODUCT", "WHEAT", 2, 6) in [
                        tuple(m) for m in ev0], "missing d0h6 WHEAT 2 top-up"
                    assert ("BUY_PRODUCT", "WHEAT", 1, 12) in [
                        tuple(m) for m in ev0], "missing d0h12 WHEAT 1 top-up"
                    tr_sell = exact_hour_trace(replay, seat, 0, 1)
                    assert tr_sell is not None
                    delta = tr_sell["money_after"] - tr_sell["money_before"]
                    assert delta == 84.0, f"d0h1 sell delta {delta} != 84"
                    assert tr_sell["action"]["market"] == [["SELL", "WHEAT", 3]]
        finally:
            del replay

    print("\n# Day-0 market-signature clusters")
    c0 = collections.Counter(sig for _, sig in sigs_by_day[0])
    for sig, n in c0.most_common():
        members = [k for k, s in sigs_by_day[0] if s == sig]
        print(f"- support={n}: {members}\n  sig={sig}")
    for d in range(1, args.days + 1):
        cd = collections.Counter(sig for _, sig in sigs_by_day[d])
        print(f"\n# Day-{d} market-signature clusters")
        for sig, n in cd.most_common(6):
            members = [k for k, s in sigs_by_day[d] if s == sig]
            print(f"- support={n}: {members}\n  sig={sig}")

    print("\nALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
