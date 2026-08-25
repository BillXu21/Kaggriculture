const fs = require("fs");
const path = require("path");
const core = require(path.join(__dirname, "..", "viewer", "viewer.js"));

function sampleTrace() {
  const emptyRow = () => Array(10).fill(null);
  const farm = {
    money: 1234.5,
    tiles: Array.from({length: 10}, () => emptyRow()),
    farmer: [2, 3],
    hands: [[4, 5]],
    unlocked_quadrants: ["NW", "SE"],
    hires_today: 1,
  };
  farm.tiles[1][2] = {
    kind: "PLANT", crop: "WHEAT", planted_day: 1, max_lifespan_step: 100,
    yield_units: 2, watered_today: false, consecutive_unwatered: 1,
    fertilized_until_day: 3,
  };
  farm.tiles[1][3] = {
    kind: "PLANT", crop: "MELON", planted_day: 2, max_lifespan_step: 100,
    yield_units: 0, watered_today: false, consecutive_unwatered: 0,
    fertilized_until_day: 0,
  };
  farm.tiles[2][2] = "WEED";
  farm.tiles[7][8] = {
    kind: "PASTURE", animal: "SHEEP", placed_day: 0, yield_units: 1,
    consecutive_unfed: 1, fed_today: false, cared_today: true,
    fertilizer_available: true, pending_care_bonus: 0,
  };
  farm.tiles[7][7] = {
    kind: "COOP", animal: "COW", placed_day: 1, yield_units: 0,
    consecutive_unfed: 0, fed_today: false, cared_today: false,
    fertilizer_available: false, pending_care_bonus: 0,
  };
  const otherFarm = JSON.parse(JSON.stringify(farm));
  const state = {
    step: 4, day: 2, hour: 4, farms: [farm, otherFarm],
    privates: [
      {shed: {WHEAT: 3}, seeds: {WHEAT: 4}, inventories: [{WHEAT: 1}, {FERTILIZER: 2}]},
      {shed: {}, seeds: {}, inventories: [{WHEAT: 0}, {}]},
    ],
    market: {inventory: {WHEAT: 10}, prices: {WHEAT: 5}},
    town: {unlocked_shops: []}, rewards: [1, 0], statuses: ["ACTIVE", "ACTIVE"],
  };
  return {
    schema_version: 1,
    metadata: {seed: 17, seat: 0, view: "joint"},
    turns: [{step: 4, day: 2, hour: 4, canonical_state: state, joint_actions: {"0": {farmer: ["EAST"], hands: [["WATER"]], market: []}}, executor_debug: {"0": {
      manager: {requested: {crop_targets: {WHEAT: 2}, land_count: 2}, feasible: {crop_targets: {WHEAT: 1}, land_count: 1}, projection_changes: {}},
      tasks: [
        {key: "FEED:7,8", kind: "FEED", priority: "MAINTENANCE", tile: [7, 8], source: "mechanical"},
        {key: "WATER:1,2", kind: "WATER", priority: "MAINTENANCE", tile: [1, 2], source: "water_must_weed_boundary"},
        {key: "WATER:1,3", kind: "WATER", priority: "PRODUCTIVE", tile: [1, 3], source: "water_yield_window"},
        {key: "HARVEST:1,4", kind: "HARVEST", priority: "PRODUCTIVE", tile: [1, 4], source: "mechanical"},
        {key: "DIG:2,2", kind: "DIG", priority: "MANAGER", tile: [2, 2], source: "manager_reconciliation"},
        {key: "FUTURE:3,3", kind: "FUTURE_TASK", priority: "FUTURE", tile: [3, 3], source: "future"},
      ],
      assignments: [
        {worker_index: 0, task_key: "WATER:1,2", action: ["EAST"], reason: "greedy_dispatch", target: [1, 2]},
        {worker_index: 1, task_key: "FEED:7,8", action: ["PICKUP", "WHEAT", 1], reason: "greedy_dispatch_via_shed", target: [7, 8]},
      ],
      unassigned: {task_keys: ["FUTURE:3,3"], reasons: {"FUTURE:3,3": "unknown_task_kind"}},
      actions: {farmer: ["EAST"], hands: [["WATER"]]},
      survival: {unfed_count: 1, starvation_boundary_count: 1, shed_wheat: 3, carried_wheat: 1, shortage: 0, expansion_suppressed: false},
    }}},
      {step: 5, day: 2, hour: 5, canonical_state: JSON.parse(JSON.stringify(state)), joint_actions: {}, executor_debug: {}},
      {step: 6, day: 2, hour: 6, canonical_state: JSON.parse(JSON.stringify(state)), joint_actions: {}, executor_debug: {}}],
  };
}

const source = process.argv[2];
const trace = source ? JSON.parse(fs.readFileSync(source, "utf8")) : sampleTrace();
const before = JSON.stringify(trace);
core.validateTrace(trace);
const model = core.buildViewModel(trace, 0, 0);
if (JSON.stringify(trace) !== before) throw new Error("viewer helpers mutated trace input");
if (model.cells.length !== 100) throw new Error("expected 100 board cells");
if (!source) {
  const pageUrl = "http://127.0.0.1:8765/viewer/";
  for (const [url, allowed] of [
    ["/artifacts/debug_traces/seed_17_seat_0.json", true],
    ["https://example.invalid/trace.json", false],
    ["//example.invalid/trace.json", false],
    ["/viewer/viewer.js", false],
    ["/artifacts/debug_traces/../README.json", false],
    ["artifacts/debug_traces/seed_17_seat_0.json", false],
  ]) {
    if (core.validateTraceUrl(url, pageUrl).ok !== allowed) throw new Error(`unexpected trace URL policy for ${url}`);
  }
  if (!model.cells.some((cell) => cell.label === "WHEAT" && cell.detail.includes("age 1d"))) throw new Error("crop lifecycle detail missing");
  if (core.classifyTask({kind: "FEED"}).category !== "feed-survival") throw new Error("FEED classification missing");
  if (core.classifyTask({kind: "WATER", source: "water_must_weed_boundary"}).category !== "water-must") throw new Error("mandatory WATER classification missing");
  if (core.classifyTask({kind: "WATER", priority: "PRODUCTIVE", source: "water_yield_window"}).category !== "water-yield") throw new Error("yield WATER classification missing");
  if (core.classifyTask({kind: "HARVEST"}).category !== "harvest") throw new Error("HARVEST classification missing");
  if (core.classifyTask({kind: "DIG", priority: "MANAGER"}).category !== "manager") throw new Error("manager classification missing");
  if (core.classifyTask({kind: "FUTURE"}).category !== "neutral") throw new Error("unknown task neutral classification missing");
  if (core.classifyTask({kind: "FUTURE"}, {blocked: true, reason: "blocked"}).category !== "blocked") throw new Error("blocked classification missing");
  if (core.cropState(model.cells[12].tile, 2, [], [1, 2]).status !== "must-water") throw new Error("crop must-water state missing");
  if (core.cropState(model.cells[13].tile, 2, [{kind: "WATER", source: "water_yield_window", priority: "PRODUCTIVE", tile: [1, 3]}], [1, 3]).status !== "yield-water") throw new Error("crop yield-water state missing");
  if (core.cropState("WEED", 2, [], [2, 2]).status !== "doomed-weed") throw new Error("weed state missing");
  if (core.animalState(model.cells[78].tile).status !== "escape-boundary") throw new Error("animal boundary state missing");
  if (core.animalState(model.cells[77].tile).status !== "unfed") throw new Error("animal unfed state missing");
  if (core.workerActionState({action: ["NORTH"], task_key: "WATER:1,2"}).status !== "traveling") throw new Error("worker movement state missing");
  if (core.workerActionState({action: ["PICKUP", "WHEAT"]}).status !== "pickup") throw new Error("worker pickup state missing");
  if (core.workerActionState({action: ["PASS"]}).status !== "idle") throw new Error("worker idle state missing");
  if (core.workerActionState({action: ["WATER"]}).status !== "interacting") throw new Error("worker interaction state missing");
  if (model.taskMarkers.length !== 6 || model.assignmentGeometry.length !== 2) throw new Error("task marker/assignment geometry missing");
  if (!model.assignmentGeometry.every((item) => item.from && item.to)) throw new Error("assignment coordinates missing");
  if (core.extractTrails(trace, 2, 0, 12)[0].points.length !== 3) throw new Error("worker trail extraction missing");
  if (!model.cells[12].detail.includes("must water before refresh")) throw new Error("crop detail rule missing");
  if (model.workers.length !== 2 || model.workers[1].inventory.FERTILIZER !== 2) throw new Error("worker private inventory missing");
  if (!model.executor || !model.executor.manager || model.executor.assignments.length !== 2) throw new Error("executor sidecar missing");
}
console.log(JSON.stringify({turns: trace.turns.length, cells: model.cells.length, workers: model.workers.length, crop: model.cells[12].label, animal: model.cells[78].label, sidecar: Boolean(model.executor), trails: core.extractTrails(trace, Math.min(2, trace.turns.length - 1), 0, 12).length}));
