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
    yield_units: 2, watered_today: true, consecutive_unwatered: 0,
    fertilized_until_day: 3,
  };
  farm.tiles[7][8] = {
    kind: "PASTURE", animal: "SHEEP", placed_day: 0, yield_units: 1,
    consecutive_unfed: 1, fed_today: false, cared_today: true,
    fertilizer_available: true, pending_care_bonus: 0,
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
      tasks: [{key: "WATER:1,2", kind: "WATER", priority: "MAINTENANCE", tile: [1, 2], source: "test"}],
      assignments: [{worker_index: 0, task_key: "WATER:1,2", action: ["EAST"], reason: "greedy_dispatch", target: [1, 2]}],
      actions: {farmer: ["EAST"], hands: [["WATER"]]},
      survival: {unfed_count: 1, starvation_boundary_count: 1, shed_wheat: 3, carried_wheat: 1, shortage: 0, expansion_suppressed: false},
    }}}],
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
  if (!model.cells.some((cell) => cell.label === "WHEAT" && cell.detail.includes("age 1d"))) throw new Error("crop lifecycle detail missing");
  if (!model.cells.some((cell) => cell.label === "SHEEP" && cell.detail.includes("age 2d"))) throw new Error("animal lifecycle detail missing");
  if (model.workers.length !== 2 || model.workers[1].inventory.FERTILIZER !== 2) throw new Error("worker private inventory missing");
  if (!model.executor || !model.executor.manager || model.executor.assignments.length !== 1) throw new Error("executor sidecar missing");
}
console.log(JSON.stringify({turns: trace.turns.length, cells: model.cells.length, workers: model.workers.length, crop: model.cells[12].label, animal: model.cells[78].label, sidecar: Boolean(model.executor)}));
