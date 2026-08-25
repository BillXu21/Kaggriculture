(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ViewerCore = factory();
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function isObject(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
  function number(value) { if (typeof value !== "number" || !isFinite(value)) return "—"; return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.00$/, ""); }
  function yesNo(value) { return value === true ? "yes" : value === false ? "no" : "—"; }
  function json(value) { return JSON.stringify(value == null ? null : value); }
  function validCoordinate(value) { return Array.isArray(value) && value.length === 2 && Number.isInteger(value[0]) && Number.isInteger(value[1]) && value[0] >= 0 && value[0] < 10 && value[1] >= 0 && value[1] < 10; }
  function sameCoordinate(left, right) { return validCoordinate(left) && validCoordinate(right) && left[0] === right[0] && left[1] === right[1]; }
  function quadrant(row, column) { return (row < 5 ? "N" : "S") + (column < 5 ? "W" : "E"); }
  var tracePathPrefix = "/artifacts/debug_traces/";

  function validateTraceUrl(value, pageUrl) {
    var raw = typeof value === "string" ? value : "";
    if (!raw || raw.charAt(0) !== "/" || raw.indexOf("//") === 0) return { ok: false, error: "Trace URL must be an absolute local artifact path." };
    try {
      var page = new URL(pageUrl || "http://127.0.0.1/viewer/"), target = new URL(raw, page.href), path = decodeURIComponent(target.pathname);
      if (target.origin !== page.origin) return { ok: false, error: "Trace URL must use the viewer's same origin." };
      if (path.indexOf("\\") !== -1 || path.indexOf(tracePathPrefix) !== 0) return { ok: false, error: "Trace URL must point under /artifacts/debug_traces/." };
      var relative = path.slice(tracePathPrefix.length), segments = relative.split("/");
      if (!relative || relative.charAt(relative.length - 1) === "/" || segments.some(function (segment) { return segment === "." || segment === ".."; }) || !/\.json$/i.test(path)) return { ok: false, error: "Trace URL must name a JSON artifact under /artifacts/debug_traces/." };
      return { ok: true, url: target.href };
    } catch (error) {
      return { ok: false, error: "Trace URL is not a valid local artifact path." };
    }
  }

  function validateTrace(document) {
    if (!isObject(document)) throw new Error("Trace must be a JSON object.");
    if (document.schema_version !== 1) throw new Error("Unsupported trace schema_version " + JSON.stringify(document.schema_version) + "; this viewer supports schema v1.");
    if (!isObject(document.metadata)) throw new Error("Trace metadata must be an object.");
    if (Object.prototype.hasOwnProperty.call(document.metadata, "seat") && (!Number.isInteger(document.metadata.seat) || document.metadata.seat < 0)) throw new Error("Trace metadata.seat must be a non-negative integer.");
    if (!Array.isArray(document.turns)) throw new Error("Trace turns must be an array.");
    document.turns.forEach(function (turn, index) {
      if (!isObject(turn)) throw new Error("turn[" + index + "] must be an object.");
      ["step", "day", "hour", "canonical_state"].forEach(function (key) { if (!(key in turn)) throw new Error("turn[" + index + "] is missing " + key + "."); });
      if (!isObject(turn.canonical_state)) throw new Error("turn[" + index + "].canonical_state must be an object.");
      var state = turn.canonical_state;
      if (!Array.isArray(state.farms) || state.farms.length < 2) throw new Error("turn[" + index + "] canonical_state.farms must contain both seats.");
      if (!Array.isArray(state.privates) || state.privates.length < state.farms.length) throw new Error("turn[" + index + "] canonical_state.privates must align with farms.");
      state.farms.forEach(function (farm, seat) {
        if (!isObject(farm) || !Array.isArray(farm.tiles) || farm.tiles.length !== 10 || farm.tiles.some(function (row) { return !Array.isArray(row) || row.length !== 10; })) throw new Error("turn[" + index + "].canonical_state.farms[" + seat + "] must contain a 10x10 tiles grid.");
        farm.tiles.forEach(function (row) { row.forEach(function (tile) { if (isObject(tile) && Object.prototype.hasOwnProperty.call(tile, "age")) throw new Error("Trace uses raw tile age; schema v1 requires planted_day/placed_day."); }); });
      });
    });
    return document;
  }

  function lifecycle(tile, day) {
    if (!isObject(tile)) return "";
    if (tile.kind === "PLANT" && Number.isInteger(tile.planted_day)) return "age " + Math.max(0, day - tile.planted_day) + "d";
    if ((tile.kind === "COOP" || tile.kind === "PASTURE") && Number.isInteger(tile.placed_day)) return "age " + Math.max(0, day - tile.placed_day) + "d";
    return "age —";
  }

  function taskCoordinate(task) { return task && validCoordinate(task.tile) ? [task.tile[0], task.tile[1]] : null; }

  function classifyTask(task, context) {
    task = isObject(task) ? task : {};
    context = context || {};
    var kind = String(task.kind || "").toUpperCase();
    var source = String(task.source || "").toLowerCase();
    var priority = String(task.priority || "").toUpperCase();
    var blocked = Boolean(context.blocked || task.blocked);
    var category = "neutral", label = kind || "unknown task", rule = "Unknown task kind/source; shown with neutral styling.";
    if (kind === "FEED") { category = "feed-survival"; label = "survival FEED"; rule = "Survival work: feed an animal that is not fed today."; }
    else if (kind === "WATER" && source === "water_must_weed_boundary") { category = "water-must"; label = "mandatory WATER"; rule = "Mandatory weed-boundary watering from the canonical task source."; }
    else if (kind === "WATER" && (source === "water_yield_window" || priority === "PRODUCTIVE")) { category = "water-yield"; label = "yield WATER"; rule = "Yield-targeted watering from task source/priority, not an internal score."; }
    else if (kind === "HARVEST") { category = "harvest"; label = "HARVEST"; rule = "Productive harvest task recorded by the executor."; }
    else if (source.indexOf("manager") !== -1 || priority === "MANAGER") { category = "manager"; label = "manager reconciliation"; rule = "Manager-directed reconciliation/layout task from sidecar metadata."; }
    if (blocked) { category = "blocked"; label = "blocked / unassigned"; rule = context.reason ? "Unassigned: " + context.reason + "." : "Task is recorded as blocked or unassigned."; }
    return { category: category, label: label, rule: rule, kind: kind, source: source, priority: priority, blocked: blocked };
  }

  function cropState(tile, day, tasks, coordinate) {
    tasks = Array.isArray(tasks) ? tasks : [];
    if (tile === "WEED" || (isObject(tile) && String(tile.kind).toUpperCase() === "WEED")) return { status: "doomed-weed", label: "already doomed / weed", detail: "Canonical tile is WEED; no crop lifecycle is active." };
    if (!isObject(tile) || tile.kind !== "PLANT") return null;
    var matched = tasks.filter(function (task) { return sameCoordinate(taskCoordinate(task), coordinate); }).map(function (task) { return { task: task, semantic: classifyTask(task) }; });
    var mandatory = matched.find(function (item) { return item.semantic.category === "water-must"; });
    var yieldWater = matched.find(function (item) { return item.semantic.category === "water-yield"; });
    if (tile.watered_today === false && Number(tile.consecutive_unwatered || 0) >= 1) return { status: "must-water", label: "must water before refresh / weed boundary", detail: "Canonical consecutive_unwatered is at least 1 and watered_today is false; this is the executor's weed-boundary rule." };
    if (mandatory) return { status: "must-water", label: "must water before refresh / weed boundary", detail: mandatory.semantic.rule };
    if (yieldWater) return { status: "yield-water", label: "targeted for yield water", detail: yieldWater.semantic.rule };
    return { status: "safe-defer", label: "safe to defer", detail: "No mandatory or yield-water task is recorded for this crop in this turn; exact future deadlines are not inferred." };
  }

  function animalState(tile) {
    if (!isObject(tile) || !tile.animal) return null;
    if (tile.fed_today === true) return { status: "fed", label: "fed", detail: "Canonical fed_today=true." };
    if (Number(tile.consecutive_unfed || 0) >= 1) return { status: "escape-boundary", label: "at escape boundary", detail: "Inferred from canonical consecutive_unfed>=1 and fed_today=false; the sidecar may only provide aggregate starvation counts." };
    return { status: "unfed", label: "unfed, not at escape boundary", detail: "Canonical fed_today=false and consecutive_unfed<1." };
  }

  function formatTile(tile, day) {
    if (tile === "LOCKED") return { label: "Locked", sub: "unavailable", detail: "Locked tile" };
    if (tile === "WEED") return { label: "WEED", sub: "doomed", detail: "Already doomed / weed tile" };
    if (tile == null) return { label: "Empty", sub: "", detail: "Empty canonical tile" };
    if (!isObject(tile)) return { label: String(tile), sub: "", detail: "Canonical tile value" };
    if (tile.kind === "PLANT") return { label: tile.crop || "Plant", sub: [lifecycle(tile, day), "yield " + number(tile.yield_units)].join(" · "), detail: ["Crop: " + (tile.crop || "unknown"), lifecycle(tile, day), "yield units: " + number(tile.yield_units), "watered today: " + yesNo(tile.watered_today), "unwatered streak: " + number(tile.consecutive_unwatered), "fertilized until day: " + number(tile.fertilized_until_day)].join("; ") };
    if (tile.animal) return { label: tile.animal, sub: [lifecycle(tile, day), "yield " + number(tile.yield_units)].join(" · "), detail: ["Animal: " + tile.animal, lifecycle(tile, day), "yield units: " + number(tile.yield_units), "fed today: " + yesNo(tile.fed_today), "unfed streak: " + number(tile.consecutive_unfed), "cared today: " + yesNo(tile.cared_today), "fertilizer available: " + yesNo(tile.fertilizer_available)].join("; ") };
    return { label: tile.kind || "Tile", sub: "", detail: "Canonical tile: " + json(tile) };
  }

  function workerData(farm, privateState) {
    var positions = [farm && farm.farmer].concat((farm && farm.hands) || []), inventories = (privateState && privateState.inventories) || [];
    return positions.map(function (position, index) { var x = Array.isArray(position) ? position[0] : "—", y = Array.isArray(position) ? position[1] : "—", inventory = isObject(inventories[index]) ? inventories[index] : {}; return { index: index, name: index === 0 ? "Farmer" : "Hand " + index, position: [y, x], inventory: inventory }; });
  }

  function workerActionState(assignment) {
    assignment = isObject(assignment) ? assignment : {};
    var action = Array.isArray(assignment.action) ? assignment.action : [], op = String(action[0] || "PASS").toUpperCase(), movement = { NORTH: true, SOUTH: true, EAST: true, WEST: true };
    if (movement[op]) return { status: "traveling", label: assignment.task_key ? "traveling toward claimed task" : "moving", action: action, taskKey: assignment.task_key || null };
    if (op === "PICKUP") return { status: "pickup", label: "picking up", action: action, taskKey: assignment.task_key || null };
    if (op === "PASS") return { status: "idle", label: assignment.task_key ? "passing / blocked task" : "passing / idle", action: action, taskKey: assignment.task_key || null };
    return { status: "interacting", label: "interacting", action: action, taskKey: assignment.task_key || null };
  }

  function extractTrails(trace, endIndex, seat, windowTurns) {
    windowTurns = Math.max(1, Number(windowTurns) || 12);
    endIndex = Math.min(Math.max(0, Number(endIndex) || 0), trace.turns.length - 1);
    var start = Math.max(0, endIndex - windowTurns + 1), byWorker = {}, maxWorkers = 0;
    for (var index = start; index <= endIndex; index += 1) {
      var turn = trace.turns[index], farm = turn && turn.canonical_state && turn.canonical_state.farms[seat];
      if (!farm) continue;
      var workers = workerData(farm, {}); maxWorkers = Math.max(maxWorkers, workers.length);
      workers.forEach(function (worker) { if (!validCoordinate(worker.position)) return; (byWorker[worker.index] || (byWorker[worker.index] = [])).push({ row: worker.position[0], column: worker.position[1], step: turn.step, turnIndex: index }); });
    }
    return Object.keys(byWorker).map(function (key) { return { workerIndex: Number(key), points: byWorker[key] }; }).sort(function (a, b) { return a.workerIndex - b.workerIndex; });
  }

  function assignmentGeometry(trace, endIndex, seat) {
    var turn = trace.turns[endIndex], state = turn && turn.canonical_state, farm = state && state.farms[seat], executor = turn && isObject(turn.executor_debug) ? turn.executor_debug[String(seat)] : null;
    if (!farm || !isObject(executor)) return [];
    var workers = workerData(farm, state.privates[seat] || {}), tasks = Array.isArray(executor.tasks) ? executor.tasks : [], taskByKey = {};
    tasks.forEach(function (task) { if (task && task.key != null) taskByKey[String(task.key)] = task; });
    var blocked = isObject(executor.unassigned) && Array.isArray(executor.unassigned.task_keys) ? executor.unassigned.task_keys : [];
    var reasons = isObject(executor.unassigned) && isObject(executor.unassigned.reasons) ? executor.unassigned.reasons : {};
    return (Array.isArray(executor.assignments) ? executor.assignments : []).map(function (assignment) {
      var task = taskByKey[String(assignment.task_key)] || {}, target = validCoordinate(assignment.target) ? assignment.target : taskCoordinate(task), worker = workers[Number(assignment.worker_index)];
      var semantic = classifyTask(task, { blocked: blocked.indexOf(assignment.task_key) !== -1, reason: reasons[assignment.task_key] });
      return { assignment: assignment, workerIndex: Number(assignment.worker_index), from: worker && validCoordinate(worker.position) ? { row: worker.position[0], column: worker.position[1] } : null, to: target ? { row: target[0], column: target[1] } : null, semantic: semantic, actionState: workerActionState(assignment) };
    });
  }

  function taskMarkers(executor) {
    if (!isObject(executor) || !Array.isArray(executor.tasks)) return [];
    var unassigned = isObject(executor.unassigned) && Array.isArray(executor.unassigned.task_keys) ? executor.unassigned.task_keys : [], reasons = isObject(executor.unassigned) && isObject(executor.unassigned.reasons) ? executor.unassigned.reasons : {};
    return executor.tasks.map(function (task) { var key = task && task.key != null ? String(task.key) : "", semantic = classifyTask(task, { blocked: unassigned.indexOf(key) !== -1, reason: reasons[key] }); return { task: task, key: key, coordinate: taskCoordinate(task), semantic: semantic }; }).filter(function (marker) { return marker.coordinate !== null; });
  }

  function boardCells(state, seat, tasks) {
    var farm = state.farms[seat], privateState = state.privates[seat] || {}, unlocked = Array.isArray(farm.unlocked_quadrants) ? farm.unlocked_quadrants : [], workers = workerData(farm, privateState), byPosition = {}, markers = Array.isArray(tasks) ? tasks.map(function (marker) { return marker && marker.task ? marker : { task: marker, coordinate: taskCoordinate(marker), semantic: classifyTask(marker) }; }) : [];
    workers.forEach(function (worker) { byPosition[worker.position.join(",")] = (byPosition[worker.position.join(",")] || []).concat(worker); });
    var cells = [];
    for (var row = 0; row < 10; row += 1) for (var column = 0; column < 10; column += 1) {
      var tile = farm.tiles[row][column], info = formatTile(tile, state.day), crop = cropState(tile, state.day, markers.map(function (marker) { return marker.task; }), [row, column]), animal = animalState(tile), lifecycleState = crop || animal, cellMarkers = markers.filter(function (marker) { return sameCoordinate(marker.coordinate, [row, column]); });
      if (lifecycleState) info.detail += "; status: " + lifecycleState.label + ". " + lifecycleState.detail;
      cells.push({ row: row, column: column, quadrant: quadrant(row, column), unlocked: unlocked.indexOf(quadrant(row, column)) !== -1, tile: tile, label: info.label, sub: info.sub, detail: info.detail, workers: byPosition[row + "," + column] || [], lifecycle: lifecycleState, taskMarkers: cellMarkers });
    }
    return cells;
  }

  function planText(plan) {
    if (!isObject(plan)) return null;
    var parts = [];
    [["crops", plan.crop_targets], ["animals", plan.animal_targets], ["fertilizer", plan.fertilizer_by_crop], ["care", plan.care_by_animal]].forEach(function (pair) { if (isObject(pair[1])) parts.push(pair[0] + " " + Object.keys(pair[1]).map(function (key) { return key + ":" + pair[1][key]; }).join(", ")); });
    if (plan.land_count != null) parts.push("land: " + plan.land_count);
    if (plan.sell_quantities != null) parts.push("sell bins: " + Object.keys(plan.sell_quantities).length);
    return parts.length ? parts : [json(plan)];
  }

  function buildViewModel(trace, index, seat, options) {
    validateTrace(trace);
    var turn = trace.turns[index];
    if (!turn) return null;
    var state = turn.canonical_state, sidecars = isObject(turn.executor_debug) ? turn.executor_debug : {}, executor = isObject(sidecars[String(seat)]) ? sidecars[String(seat)] : null, markers = taskMarkers(executor), opts = options || {}, farm = state.farms[seat], privateState = state.privates[seat] || {};
    var workers = workerData(farm, privateState), assignments = assignmentGeometry(trace, index, seat), assignmentsByWorker = {}, jointActions = turn.joint_actions && turn.joint_actions[String(seat)] || {};
    assignments.forEach(function (item) { assignmentsByWorker[item.workerIndex] = item; });
    var workerStates = workers.map(function (worker) { var item = assignmentsByWorker[worker.index], actionSource = executor && executor.actions ? executor.actions : jointActions, fallback = actionSource ? (worker.index === 0 ? actionSource.farmer : (actionSource.hands || [])[worker.index - 1]) : ["PASS"], actionState = item ? item.actionState : workerActionState({ action: fallback }); return { worker: worker, assignment: item, actionState: actionState }; });
    return { trace: trace, turn: turn, state: state, seat: seat, farm: farm, privateState: privateState, workers: workers, workerStates: workerStates, cells: boardCells(state, seat, markers), executor: executor, plan: executor && executor.manager ? executor.manager : null, taskMarkers: markers, assignmentGeometry: assignments, trails: extractTrails(trace, index, seat, opts.trailWindow || 12) };
  }

  return { validateTrace: validateTrace, validateTraceUrl: validateTraceUrl, formatTile: formatTile, workerData: workerData, workerActionState: workerActionState, classifyTask: classifyTask, cropState: cropState, animalState: animalState, extractTrails: extractTrails, assignmentGeometry: assignmentGeometry, taskMarkers: taskMarkers, boardCells: boardCells, buildViewModel: buildViewModel, planText: planText };
}));

(function () {
  "use strict";
  if (typeof document === "undefined") return;
  var core = window.ViewerCore, trace = null, step = 0, seat = 0, timer = null, overlays = { trails: true, assignments: true, tasks: true, urgency: true, labels: true, trailWindow: 12 };
  var $ = function (id) { return document.getElementById(id); };
  var escapeHtml = function (value) { return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]; }); };
  var text = function (value) { return escapeHtml(value == null ? "—" : value); };
  var list = function (items, empty) { return items && items.length ? '<ul class="list">' + items.map(function (item) { return "<li>" + text(item) + "</li>"; }).join("") + "</ul>" : '<p class="empty">' + text(empty || "None recorded") + "</p>"; };
  var kv = function (key, value) { return '<div class="kv-row"><dt>' + text(key) + '</dt><dd>' + text(value) + '</dd></div>'; };
  var dictLines = function (value) { return isObject(value) && Object.keys(value).length ? Object.keys(value).sort().map(function (key) { return key + ": " + value[key]; }) : []; };
  var isObject = function (value) { return value !== null && typeof value === "object" && !Array.isArray(value); };
  function setStatus(message, kind) { $("load-status").textContent = message; $("load-status").className = "status " + (kind || ""); }
  function unavailable(label) { return '<p class="empty">' + text(label || "Unavailable in this turn") + "</p>"; }
  function stop() { if (timer !== null) { window.clearInterval(timer); timer = null; } $("play-toggle").textContent = "▶ Play"; }
  function overlayClass(marker) { return marker && marker.semantic ? marker.semantic.category : "neutral"; }
  function cellClass(cell) { return cell.lifecycle && cell.lifecycle.status ? "status-" + cell.lifecycle.status : ""; }
  function markerHtml(marker) { var label = marker.semantic.label; return '<span class="task-marker ' + overlayClass(marker) + '" title="' + text(label + ": " + marker.semantic.rule) + '"><span class="task-marker-dot"></span><span class="task-marker-label">' + (overlays.labels ? text(label) : "") + "</span></span>"; }
  function svgPoint(point) { return (point.column * 10 + 5) + "," + (point.row * 10 + 5); }

  function renderOverlaySvg(model) {
    var parts = [];
    if (overlays.trails) model.trails.forEach(function (trail) { if (trail.points.length > 1) parts.push('<polyline class="trail trail-' + trail.workerIndex + '" points="' + trail.points.map(svgPoint).join(" ") + '" />'); });
    if (overlays.assignments) model.assignmentGeometry.forEach(function (item) { if (item.from && item.to) parts.push('<line class="assignment-line ' + overlayClass(item) + '" x1="' + (item.from.column * 10 + 5) + '" y1="' + (item.from.row * 10 + 5) + '" x2="' + (item.to.column * 10 + 5) + '" y2="' + (item.to.row * 10 + 5) + '" />'); if (item.to) parts.push('<circle class="assignment-target ' + overlayClass(item) + '" cx="' + (item.to.column * 10 + 5) + '" cy="' + (item.to.row * 10 + 5) + '" r="2.2" />'); });
    $("overlay-svg").innerHTML = parts.join("");
  }

  function render() {
    var hasTrace = trace && trace.turns.length;
    $("viewer-content").hidden = !hasTrace; $("step-slider").disabled = !hasTrace; $("step-back").disabled = !hasTrace || step <= 0; $("step-forward").disabled = !hasTrace || step >= trace.turns.length - 1;
    if (!hasTrace) return;
    var model = core.buildViewModel(trace, step, seat, { trailWindow: overlays.trailWindow }), turn = model.turn, state = model.state, farm = model.farm, privateState = model.privateState;
    $("step-slider").max = Math.max(0, trace.turns.length - 1); $("step-slider").value = step; $("step-label").textContent = "Step " + step + " / " + (trace.turns.length - 1); $("turn-label").textContent = "day " + turn.day + " · hour " + turn.hour + " · trace step " + turn.step; $("view-label").textContent = "Seat " + seat + " · " + (trace.metadata.view || "canonical view") + " · " + (trace.metadata.backend || "backend unavailable"); $("board-title").textContent = "Farm " + seat;
    $("board").innerHTML = model.cells.map(function (cell) { var workerLabels = cell.workers.map(function (worker) { return '<span class="worker-marker ' + (worker.index ? "hand" : "") + '">' + (worker.index ? "H" + worker.index : "F") + '</span>'; }).join(""), taskLabels = overlays.tasks ? cell.taskMarkers.map(markerHtml).join("") : "", urgency = overlays.urgency ? cellClass(cell) : ""; return '<button class="cell ' + (cell.unlocked ? "unlocked" : "locked") + " " + urgency + '" type="button" role="gridcell" aria-label="[' + cell.row + ',' + cell.column + '] ' + text(cell.label) + '" title="[' + cell.row + ',' + cell.column + '] ' + text(cell.detail) + '"><span class="cell-coordinate">' + cell.row + "," + cell.column + " · " + cell.quadrant + '</span><span class="cell-label">' + text(cell.label) + '</span><span class="cell-sub">' + text(cell.sub) + '</span>' + taskLabels + workerLabels + '</button>'; }).join("");
    renderOverlaySvg(model);
    $("workers").innerHTML = model.workerStates.map(function (entry) { var worker = entry.worker, inventory = dictLines(worker.inventory), assignment = entry.assignment, context = assignment && assignment.assignment.task_key ? " · " + assignment.assignment.task_key : "", action = entry.actionState.action || []; return '<div class="worker-card state-' + text(entry.actionState.status) + '"><span class="worker-badge ' + (worker.index ? "hand" : "") + '">' + text(worker.index ? "H" + worker.index : "F") + '</span><div><strong>' + text(worker.name) + '</strong><small>position [' + text(worker.position[0]) + ", " + text(worker.position[1]) + ']</small><div class="worker-state">' + text(entry.actionState.label + context) + ' · ' + text(action.join(" ")) + '</div><div class="inventory-list">' + text(inventory.length ? inventory.join(" · ") : "empty inventory") + '</div></div></div>'; }).join("");
    $("worker-count").textContent = model.workers.length + " worker" + (model.workers.length === 1 ? "" : "s");
    $("time-panel").innerHTML = '<h2>Time & status</h2><dl>' + kv("Day / hour", turn.day + " / " + turn.hour) + kv("Canonical step", state.step) + kv("Seat status", (state.statuses || [])[seat]) + kv("Reward", (state.rewards || [])[seat]) + kv("Current seat", turn.current_seat == null ? "—" : turn.current_seat) + '</dl>';
    $("economy-panel").innerHTML = '<h2>Economy</h2><dl>' + kv("Cash", number(farm.money)) + kv("Hires today", farm.hires_today) + kv("Unlocked", (farm.unlocked_quadrants || []).join(", ") || "none") + kv("Town shops", ((state.town || {}).unlocked_shops || []).join(", ") || "none") + '</dl>';
    $("storage-panel").innerHTML = '<h2>Shed & seeds</h2><dl>' + kv("Shed", dictLines(privateState.shed).join(" · ") || "empty") + kv("Seeds", dictLines(privateState.seeds).join(" · ") || "none") + kv("Carried", model.workers.map(function (worker) { return worker.name + ": " + (dictLines(worker.inventory).join(", ") || "empty"); }).join("; ")) + '</dl>';
    var market = state.market || {}; $("market-panel").innerHTML = '<h2>Market</h2><dl>' + kv("Prices", dictLines(market.prices).join(" · ") || "unavailable") + kv("Inventory", dictLines(market.inventory).join(" · ") || "unavailable") + '</dl>';
    var manager = model.executor && model.executor.manager; $("manager-panel").innerHTML = '<h2>Manager plan</h2>' + (manager ? '<div class="kv-row"><dt>Requested</dt><dd>' + text((core.planText(manager.requested) || []).join(" · ")) + '</dd></div><div class="kv-row"><dt>Feasible</dt><dd>' + text((core.planText(manager.feasible) || []).join(" · ")) + '</dd></div>' + list(dictLines(manager.projection_changes), "No projection changes") : unavailable("No manager/executor sidecar recorded for this turn."));
    var executor = model.executor, taskLines = executor && Array.isArray(executor.tasks) ? executor.tasks.map(function (task) { var semantic = core.classifyTask(task); return semantic.label + ": " + (task.key || "") + (task.tile ? " @ [" + task.tile.join(",") + "]" : "") + (task.source ? " · " + task.source : ""); }) : [], assignmentLines = executor && Array.isArray(executor.assignments) ? executor.assignments.map(function (assignment) { var actionState = core.workerActionState(assignment); return "worker " + assignment.worker_index + ": " + actionState.label + " / " + (assignment.action || []).join(" ") + " → " + (assignment.task_key || "PASS") + (assignment.target ? " @ [" + assignment.target.join(",") + "]" : "") + " (" + (assignment.reason || "") + ")"; }) : [];
    $("executor-panel").innerHTML = '<h2>Executor tasks & assignments</h2>' + (executor ? '<h3>Tasks <span class="pill">' + taskLines.length + '</span></h3>' + list(taskLines, "No tasks") + '<h3>Assignments</h3>' + list(assignmentLines, "No assignments") : unavailable("Executor sidecar absent for this opening turn."));
    var survival = executor && executor.survival; $("survival-panel").innerHTML = '<h2>Survival / feed</h2>' + (survival ? '<dl>' + kv("Unfed", survival.unfed_count) + kv("Starvation boundary", survival.starvation_boundary_count) + kv("Shed wheat", survival.shed_wheat) + kv("Carried wheat", survival.carried_wheat) + kv("Shortage", survival.shortage) + kv("Expansion suppressed", survival.expansion_suppressed == null ? "—" : survival.expansion_suppressed ? "yes" : "no") + '</dl>' : unavailable("No survival diagnostics recorded."));
    var actions = turn.joint_actions && turn.joint_actions[String(seat)], actionLines = actions ? ["canonical farmer: " + json(actions.farmer), "canonical hands: " + json(actions.hands), "canonical market: " + json(actions.market)] : []; if (executor && executor.actions) { actionLines.push("executor farmer: " + json(executor.actions.farmer)); actionLines.push("executor hands: " + json(executor.actions.hands)); } if (executor && executor.market) actionLines.push("submitted market: " + json(executor.market.submitted)); $("actions-panel").innerHTML = '<h2>Actions & movement</h2>' + list(actionLines, "No action data recorded for this turn.");
  }

  function makeSeatToggle() { var count = trace && trace.turns.length ? trace.turns[0].canonical_state.farms.length : 0; $("seat-toggle").innerHTML = ""; for (var index = 0; index < count; index += 1) { var button = document.createElement("button"); button.type = "button"; button.className = "toggle-button" + (index === seat ? " active" : ""); button.textContent = "Seat " + index; button.dataset.seat = index; button.onclick = function () { seat = Number(this.dataset.seat); makeSeatToggle(); render(); }; $("seat-toggle").appendChild(button); } }
  function setTrace(document, source) { try { core.validateTrace(document); trace = document; step = 0; var farmCount = document.turns.length ? document.turns[0].canonical_state.farms.length : 0; seat = Number.isInteger(document.metadata.seat) && document.metadata.seat >= 0 && document.metadata.seat < farmCount ? document.metadata.seat : 0; makeSeatToggle(); render(); setStatus("Loaded " + source + " · " + document.turns.length + " turn" + (document.turns.length === 1 ? "" : "s"), "ok"); } catch (error) { stop(); trace = null; render(); setStatus(error.message || String(error), "error"); } }
  function loadFile(file) { if (!file) return; file.text().then(function (contents) { setTrace(JSON.parse(contents), file.name); }).catch(function (error) { setTrace(null, file.name); setStatus("Could not load " + file.name + ": " + (error.message || error), "error"); }); }
  function loadUrl(url) { var checked = core.validateTraceUrl(url, window.location.href); if (!checked.ok) { setStatus("Could not load trace URL: " + checked.error, "error"); return; } setStatus("Loading " + checked.url + "…"); fetch(checked.url).then(function (response) { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); }).then(function (document) { setTrace(document, checked.url); }).catch(function (error) { setStatus("Could not load trace URL: " + (error.message || error), "error"); }); }
  function bindOverlay(id, key) { $(id).addEventListener("change", function () { overlays[key] = this.checked; render(); }); }

  document.addEventListener("DOMContentLoaded", function () {
    $("trace-file").addEventListener("change", function () { loadFile(this.files[0]); }); $("drop-zone").addEventListener("dragover", function (event) { event.preventDefault(); this.classList.add("dragging"); }); $("drop-zone").addEventListener("dragleave", function () { this.classList.remove("dragging"); }); $("drop-zone").addEventListener("drop", function (event) { event.preventDefault(); this.classList.remove("dragging"); loadFile(event.dataTransfer.files[0]); });
    $("step-slider").addEventListener("input", function () { stop(); step = Number(this.value); render(); }); $("step-back").onclick = function () { stop(); if (step > 0) step -= 1; render(); }; $("step-forward").onclick = function () { if (trace && step < trace.turns.length - 1) step += 1; if (!trace || step >= trace.turns.length - 1) stop(); render(); }; $("play-toggle").onclick = function () { if (!trace || !trace.turns.length) return; if (timer !== null) stop(); else { if (step >= trace.turns.length - 1) step = 0; timer = window.setInterval(function () { if (step >= trace.turns.length - 1) { stop(); render(); } else { step += 1; render(); } }, Number($("speed-select").value)); $("play-toggle").textContent = "❚❚ Pause"; } }; $("speed-select").onchange = function () { if (timer !== null) { stop(); $("play-toggle").click(); } }; $("clear-trace").onclick = function () { stop(); trace = null; $("seat-toggle").innerHTML = ""; render(); setStatus("Choose a trace JSON file or drop one here."); };
    bindOverlay("trail-toggle", "trails"); bindOverlay("assignment-toggle", "assignments"); bindOverlay("task-toggle", "tasks"); bindOverlay("urgency-toggle", "urgency"); bindOverlay("labels-toggle", "labels"); $("trail-window").addEventListener("input", function () { overlays.trailWindow = Number(this.value); $("trail-window-label").textContent = this.value + " turns"; render(); });
    document.addEventListener("keydown", function (event) { if (event.target && ["INPUT", "SELECT", "TEXTAREA"].indexOf(event.target.tagName) !== -1) return; if (event.key === "ArrowLeft") { $("step-back").click(); event.preventDefault(); } if (event.key === "ArrowRight") { $("step-forward").click(); event.preventDefault(); } if (event.key === " ") { $("play-toggle").click(); event.preventDefault(); } }); var query = new URLSearchParams(window.location.search), traceUrl = query.get("trace") || query.get("url"); if (traceUrl) loadUrl(traceUrl);
  });
}());
