(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ViewerCore = factory();
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function validateTrace(document) {
    if (!isObject(document)) throw new Error("Trace must be a JSON object.");
    if (document.schema_version !== 1) {
      throw new Error("Unsupported trace schema_version " + JSON.stringify(document.schema_version) + "; this viewer supports schema v1.");
    }
    if (!isObject(document.metadata)) throw new Error("Trace metadata must be an object.");
    if (Object.prototype.hasOwnProperty.call(document.metadata, "seat") && (!Number.isInteger(document.metadata.seat) || document.metadata.seat < 0)) throw new Error("Trace metadata.seat must be a non-negative integer.");
    if (!Array.isArray(document.turns)) throw new Error("Trace turns must be an array.");
    document.turns.forEach(function (turn, index) {
      if (!isObject(turn)) throw new Error("turn[" + index + "] must be an object.");
      ["step", "day", "hour", "canonical_state"].forEach(function (key) {
        if (!(key in turn)) throw new Error("turn[" + index + "] is missing " + key + ".");
      });
      if (!isObject(turn.canonical_state)) throw new Error("turn[" + index + "].canonical_state must be an object.");
      var state = turn.canonical_state;
      if (!Array.isArray(state.farms) || state.farms.length < 2) throw new Error("turn[" + index + "] canonical_state.farms must contain both seats.");
      if (!Array.isArray(state.privates) || state.privates.length < state.farms.length) throw new Error("turn[" + index + "] canonical_state.privates must align with farms.");
      state.farms.forEach(function (farm, seat) {
        if (!isObject(farm) || !Array.isArray(farm.tiles) || farm.tiles.length !== 10 || farm.tiles.some(function (row) { return !Array.isArray(row) || row.length !== 10; })) {
          throw new Error("turn[" + index + "].canonical_state.farms[" + seat + "] must contain a 10x10 tiles grid.");
        }
        farm.tiles.forEach(function (row) { row.forEach(function (tile) {
          if (isObject(tile) && Object.prototype.hasOwnProperty.call(tile, "age")) throw new Error("Trace uses raw tile age; schema v1 requires planted_day/placed_day.");
        }); });
      });
    });
    return document;
  }

  function number(value) {
    if (typeof value !== "number" || !isFinite(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.00$/, "");
  }

  function yesNo(value) { return value === true ? "yes" : value === false ? "no" : "—"; }
  function json(value) { return JSON.stringify(value == null ? null : value); }

  function quadrant(row, column) {
    return (row < 5 ? "N" : "S") + (column < 5 ? "W" : "E");
  }

  function lifecycle(tile, day) {
    if (!isObject(tile)) return "";
    if (tile.kind === "PLANT" && Number.isInteger(tile.planted_day)) return "age " + Math.max(0, day - tile.planted_day) + "d";
    if ((tile.kind === "COOP" || tile.kind === "PASTURE") && Number.isInteger(tile.placed_day)) return "age " + Math.max(0, day - tile.placed_day) + "d";
    return "age —";
  }

  function formatTile(tile, day) {
    if (tile === "LOCKED") return { label: "Locked", sub: "unavailable", detail: "Locked tile" };
    if (tile == null) return { label: "Empty", sub: "", detail: "Empty canonical tile" };
    if (!isObject(tile)) return { label: String(tile), sub: "", detail: "Canonical tile value" };
    if (tile.kind === "PLANT") {
      return {
        label: tile.crop || "Plant",
        sub: [lifecycle(tile, day), "yield " + number(tile.yield_units)].join(" · "),
        detail: ["Crop: " + (tile.crop || "unknown"), lifecycle(tile, day), "yield units: " + number(tile.yield_units), "watered today: " + yesNo(tile.watered_today), "unwatered streak: " + number(tile.consecutive_unwatered), "fertilized until day: " + number(tile.fertilized_until_day)].join("; ")
      };
    }
    if (tile.animal) {
      return {
        label: tile.animal,
        sub: [lifecycle(tile, day), "yield " + number(tile.yield_units)].join(" · "),
        detail: ["Animal: " + tile.animal, lifecycle(tile, day), "yield units: " + number(tile.yield_units), "fed today: " + yesNo(tile.fed_today), "unfed streak: " + number(tile.consecutive_unfed), "cared today: " + yesNo(tile.cared_today), "fertilizer available: " + yesNo(tile.fertilizer_available)].join("; ")
      };
    }
    return { label: tile.kind || "Tile", sub: "", detail: "Canonical tile: " + json(tile) };
  }

  function workerData(farm, privateState) {
    var positions = [farm && farm.farmer].concat((farm && farm.hands) || []);
    var inventories = (privateState && privateState.inventories) || [];
    return positions.map(function (position, index) {
      var x = Array.isArray(position) ? position[0] : "—";
      var y = Array.isArray(position) ? position[1] : "—";
      var inventory = isObject(inventories[index]) ? inventories[index] : {};
      return { index: index, name: index === 0 ? "Farmer" : "Hand " + index, position: [y, x], inventory: inventory };
    });
  }

  function boardCells(state, seat) {
    var farm = state.farms[seat];
    var privateState = state.privates[seat] || {};
    var unlocked = Array.isArray(farm.unlocked_quadrants) ? farm.unlocked_quadrants : [];
    var workers = workerData(farm, privateState);
    var byPosition = {};
    workers.forEach(function (worker) { byPosition[worker.position.join(",")] = (byPosition[worker.position.join(",")] || []).concat(worker); });
    var cells = [];
    for (var row = 0; row < 10; row += 1) for (var column = 0; column < 10; column += 1) {
      var tile = farm.tiles[row][column];
      var info = formatTile(tile, state.day);
      var area = quadrant(row, column);
      cells.push({ row: row, column: column, quadrant: area, unlocked: unlocked.indexOf(area) !== -1, tile: tile, label: info.label, sub: info.sub, detail: info.detail, workers: byPosition[row + "," + column] || [] });
    }
    return cells;
  }

  function planText(plan) {
    if (!isObject(plan)) return null;
    var parts = [];
    [["crops", plan.crop_targets], ["animals", plan.animal_targets], ["fertilizer", plan.fertilizer_by_crop], ["care", plan.care_by_animal]].forEach(function (pair) {
      if (isObject(pair[1])) parts.push(pair[0] + " " + Object.keys(pair[1]).map(function (key) { return key + ":" + pair[1][key]; }).join(", "));
    });
    if (plan.land_count != null) parts.push("land: " + plan.land_count);
    if (plan.sell_quantities != null) parts.push("sell bins: " + Object.keys(plan.sell_quantities).length);
    return parts.length ? parts : [json(plan)];
  }

  function buildViewModel(trace, index, seat) {
    validateTrace(trace);
    var turn = trace.turns[index];
    if (!turn) return null;
    var state = turn.canonical_state;
    var sidecars = isObject(turn.executor_debug) ? turn.executor_debug : {};
    var executor = isObject(sidecars[String(seat)]) ? sidecars[String(seat)] : null;
    return { trace: trace, turn: turn, state: state, seat: seat, farm: state.farms[seat], privateState: state.privates[seat] || {}, workers: workerData(state.farms[seat], state.privates[seat] || {}), cells: boardCells(state, seat), executor: executor, plan: executor && executor.manager ? executor.manager : null };
  }

  return { validateTrace: validateTrace, formatTile: formatTile, workerData: workerData, boardCells: boardCells, buildViewModel: buildViewModel, planText: planText };
}));

(function () {
  "use strict";
  if (typeof document === "undefined") return;
  var core = window.ViewerCore;
  var trace = null;
  var step = 0;
  var seat = 0;
  var timer = null;
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

  function render() {
    var hasTrace = trace && trace.turns.length;
    $("viewer-content").hidden = !hasTrace;
    $("step-slider").disabled = !hasTrace;
    $("step-back").disabled = !hasTrace || step <= 0;
    $("step-forward").disabled = !hasTrace || step >= trace.turns.length - 1;
    if (!hasTrace) return;
    var model = core.buildViewModel(trace, step, seat);
    var turn = model.turn, state = model.state, farm = model.farm, privateState = model.privateState;
    $("step-slider").max = Math.max(0, trace.turns.length - 1); $("step-slider").value = step;
    $("step-label").textContent = "Step " + step + " / " + (trace.turns.length - 1);
    $("turn-label").textContent = "day " + turn.day + " · hour " + turn.hour + " · trace step " + turn.step;
    $("view-label").textContent = "Seat " + seat + " · " + (trace.metadata.view || "canonical view") + " · " + (trace.metadata.backend || "backend unavailable");
    $("board-title").textContent = "Farm " + seat;
    $("board").innerHTML = model.cells.map(function (cell) {
      var workerLabels = cell.workers.map(function (worker) { return '<span class="worker-marker ' + (worker.index ? "hand" : "") + '">' + (worker.index ? "H" + worker.index : "F") + '</span>'; }).join("");
      return '<button class="cell ' + (cell.unlocked ? "unlocked" : "locked") + '" type="button" role="gridcell" aria-label="[' + cell.row + ',' + cell.column + '] ' + text(cell.label) + '" title="[' + cell.row + ',' + cell.column + '] ' + text(cell.detail) + '"><span class="cell-coordinate">' + cell.row + "," + cell.column + " · " + cell.quadrant + '</span><span class="cell-label">' + text(cell.label) + '</span><span class="cell-sub">' + text(cell.sub) + '</span>' + workerLabels + '</button>';
    }).join("");
    $("workers").innerHTML = model.workers.map(function (worker) {
      var inventory = dictLines(worker.inventory);
      return '<div class="worker-card"><span class="worker-badge ' + (worker.index ? "hand" : "") + '">' + text(worker.index ? "H" + worker.index : "F") + '</span><div><strong>' + text(worker.name) + '</strong><small>position [' + text(worker.position[0]) + ", " + text(worker.position[1]) + ']</small><div class="inventory-list">' + text(inventory.length ? inventory.join(" · ") : "empty inventory") + '</div></div></div>';
    }).join("");
    $("worker-count").textContent = model.workers.length + " worker" + (model.workers.length === 1 ? "" : "s");
    $("time-panel").innerHTML = '<h2>Time & status</h2><dl>' + kv("Day / hour", turn.day + " / " + turn.hour) + kv("Canonical step", state.step) + kv("Seat status", (state.statuses || [])[seat]) + kv("Reward", (state.rewards || [])[seat]) + kv("Current seat", turn.current_seat == null ? "—" : turn.current_seat) + '</dl>';
    $("economy-panel").innerHTML = '<h2>Economy</h2><dl>' + kv("Cash", number(farm.money)) + kv("Hires today", farm.hires_today) + kv("Unlocked", (farm.unlocked_quadrants || []).join(", ") || "none") + kv("Town shops", ((state.town || {}).unlocked_shops || []).join(", ") || "none") + '</dl>';
    $("storage-panel").innerHTML = '<h2>Shed & seeds</h2><dl>' + kv("Shed", dictLines(privateState.shed).join(" · ") || "empty") + kv("Seeds", dictLines(privateState.seeds).join(" · ") || "none") + kv("Carried", model.workers.map(function (worker) { return worker.name + ": " + (dictLines(worker.inventory).join(", ") || "empty"); }).join("; ")) + '</dl>';
    var market = state.market || {};
    $("market-panel").innerHTML = '<h2>Market</h2><dl>' + kv("Prices", dictLines(market.prices).join(" · ") || "unavailable") + kv("Inventory", dictLines(market.inventory).join(" · ") || "unavailable") + '</dl>';
    var manager = model.executor && model.executor.manager;
    $("manager-panel").innerHTML = '<h2>Manager plan</h2>' + (manager ? '<div class="kv-row"><dt>Requested</dt><dd>' + text((core.planText(manager.requested) || []).join(" · ")) + '</dd></div><div class="kv-row"><dt>Feasible</dt><dd>' + text((core.planText(manager.feasible) || []).join(" · ")) + '</dd></div>' + list(dictLines(manager.projection_changes), "No projection changes") : unavailable("No manager/executor sidecar recorded for this turn."));
    var executor = model.executor;
    var taskLines = executor && Array.isArray(executor.tasks) ? executor.tasks.map(function (task) { return (task.priority || "") + " " + (task.kind || "task") + " " + (task.key || "") + (task.tile ? " @ [" + task.tile.join(",") + "]" : "") + (task.source ? " · " + task.source : ""); }) : [];
    var assignmentLines = executor && Array.isArray(executor.assignments) ? executor.assignments.map(function (assignment) { return "worker " + assignment.worker_index + ": " + (assignment.action || []).join(" ") + " → " + (assignment.task_key || "PASS") + (assignment.target ? " @ [" + assignment.target.join(",") + "]" : "") + " (" + (assignment.reason || "") + ")"; }) : [];
    $("executor-panel").innerHTML = '<h2>Executor tasks & assignments</h2>' + (executor ? '<h3>Tasks <span class="pill">' + taskLines.length + '</span></h3>' + list(taskLines, "No tasks") + '<h3>Assignments</h3>' + list(assignmentLines, "No assignments") : unavailable("Executor sidecar absent for this opening turn."));
    var survival = executor && executor.survival;
    $("survival-panel").innerHTML = '<h2>Survival / feed</h2>' + (survival ? '<dl>' + kv("Unfed", survival.unfed_count) + kv("Starvation boundary", survival.starvation_boundary_count) + kv("Shed wheat", survival.shed_wheat) + kv("Carried wheat", survival.carried_wheat) + kv("Shortage", survival.shortage) + kv("Expansion suppressed", survival.expansion_suppressed == null ? "—" : survival.expansion_suppressed ? "yes" : "no") + '</dl>' : unavailable("No survival diagnostics recorded."));
    var actions = turn.joint_actions && turn.joint_actions[String(seat)];
    var actionLines = actions ? ["canonical farmer: " + json(actions.farmer), "canonical hands: " + json(actions.hands), "canonical market: " + json(actions.market)] : [];
    if (executor && executor.actions) { actionLines.push("executor farmer: " + json(executor.actions.farmer)); actionLines.push("executor hands: " + json(executor.actions.hands)); }
    if (executor && executor.market) actionLines.push("submitted market: " + json(executor.market.submitted));
    $("actions-panel").innerHTML = '<h2>Actions & movement</h2>' + list(actionLines, "No action data recorded for this turn.");
  }

  function setTrace(document, source) {
    try { core.validateTrace(document); trace = document; step = 0; var farmCount = document.turns.length ? document.turns[0].canonical_state.farms.length : 0; seat = Number.isInteger(document.metadata.seat) && document.metadata.seat >= 0 && document.metadata.seat < farmCount ? document.metadata.seat : 0; makeSeatToggle(); render(); setStatus("Loaded " + source + " · " + document.turns.length + " turn" + (document.turns.length === 1 ? "" : "s"), "ok"); }
    catch (error) { stop(); trace = null; render(); setStatus(error.message || String(error), "error"); }
  }
  function makeSeatToggle() {
    var count = trace && trace.turns.length ? trace.turns[0].canonical_state.farms.length : 0;
    $("seat-toggle").innerHTML = "";
    for (var index = 0; index < count; index += 1) { var button = document.createElement("button"); button.type = "button"; button.className = "toggle-button" + (index === seat ? " active" : ""); button.textContent = "Seat " + index; button.dataset.seat = index; button.onclick = function () { seat = Number(this.dataset.seat); makeSeatToggle(); render(); }; $("seat-toggle").appendChild(button); }
  }
  function loadFile(file) { if (!file) return; file.text().then(function (contents) { setTrace(JSON.parse(contents), file.name); }).catch(function (error) { setTrace(null, file.name); setStatus("Could not load " + file.name + ": " + (error.message || error), "error"); }); }
  function loadUrl(url) { setStatus("Loading " + url + "…"); fetch(url).then(function (response) { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); }).then(function (document) { setTrace(document, url); }).catch(function (error) { setStatus("Could not load trace URL: " + (error.message || error), "error"); }); }

  document.addEventListener("DOMContentLoaded", function () {
    $("trace-file").addEventListener("change", function () { loadFile(this.files[0]); });
    $("drop-zone").addEventListener("dragover", function (event) { event.preventDefault(); this.classList.add("dragging"); });
    $("drop-zone").addEventListener("dragleave", function () { this.classList.remove("dragging"); });
    $("drop-zone").addEventListener("drop", function (event) { event.preventDefault(); this.classList.remove("dragging"); loadFile(event.dataTransfer.files[0]); });
    $("step-slider").addEventListener("input", function () { stop(); step = Number(this.value); render(); });
    $("step-back").onclick = function () { stop(); if (step > 0) step -= 1; render(); };
    $("step-forward").onclick = function () { if (trace && step < trace.turns.length - 1) step += 1; if (!trace || step >= trace.turns.length - 1) stop(); render(); };
    $("play-toggle").onclick = function () { if (!trace || !trace.turns.length) return; if (timer !== null) stop(); else { if (step >= trace.turns.length - 1) step = 0; timer = window.setInterval(function () { if (step >= trace.turns.length - 1) { stop(); render(); } else { step += 1; render(); } }, Number($("speed-select").value)); $("play-toggle").textContent = "❚❚ Pause"; } };
    $("speed-select").onchange = function () { if (timer !== null) { stop(); $("play-toggle").click(); } };
    $("clear-trace").onclick = function () { stop(); trace = null; $("seat-toggle").innerHTML = ""; render(); setStatus("Choose a trace JSON file or drop one here."); };
    document.addEventListener("keydown", function (event) { if (event.target && ["INPUT", "SELECT", "TEXTAREA"].indexOf(event.target.tagName) !== -1) return; if (event.key === "ArrowLeft") { $("step-back").click(); event.preventDefault(); } if (event.key === "ArrowRight") { $("step-forward").click(); event.preventDefault(); } if (event.key === " ") { $("play-toggle").click(); event.preventDefault(); } });
    var query = new URLSearchParams(window.location.search); var traceUrl = query.get("trace") || query.get("url"); if (traceUrl) loadUrl(traceUrl);
  });
}());
