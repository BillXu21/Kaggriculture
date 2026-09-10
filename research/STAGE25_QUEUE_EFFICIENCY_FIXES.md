# Stage 2.5 queue-efficiency fixes

The repaired queue path now has two independent, default-off controls:

* `batch_reserved_supplies` lets one worker pick up compatible shed demand
  for multiple tasks in its own queue, bounded by `pickup_batch`, carried
  inventory, actual stock, and that worker's reservations. Consumption is
  reconciled before later queue transfers/releases so carried units are not
  counted again.
* `underfoot_queue_insertion` permits at most one immediately executable,
  equal-priority underfoot interaction before a valid queue head. The retained
  destination stays in the queue; a local conservative route/deadline check
  rejects insertions that would break it.

Both controls require `persistent_worker_queues` and `queue_ownership_repair`
and are candidate-only in the stage25 evaluator and sharded wrapper. Capture
and checkpoint promotion runs were not performed for this upkeep patch. The
targeted evidence remains unit-level: 57 queue/foreman tests for batching and
156 combined queue/foreman/agent/evaluator tests including insertion. These
tests do not establish full-game gains; fixed-seed capture replay remains an
external follow-up when the corresponding artifacts are available.
