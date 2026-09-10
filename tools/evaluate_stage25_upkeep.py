"""Paired old-PPO upkeep ablation; candidate-only changes, stochastic decode.

Run as python -m tools.evaluate_stage25_upkeep --help. No training occurs.

Episode identity contract (do not change): for a seeds list of length N run
with ``--master-seed M``, the game for ``seeds[index]`` at seat ``s`` always
uses ``episode_id = M * (2 * N) + 2 * index + s``, independent of the variant
and of how many variants are selected. ``--game-filter SEED:SEAT`` selects a
subset of games while preserving these identities; passing a shorter
``--seeds`` list instead DOES renumber every game and must not be compared
against a panel run with different seeds.
"""
from __future__ import annotations
import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

from rl_manager.stage25_capture import game_dir, write_game_capture

VARIANTS = {
    'baseline': (False, False, False),
    'care': (True, False, False),
    'fertilizer': (False, True, False),
    'fertilizer_wheat3': (False, True, True),
    'combined': (True, True, False),
    'combined_wheat3': (True, True, True),
}
COMPARISON_REFERENCES = {
    'baseline': None,
    'care': 'baseline',
    'fertilizer': 'baseline',
    'fertilizer_wheat3': 'fertilizer',
    'combined': 'baseline',
    'combined_wheat3': 'combined',
}


def episode_id_for(master_seed: int, num_seeds: int, index: int,
                   seat: int) -> int:
    """Original sampling identity; depends on the FULL seeds list order."""
    return master_seed * (2 * num_seeds) + 2 * index + seat


def parse_game_filter(values: list[str] | None,
                      seeds: list[int]) -> set[tuple[int, int]] | None:
    """Validate SEED:SEAT selections against the full seeds list.

    Returns ``{(index, seat)}`` using positions in the original ``seeds``
    order so episode IDs are preserved, or None when no filter was given.
    """
    if not values:
        return None
    selected: set[tuple[int, int]] = set()
    for value in values:
        try:
            seed_text, seat_text = value.split(':', 1)
            seed, seat = int(seed_text), int(seat_text)
        except (AttributeError, ValueError):
            raise ValueError(f'game filter must be SEED:SEAT, got {value!r}')
        if seat not in (0, 1):
            raise ValueError(f'game filter seat must be 0 or 1, got {value!r}')
        if seed not in seeds:
            raise ValueError(
                f'game filter seed {seed} is not in --seeds {seeds}; '
                'extend --seeds instead of renumbering')
        selected.add((seeds.index(seed), seat))
    return selected


def _capture_telemetry(result, candidate_seat: int) -> dict[str, object]:
    """Extract only scalar telemetry already exposed by executor capture.

    Capture/audit-only measures such as completed useful work, duplicate claims,
    and target abandonment are intentionally not inferred here.  The sharded
    report will mark them unavailable until the capture audit exposes them.
    """
    diagnostics = getattr(result, 'executor_full_diagnostics', None)
    if not isinstance(diagnostics, (list, tuple)) \
            or candidate_seat >= len(diagnostics):
        return {}
    candidate = diagnostics[candidate_seat]
    if not isinstance(candidate, Mapping):
        return {}
    days = candidate.get('days')
    if not isinstance(days, Mapping):
        return {}
    hiring_expense = 0.0
    missed_maintenance = 0
    movement = 0
    scheduler_runtime = 0.0
    for record in days.values():
        if not isinstance(record, Mapping):
            continue
        previous_labor = record.get('previous_labor')
        if isinstance(previous_labor, Mapping):
            value = previous_labor.get('hire_cost')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                hiring_expense += float(value)
        missed = record.get('missed_maintenance')
        if isinstance(missed, (list, tuple, set, Mapping)):
            missed_maintenance += len(missed)
        foreman_counts = record.get('foreman_counts')
        if isinstance(foreman_counts, Mapping):
            value = foreman_counts.get('movement')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                movement += int(value)
        scheduler = record.get('scheduler')
        if isinstance(scheduler, Mapping):
            value = scheduler.get('runtime_ms')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scheduler_runtime += float(value)
    return {
        'hiring_expense': hiring_expense,
        'missed_maintenance': missed_maintenance,
        'movement_between_interactions': movement,
        'scheduler_runtime': scheduler_runtime,
        'source': 'executor_full_diagnostics',
    }


@dataclass(frozen=True)
class UpkeepFactory:
    candidate_seat: int
    variant: str
    name: str = 'stage25_upkeep'
    capture: bool = False
    underfoot_first: bool = False
    deadline_safe_planting: bool = False
    deadline_safe_hiring: bool = False
    persistent_worker_queues: bool = False
    queue_ownership_repair: bool = False
    batch_reserved_supplies: bool = False
    underfoot_queue_insertion: bool = False
    schedule_informed_hiring: bool = False
    schedule_hiring_economic_repair: bool = False
    starvation_workload_visibility_repair: bool = False
    suppress_expansion_from_prior_debt: bool = True

    @property
    def version(self) -> str:
        base = f'v1:{self.variant}:candidate-seat-{self.candidate_seat}'
        if self.persistent_worker_queues:
            base += ':persistent-worker-queues'
            if self.queue_ownership_repair:
                base += ':queue-ownership-repair'
                if self.batch_reserved_supplies:
                    base += ':batch-reserved-supplies'
                if self.underfoot_queue_insertion:
                    base += ':underfoot-queue-insertion'
        if self.schedule_informed_hiring:
            base += ':schedule-informed-hiring'
            if self.schedule_hiring_economic_repair:
                base += ':economic-repair'
        if not self.suppress_expansion_from_prior_debt:
            base += ':prior-debt-expansion-veto-off'
        return base + ':capture' if self.capture else base

    def create(self, *, backend_name, seat, configuration, provider):
        from executor_v0.agent import AgentConfig, make_agent
        from executor_v0.foreman import ForemanConfig
        candidate = seat == self.candidate_seat
        care, fert, wheat3 = (
            VARIANTS[self.variant] if candidate
            else (False, False, False))
        # Capture enables read-only per-turn snapshots only; the returned
        # primitive action is computed before any snapshot exists.
        return make_agent(provider=provider, seat=seat, config=AgentConfig(
            strict=True, optional_spare_watering=True,
            record_turn_snapshot=self.capture,
            suppress_expansion_from_prior_debt=(
                self.suppress_expansion_from_prior_debt if candidate else True),
            foreman=ForemanConfig(
                underfoot_first=self.underfoot_first and candidate),
            deadline_safe_planting=(self.deadline_safe_planting and candidate),
            deadline_safe_hiring=(self.deadline_safe_hiring and candidate),
            persistent_worker_queues=(self.persistent_worker_queues and candidate),
            queue_ownership_repair=(
                self.queue_ownership_repair
                and self.persistent_worker_queues
                and candidate),
            batch_reserved_supplies=(
                self.batch_reserved_supplies
                and self.queue_ownership_repair
                and self.persistent_worker_queues
                and candidate),
            underfoot_queue_insertion=(
                self.underfoot_queue_insertion
                and self.queue_ownership_repair
                and self.persistent_worker_queues
                and candidate),
            schedule_informed_hiring=(self.schedule_informed_hiring and candidate),
            schedule_hiring_economic_repair=(
                self.schedule_hiring_economic_repair and candidate),
            starvation_workload_visibility_repair=(
                self.starvation_workload_visibility_repair and candidate),
            heuristic_care=care, heuristic_fertilizer=fert,
            wheat_harvest_threshold=wheat3))


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--e-checkpoint', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--master-seed', type=int, default=25)
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--backend', choices=['fast','official'], default='official')
    p.add_argument('--e-history-version', default='E_LEGACY', choices=['E_LEGACY','E_CORRECTED_V1'])
    p.add_argument('--capture-dir', default=None, type=Path,
                   help='opt-in paired replay/executor capture root; default off (no capture)')
    p.add_argument('--underfoot-first', action='store_true')
    p.add_argument('--deadline-safe-planting', action='store_true')
    p.add_argument('--deadline-safe-hiring', action='store_true')
    p.add_argument('--persistent-worker-queues', action='store_true')
    p.add_argument('--queue-ownership-repair', action='store_true',
                   help='candidate-only repair; effective only with persistent worker queues')
    p.add_argument('--batch-reserved-supplies', action='store_true',
                   help='candidate-only repair; effective with queue ownership repair')
    p.add_argument('--underfoot-queue-insertion', action='store_true',
                   help='candidate-only repair; effective with queue ownership repair')
    p.add_argument('--schedule-informed-hiring', action='store_true')
    p.add_argument('--schedule-hiring-economic-repair', action='store_true',
                   help='candidate-only repair; meaningful with schedule-informed hiring')
    p.add_argument('--starvation-workload-visibility-repair', action='store_true',
                   help='candidate-only repair')
    p.add_argument('--suppress-expansion-from-prior-debt',
                   choices=('on', 'off'), default='on',
                   help='candidate-only prior-day work-debt expansion veto; historical default is on')
    p.add_argument('--game-filter', nargs='*', default=None, metavar='SEED:SEAT',
                   help='run only these SEED:SEAT games, preserving original episode IDs')
    return p


def main(argv=None) -> None:
    p=build_parser()
    args=p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        p.error('seeds and variants must be unique')
    if args.master_seed < 0:
        p.error('master seed must be nonnegative')
    missing_threshold_refs = [
        f'{variant} -> {COMPARISON_REFERENCES[variant]}'
        for variant in ('fertilizer_wheat3', 'combined_wheat3')
        if variant in args.variants
        and COMPARISON_REFERENCES[variant] not in args.variants
    ]
    if missing_threshold_refs:
        p.error('include threshold comparison references: ' +
                ', '.join(missing_threshold_refs))
    try:
        game_selection = parse_game_filter(args.game_filter, list(args.seeds))
    except ValueError as exc:
        p.error(str(exc))
    for path in (args.checkpoint, args.e_checkpoint):
        if not path.is_file():
            p.error(f'missing checkpoint: {path}')
    capture = args.capture_dir is not None
    if capture:
        args.capture_dir.mkdir(parents=True,exist_ok=False)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    # Record the effective control state, not an ineffective repair request
    # supplied without its persistent-queue prerequisite.
    manifest['queue_ownership_repair'] = bool(
        args.queue_ownership_repair and args.persistent_worker_queues)
    manifest['batch_reserved_supplies'] = bool(
        args.batch_reserved_supplies and manifest['queue_ownership_repair'])
    manifest['underfoot_queue_insertion'] = bool(
        args.underfoot_queue_insertion and manifest['queue_ownership_repair'])
    manifest['starvation_workload_visibility_repair'] = bool(
        args.starvation_workload_visibility_repair)
    manifest['suppress_expansion_from_prior_debt'] = (
        args.suppress_expansion_from_prior_debt == 'on')
    manifest.update(schema_version=1, stochastic=True, opening='standard_mixed',
                    games=2*len(args.seeds)*len(args.variants),
                    comparison_references={name: COMPARISON_REFERENCES[name]
                                           for name in args.variants},
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                    source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest())
    if capture:
        manifest.update(capture_schema_version=1)
    for name,path in [('ppo',args.checkpoint),('bc_e',args.e_checkpoint)]:
        manifest[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
    from bc_manager_jax.checkpoint import load_torch_checkpoint
    from bc_manager_jax.model import ManagerConfig
    from rl_manager.ppo_checkpoint import load_ppo_checkpoint
    from rl_manager.ppo_policy import CurriculumMaskConfig
    from rl_manager.ppo_adapter import ppo_batched_policy_from_state
    from rl_manager.policy import JaxEPlanPolicy
    from rl_manager.runner import RunnerConfig, SelfPlayRunner, build_episode_spec
    from rl_manager.evaluation import summarize_evaluation
    frozen,metadata=load_torch_checkpoint(args.e_checkpoint,expected_e_history_version=args.e_history_version)
    config=ManagerConfig(**metadata['model_config'])
    state,checkpoint_meta=load_ppo_checkpoint(args.checkpoint,config=config,expected_e_history_version=args.e_history_version)
    candidate=ppo_batched_policy_from_state(state,config,name='ppo_candidate',deterministic=False,
        e_history_version=args.e_history_version,curriculum=CurriculumMaskConfig.from_json_dict(checkpoint_meta.get('curriculum')))
    opponent=JaxEPlanPolicy(frozen,config,name='frozen_e',e_history_version=args.e_history_version)
    runner_config=RunnerConfig(backend_name=args.backend,e_history_version=args.e_history_version,
                               low_telemetry=True,backend_configuration={'seed':0,'numThreads':1},
                               record_rollout=capture, record_debug_trace=capture,
                               record_executor_full_diagnostics=capture,
                               record_official_replay=capture)
    all_rows=[]
    for variant in args.variants:
        results=[]
        for index,seed in enumerate(args.seeds):
            for seat,orientation in enumerate(('candidate_vs_frozen','frozen_vs_candidate')):
                if game_selection is not None and (index,seat) not in game_selection:
                    continue
                episode_id=episode_id_for(args.master_seed,len(args.seeds),index,seat)
                spec=build_episode_spec(episode_id,seed,orientation,candidate,opponent)
                runner=SelfPlayRunner(runner_config,
                                      executor_factory=UpkeepFactory(
                                          seat, variant, capture=capture,
                                          underfoot_first=args.underfoot_first,
                                          deadline_safe_planting=args.deadline_safe_planting,
                                          deadline_safe_hiring=args.deadline_safe_hiring,
                                          persistent_worker_queues=args.persistent_worker_queues,
                                          queue_ownership_repair=args.queue_ownership_repair,
                                          batch_reserved_supplies=args.batch_reserved_supplies,
                                          underfoot_queue_insertion=args.underfoot_queue_insertion,
                                          schedule_informed_hiring=args.schedule_informed_hiring,
                                          schedule_hiring_economic_repair=(
                                              args.schedule_hiring_economic_repair),
                                          starvation_workload_visibility_repair=(
                                              args.starvation_workload_visibility_repair),
                                          suppress_expansion_from_prior_debt=(
                                              args.suppress_expansion_from_prior_debt == 'on')),
                                       master_seed=args.master_seed)
                result=runner.run([spec])[0]
                if game_selection is None:
                    results.append(result)
                    summary=summarize_evaluation(results,expected_seeds=args.seeds,
                        provenance={'manifest':manifest,'variant':variant,
                                    'engine':runner.provenance['backend'],
                                    'executor':{'name':runner.executor_factory.name,'version':runner.executor_factory.version},
                                    'candidate_identity':candidate.identity.to_json_dict(),
                                    'opponent_identity':opponent.identity.to_json_dict()})
                    (args.output_dir/f'{variant}.partial.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
                bank = float(result.final_banks[seat])
                opp = float(result.final_banks[1-seat])
                row={'variant':variant,'seed':seed,'seat':seat,'episode_id':episode_id,
                     'bank':bank,'opponent_bank':opp,
                     'margin':bank-opp,
                     'statuses':list(result.statuses)}
                if capture:
                    row['telemetry'] = _capture_telemetry(result, seat)
                    directory=game_dir(args.capture_dir,variant,episode_id,seed,seat)
                    meta={'variant':variant,'seed':seed,'seat':seat,
                          'episode_id':episode_id,'master_seed':args.master_seed,
                          'candidate_seat':seat,'composition':orientation,
                          'final_banks':[float(b) for b in result.final_banks],
                          'margin':float(result.margin),
                          'winner_seat':int(result.winner_seat),
                          'rewards':[float(r) for r in result.rewards],
                          'statuses':list(result.statuses),
                          'terminated':bool(result.terminated),
                          'trace_digest':str(result.trace_digest),
                          'engine':runner.provenance['backend'],
                          'executor':{'name':runner.executor_factory.name,
                                      'version':runner.executor_factory.version},
                          'candidate_identity':candidate.identity.to_json_dict(),
                          'opponent_identity':opponent.identity.to_json_dict()}
                    report=write_game_capture(
                        directory, meta=meta, debug_trace=result.debug_trace,
                        rollout=result.rollout,
                        executor_full_diagnostics=result.executor_full_diagnostics,
                        official_replay=result.official_replay,
                        status_history=result.status_history)
                    row['capture']=str(directory)
                    row['capture_complete']=bool(report.get('complete'))
                    print(json.dumps({'capture':row['capture'],
                                      'complete':row['capture_complete'],
                                      **({'capture_error':report['capture_error']}
                                         if not report.get('complete') else {})}),flush=True)
                    if not report.get('complete'):
                        print(f"WARNING: partial capture for {row['capture']}; "
                              f"game result stands, investigate before the panel",flush=True)
                all_rows.append(row)
                with (args.output_dir/'games.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                if list(result.statuses)!=['DONE','DONE'] or not result.terminated:
                    raise RuntimeError('Incomplete/failed game; partial results saved; stop before comparing')
        if game_selection is None:
            (args.output_dir/f'{variant}.partial.json').rename(args.output_dir/f'{variant}.json')
    if game_selection is not None:
        print(json.dumps({'filtered_games':len(all_rows),
                          'note':'game-filter run preserves original episode IDs; '
                                 'rerun the full panel for summaries/comparison'}),flush=True)
        return
    rows_by_variant = {
        variant: {(r['seed'], r['seat']): r for r in all_rows
                  if r['variant'] == variant}
        for variant in args.variants
    }
    comparison=[]
    for variant in args.variants:
        rows=[r for r in all_rows if r['variant']==variant]
        reference = COMPARISON_REFERENCES[variant]
        reference_rows = rows_by_variant.get(reference, {}) if reference else {}
        deltas = [r['bank'] - reference_rows[r['seed'], r['seat']]['bank']
                  for r in rows if reference and
                  (r['seed'], r['seat']) in reference_rows]
        margin_deltas = [
            r['margin'] - reference_rows[r['seed'], r['seat']]['margin']
            for r in rows if reference and
            (r['seed'], r['seat']) in reference_rows]
        comparison.append({'variant':variant,'games':len(rows),
                           'mean_bank':sum(r['bank'] for r in rows)/len(rows),
                           'mean_opponent_bank': sum(r['opponent_bank'] for r in rows) / len(rows),
                           'mean_margin': sum(r['margin'] for r in rows) / len(rows),
                           'comparison_reference': reference,
                           'mean_paired_bank_delta': (sum(deltas)/len(deltas)
                                                      if deltas else None),
                           'paired_bank_deltas':deltas,
                           'mean_paired_margin_delta': (sum(margin_deltas) / len(margin_deltas)
                                                       if margin_deltas else None),
                           'paired_margin_deltas': margin_deltas})
    (args.output_dir/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2),flush=True)


if __name__ == '__main__':
    main()
