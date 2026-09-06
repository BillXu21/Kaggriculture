"""Paired old-PPO upkeep ablation; candidate-only changes, stochastic decode.

Run as python -m tools.evaluate_stage25_upkeep --help. No training occurs.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

VARIANTS = {'baseline': (False, False), 'care': (True, False),
            'fertilizer': (False, True), 'combined': (True, True)}


@dataclass(frozen=True)
class UpkeepFactory:
    candidate_seat: int
    variant: str
    name: str = 'stage25_upkeep'

    @property
    def version(self) -> str:
        return f'v1:{self.variant}:candidate-seat-{self.candidate_seat}'

    def create(self, *, backend_name, seat, configuration, provider):
        from executor_v0.agent import AgentConfig, make_agent
        care, fert = VARIANTS[self.variant] if seat == self.candidate_seat else (False, False)
        return make_agent(provider=provider, seat=seat, config=AgentConfig(
            strict=True, optional_spare_watering=True, record_turn_snapshot=False,
            heuristic_care=care, heuristic_fertilizer=fert))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--e-checkpoint', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--master-seed', type=int, default=25)
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--backend', choices=['fast','official'], default='official')
    p.add_argument('--e-history-version', default='E_LEGACY', choices=['E_LEGACY','E_CORRECTED_V1'])
    args=p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        p.error('seeds and variants must be unique')
    if args.master_seed < 0:
        p.error('master seed must be nonnegative')
    if 'baseline' not in args.variants:
        p.error('include baseline for a paired comparison')
    for path in (args.checkpoint,args.e_checkpoint):
        if not path.is_file():p.error(f'missing checkpoint: {path}')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update(schema_version=1, stochastic=True, opening='standard_mixed',
                    games=2*len(args.seeds)*len(args.variants),
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                    source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest())
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
                               low_telemetry=True,backend_configuration={'seed':0,'numThreads':1})
    all_rows=[]
    for variant in args.variants:
        results=[]
        for index,seed in enumerate(args.seeds):
            for seat,orientation in enumerate(('candidate_vs_frozen','frozen_vs_candidate')):
                episode_id=args.master_seed * (2 * len(args.seeds)) + 2*index+seat
                spec=build_episode_spec(episode_id,seed,orientation,candidate,opponent)
                runner=SelfPlayRunner(runner_config,executor_factory=UpkeepFactory(seat,variant),master_seed=args.master_seed)
                result=runner.run([spec])[0]
                results.append(result)
                summary=summarize_evaluation(results,expected_seeds=args.seeds,
                    provenance={'manifest':manifest,'variant':variant,
                                'engine':runner.provenance['backend'],
                                'executor':{'name':runner.executor_factory.name,'version':runner.executor_factory.version},
                                'candidate_identity':candidate.identity.to_json_dict(),
                                'opponent_identity':opponent.identity.to_json_dict()})
                (args.output_dir/f'{variant}.partial.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
                bank=float(result.final_banks[seat]);opp=float(result.final_banks[1-seat])
                row={'variant':variant,'seed':seed,'seat':seat,'bank':bank,'opponent_bank':opp,
                     'statuses':list(result.statuses)}
                all_rows.append(row)
                with (args.output_dir/'games.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                if list(result.statuses)!=['DONE','DONE'] or not result.terminated:
                    raise RuntimeError('Incomplete/failed game; partial results saved; stop before comparing')
        (args.output_dir/f'{variant}.partial.json').rename(args.output_dir/f'{variant}.json')
    baseline={(r['seed'],r['seat']):r for r in all_rows if r['variant']=='baseline'}
    comparison=[]
    for variant in args.variants:
        rows=[r for r in all_rows if r['variant']==variant]
        deltas=[r['bank']-baseline[r['seed'],r['seat']]['bank'] for r in rows]
        comparison.append({'variant':variant,'games':len(rows),
                           'mean_bank':sum(r['bank'] for r in rows)/len(rows),
                           'mean_paired_bank_delta':sum(deltas)/len(deltas),
                           'paired_bank_deltas':deltas})
    (args.output_dir/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
