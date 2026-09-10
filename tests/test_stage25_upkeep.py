"""Mechanics and old-model-compatible task integration for upkeep ablations."""
import pytest
from executor_v0.agent import AgentConfig, make_agent
from executor_v0.upkeep import care_has_payoff, fertilizer_extra_units
from executor_v0.tasks import generate_tasks
from test_executor_v0_tasks import make_obs, make_plan, plant_tile, animal_tile


def test_care_requires_feed_and_future_payable_production():
    cow = animal_tile('COW', fed_today=True)
    assert care_has_payoff(cow, 5)
    assert not care_has_payoff({**cow, 'fed_today': False}, 5)
    assert not care_has_payoff({**cow, 'cared_today': True}, 5)
    # Cow produces at 8,10,...28. Care banked on 27 is too late for 28.
    assert care_has_payoff(cow, 26)
    assert not care_has_payoff(cow, 27)
    assert not care_has_payoff(animal_tile('GOOSE'), 28)


@pytest.mark.parametrize('crop,age,watered,expected', [
    ('STRAWBERRY',9,True,2), ('STRAWBERRY',13,True,2),
    ('STRAWBERRY',15,True,1), ('STRAWBERRY',16,True,0),
    ('STRAWBERRY',0,True,0), ('WHEAT',2,False,1),
    ('WHEAT',2,True,0), ('CARROT',2,False,1),
    ('MELON',0,True,0),
])
def test_fertilizer_respects_age_water_order_and_harvest(crop,age,watered,expected):
    t=plant_tile(crop,planted_day=0,watered_today=watered,
                 yield_units=0 if crop=='STRAWBERRY' else 1)
    assert fertilizer_extra_units(t,age)==expected


def test_fertilizer_excludes_expiry_cap_and_endgame():
    t=plant_tile('STRAWBERRY',planted_day=0,yield_units=0)
    assert fertilizer_extra_units({**t,'fertilized_until_day':11},9)==0
    assert fertilizer_extra_units({**t,'yield_units':4},15)==0
    assert fertilizer_extra_units({**t,'planted_day':20},29)==0


def test_zero_model_heads_can_generate_care_and_fertilizer():
    board=[[None]*10 for _ in range(10)]
    board[0][0]=plant_tile('STRAWBERRY',planted_day=0,yield_units=0)
    board[0][1]=animal_tile('COW')
    obs=make_obs(day=9,step=218,tiles=board)
    obs['market']['prices']={'STRAWBERRY':120,'FERTILIZER':100}
    plan=make_plan(crop_targets={'STRAWBERRY':1},animal_targets={'COW':1})
    old=generate_tasks(obs,0,feasible_plan=plan,remaining_sells={})
    assert not any(t.kind in ('CARE','FERTILIZE') for t in old.tasks)
    new=generate_tasks(obs,0,feasible_plan=plan,remaining_sells={},
                       heuristic_care=True,heuristic_fertilizer=True)
    assert sum(t.kind=='CARE' for t in new.tasks)==1
    assert sum(t.kind=='FERTILIZE' for t in new.tasks)==1
    assert any(t.kind=='BUY_PRODUCT' and t.product=='FERTILIZER' for t in new.tasks)


def test_unprofitable_fertilizer_overrides_large_model_request():
    board=[[None]*10 for _ in range(10)]
    board[0][0]=plant_tile('WHEAT',planted_day=0,watered_today=False,yield_units=1)
    obs=make_obs(day=2,step=50,tiles=board)
    obs['market']['prices']={'WHEAT':25,'FERTILIZER':100}
    plan=make_plan(crop_targets={'WHEAT':1},fertilizer_by_crop={'WHEAT':100})
    result=generate_tasks(obs,0,feasible_plan=plan,remaining_sells={},heuristic_fertilizer=True)
    assert not any(t.kind=='FERTILIZE' for t in result.tasks)


def test_real_agent_accepts_unchanged_daily_plan():
    class Provider:
        def daily_plan(self,obs,seat,previous_execution=None):return make_plan()
    agent=make_agent(provider=Provider(),seat=0,config=AgentConfig(
        strict=True,heuristic_care=True,heuristic_fertilizer=True))
    assert 'farmer' in agent(make_obs(day=0,hour=0,step=0))


def test_ablation_factory_changes_candidate_only():
    from tools.evaluate_stage25_upkeep import UpkeepFactory
    class Provider:
        def daily_plan(self,obs,seat,previous_execution=None):return make_plan()
    for seat in (0,1):
        factory=UpkeepFactory(seat,'combined_wheat3')
        for acting_seat in (0,1):
            agent=factory.create(backend_name='official',seat=acting_seat,
                                 configuration={},provider=Provider())
            assert agent.config.heuristic_care == (seat==acting_seat)
            assert agent.config.heuristic_fertilizer == (seat==acting_seat)
            assert agent.config.wheat_harvest_threshold == (seat==acting_seat)


def test_prior_debt_expansion_veto_ablation_is_candidate_only():
    from tools.evaluate_stage25_upkeep import UpkeepFactory

    class Provider:
        def daily_plan(self, obs, seat, previous_execution=None):
            return make_plan()

    factory = UpkeepFactory(
        0, 'baseline', suppress_expansion_from_prior_debt=False)
    candidate = factory.create(
        backend_name='official', seat=0, configuration={}, provider=Provider())
    opponent = factory.create(
        backend_name='official', seat=1, configuration={}, provider=Provider())

    assert candidate.config.suppress_expansion_from_prior_debt is False
    assert opponent.config.suppress_expansion_from_prior_debt is True
    assert factory.version.endswith(':prior-debt-expansion-veto-off')


def test_fertilizer_order_never_blocks_survival():
    board=[[None]*10 for _ in range(10)]
    board[0][0]=plant_tile('WHEAT',planted_day=0,watered_today=False,yield_units=1)
    obs=make_obs(day=2,step=50,tiles=board)
    obs['market']['prices']={'WHEAT':100,'FERTILIZER':10}
    plan=make_plan(crop_targets={'WHEAT':1})
    new=generate_tasks(obs,0,feasible_plan=plan,remaining_sells={},heuristic_fertilizer=True)
    water=next(t for t in new.tasks if t.kind=='WATER')
    assert water.depends_on==('FERTILIZE:WHEAT:0,0',)
    board[0][0]['consecutive_unwatered']=1
    new=generate_tasks(obs,0,feasible_plan=plan,remaining_sells={},heuristic_fertilizer=True)
    water=next(t for t in new.tasks if t.kind=='WATER')
    assert not water.depends_on
    assert not any(t.kind=='FERTILIZE' for t in new.tasks)
