"""Tests for clutteredstorage2d.py."""

import numpy as np
import prbench
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models
from prbench_bilevel_planning.env_models.clutteredstorage2d import (
    get_robot_transfer_position,
)

prbench.register_all_environments()


def test_clutteredstorage2d_observation_to_state():
    """Tests for observation_to_state() in the ClutteredStorage2D environment."""
    env = prbench.make("prbench/ClutteredStorage2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d", env.observation_space, env.action_space, num_blocks=1
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)  # states are hashable for bilevel planning
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_clutteredstorage2d_transition_fn():
    """Tests for transition_fn() in the ClutteredStorage2D environment."""
    env = prbench.make("prbench/ClutteredStorage2D-b1-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d", env.observation_space, env.action_space, num_blocks=1
    )
    transition_fn = env_models.transition_fn
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    for _ in range(100):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        predicted_next_state = transition_fn(state, action)
        assert next_state == predicted_next_state
        state = next_state
    env.close()


def test_clutteredstorage2d_goal_deriver():
    """Tests for goal_deriver() in the ClutteredStorage2D environment."""
    env = prbench.make("prbench/ClutteredStorage2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d", env.observation_space, env.action_space, num_blocks=1
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(OnShelf block0)"

    # Test with multiple blocks
    env_multi = prbench.make("prbench/ClutteredStorage2D-b7-v0")
    env_models_multi = create_bilevel_planning_models(
        "clutteredstorage2d", env_multi.observation_space, env_multi.action_space, num_blocks=7
    )
    goal_deriver_multi = env_models_multi.goal_deriver
    obs_multi, _ = env_multi.reset(seed=123)
    state_multi = env_models_multi.observation_to_state(obs_multi)
    goal_multi = goal_deriver_multi(state_multi)
    assert len(goal_multi.atoms) == 7
    goal_atoms = {str(atom) for atom in goal_multi.atoms}
    expected_atoms = {f"(OnShelf block{i})" for i in range(7)}
    assert goal_atoms == expected_atoms
    env_multi.close()


def test_clutteredstorage2d_state_abstractor():
    """Tests for state_abstractor() in the ClutteredStorage2D environment."""
    env = prbench.make("prbench/ClutteredStorage2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d", env.observation_space, env.action_space, num_blocks=1
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    Holding = pred_name_to_pred["Holding"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    NotOnShelf = pred_name_to_pred["NotOnShelf"]
    OnShelf = pred_name_to_pred["OnShelf"]
    env.reset(seed=123)
    obs, _, _, _, _ = env.step((0, 0, 0, 0.1, 0.0))  # extend the arm
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    block0 = obj_name_to_obj["block0"]
    shelf = obj_name_to_obj["shelf"]
    
    # Check that we have the expected objects
    assert "robot" in obj_name_to_obj
    assert "shelf" in obj_name_to_obj
    assert "block0" in obj_name_to_obj
    
    # Initially, robot hand should be empty and block should be either on shelf or not on shelf
    assert HandEmpty([robot]) in abstract_state.atoms
    assert (
        NotOnShelf([block0]) in abstract_state.atoms
        or OnShelf([block0]) in abstract_state.atoms
    )
    
    # For now, just test that the basic state abstraction works
    # We'll test holding behavior through the skills test instead
    env.close()


def _skill_test_helper(ground_skill, env_models, env, obs, params=None):
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    operator = ground_skill.operator
    assert operator.preconditions.issubset(abstract_state.atoms)
    predicted_next_atoms = (
        abstract_state.atoms - operator.delete_effects
    ) | operator.add_effects
    controller = ground_skill.controller
    if params is None:
        params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    for _ in range(100):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        controller.observe(next_state)
        assert env_models.transition_fn(state, action) == next_state
        state = next_state

        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"
    next_abstract_state = env_models.state_abstractor(state)
    assert next_abstract_state.atoms == predicted_next_atoms
    return obs


def test_clutteredstorage2d_skills():
    """Tests for skills in the ClutteredStorage2D environment."""
    env = prbench.make("prbench/ClutteredStorage2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d", env.observation_space, env.action_space, num_blocks=1
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    PickBlockNotOnShelf = skill_name_to_skill["PickBlockNotOnShelf"]
    PickBlockOnShelf = skill_name_to_skill["PickBlockOnShelf"]
    PlaceBlockNotOnShelf = skill_name_to_skill["PlaceBlockNotOnShelf"]
    PlaceBlockOnShelf = skill_name_to_skill["PlaceBlockOnShelf"]
    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    block0 = obj_name_to_obj["block0"]
    pick_block_not_on_shelf = PickBlockNotOnShelf.ground((robot, block0))
    pick_block_on_shelf = PickBlockOnShelf.ground((robot, block0))
    place_block_not_on_shelf = PlaceBlockNotOnShelf.ground((robot, block0))
    place_block_on_shelf = PlaceBlockOnShelf.ground((robot, block0))
    
    # Determine initial block state
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    NotOnShelf = pred_name_to_pred["NotOnShelf"]
    OnShelf = pred_name_to_pred["OnShelf"]
    
    initial_abstract_state = state_abstractor(state0)
    block_not_on_shelf = NotOnShelf([block0]) in initial_abstract_state.atoms
    block_on_shelf = OnShelf([block0]) in initial_abstract_state.atoms
    
    if block_not_on_shelf:
        # Test picking the block when not on shelf.
        obs1 = _skill_test_helper(pick_block_not_on_shelf, env_models, env, obs0)
        # Test placing the block on the shelf.
        obs2 = _skill_test_helper(place_block_on_shelf, env_models, env, obs1)
        # Test picking the block from the shelf.
        obs3 = _skill_test_helper(pick_block_on_shelf, env_models, env, obs2)
        # Test placing the block not on shelf.
        _skill_test_helper(place_block_not_on_shelf, env_models, env, obs3)
    elif block_on_shelf:
        # Test picking the block from the shelf.
        obs1 = _skill_test_helper(pick_block_on_shelf, env_models, env, obs0)
        # Test placing the block not on shelf.
        obs2 = _skill_test_helper(place_block_not_on_shelf, env_models, env, obs1)
        # Test picking the block when not on shelf.
        obs3 = _skill_test_helper(pick_block_not_on_shelf, env_models, env, obs2)
        # Test placing the block back on the shelf.
        _skill_test_helper(place_block_on_shelf, env_models, env, obs3)
    else:
        assert False, "Block must be either on shelf or not on shelf initially"
    
    env.close()


@pytest.mark.parametrize(
    "num_blocks, max_abstract_plans, samples_per_step",
    [
        (1, 1, 1),
        (7, 10, 1),
    ],
)
def test_clutteredstorage2d_bilevel_planning(
    num_blocks, max_abstract_plans, samples_per_step
):
    """Tests for bilevel planning in the ClutteredStorage2D environment.

    Note that we only test a small number of blocks to keep tests fast. Use
    experiment scripts to evaluate at scale.
    """

    env = prbench.make(
        f"prbench/ClutteredStorage2D-b{num_blocks}-v0", render_mode="rgb_array"
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"ClutteredStorage2D-b{num_blocks}"
        )

    env_models = create_bilevel_planning_models(
        "clutteredstorage2d",
        env.observation_space,
        env.action_space,
        num_blocks=num_blocks,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
    )

    obs, info = env.reset(seed=123)

    total_reward = 0
    agent.reset(obs, info)
    for _ in range(1000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break
    else:
        assert False, "Did not terminate successfully"

    env.close()