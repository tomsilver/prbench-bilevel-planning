"""Tests for clutteredstorage2d.py."""

import time

import imageio.v2 as iio
import numpy as np
import prbench

from prbench_bilevel_planning.env_models import create_bilevel_planning_models

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
    assert str(goal_atom) == "(OnShelf block0 shelf)"

    # Test with multiple blocks
    env_multi = prbench.make("prbench/ClutteredStorage2D-b7-v0")
    env_models_multi = create_bilevel_planning_models(
        "clutteredstorage2d",
        env_multi.observation_space,
        env_multi.action_space,
        num_blocks=7,
    )
    goal_deriver_multi = env_models_multi.goal_deriver
    obs_multi, _ = env_multi.reset(seed=123)
    state_multi = env_models_multi.observation_to_state(obs_multi)
    goal_multi = goal_deriver_multi(state_multi)
    assert len(goal_multi.atoms) == 7
    goal_atoms = {str(atom) for atom in goal_multi.atoms}
    expected_atoms = {f"(OnShelf block{i} shelf)" for i in range(7)}
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

    # Initially, robot hand should be empty and block should be either on shelf or
    # not on shelf
    assert HandEmpty([robot]) in abstract_state.atoms
    assert (
        NotOnShelf([block0, shelf]) in abstract_state.atoms
        or OnShelf([block0, shelf]) in abstract_state.atoms
    )
    assert Holding([robot, block0]) not in abstract_state.atoms

    # For now, just test that the basic state abstraction works
    # We'll test holding behavior through the skills test instead
    env.close()


def _skill_test_helper(ground_skill, env_models, env, obs, params=None, debug=False):
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    operator = ground_skill.operator
    assert operator.preconditions.issubset(abstract_state.atoms)
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
        if debug:
            img = env.render()
            iio.imsave(f"debug/debug-test-{int(time.time()*1000.0)}.png", img)

        if controller.terminated():
            break
    return obs


def test_clutteredstorage2d_skills():
    """Tests for skills in the ClutteredStorage2D environment."""
    env = prbench.make("prbench/ClutteredStorage2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d", env.observation_space, env.action_space, num_blocks=1
    )
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    pick_block_not_on_shelf = skill_name_to_skill["PickBlockNotOnShelf"]
    # Other skills are available but not used in this basic test
    # pick_block_on_shelf = skill_name_to_skill["PickBlockOnShelf"]
    # place_block_not_on_shelf = skill_name_to_skill["PlaceBlockNotOnShelf"]
    # place_block_on_shelf = skill_name_to_skill["PlaceBlockOnShelf"]

    # Test pick and place the block that is not on shelf
    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    block0 = obj_name_to_obj["block0"]
    shelf = obj_name_to_obj["shelf"]
    NotOnShelf = pred_name_to_pred["NotOnShelf"]
    assert NotOnShelf([block0, shelf]) in abstract_state.atoms
    pick_block_not_on_shelf_skill = pick_block_not_on_shelf.ground(
        (robot, block0, shelf)
    )

    # First pick the block
    obs1 = _skill_test_helper(pick_block_not_on_shelf_skill, env_models, env, obs0)
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    assert pred_name_to_pred["Holding"]([robot, block0]) in abstract_state1.atoms
    env.close()
