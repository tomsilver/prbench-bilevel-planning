"""Tests for motion2d.py."""

import time

import imageio.v2 as iio
import numpy as np
import prbench
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

prbench.register_all_environments()


def test_motion2d_observation_to_state():
    """Tests for observation_to_state() in the Motion2D environment."""
    env = prbench.make("prbench/Motion2D-p2-v0")
    env_models = create_bilevel_planning_models(
        "motion2d", env.observation_space, env.action_space, num_passages=2
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_motion2d_transition_fn():
    """Tests for transition_fn() in the Motion2D environment."""
    env = prbench.make("prbench/Motion2D-p2-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "motion2d", env.observation_space, env.action_space, num_passages=2
    )
    transition_fn = env_models.transition_fn
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    for _ in range(100):
        executable = env.action_space.sample()
        obs, _, _, _, _ = env.step(executable)
        next_state = env_models.observation_to_state(obs)
        predicted_next_state = transition_fn(state, executable)
        assert next_state == predicted_next_state
        state = next_state
    env.close()


def test_motion2d_goal_deriver():
    """Tests for goal_deriver() in the Motion2D environment."""
    env = prbench.make("prbench/Motion2D-p2-v0")
    env_models = create_bilevel_planning_models(
        "motion2d", env.observation_space, env.action_space, num_passages=2
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(At robot target_region)"


def test_motion2d_state_abstractor():
    """Tests for state_abstractor() in the Motion2D environment."""
    env = prbench.make("prbench/Motion2D-p2-v0")
    env_models = create_bilevel_planning_models(
        "motion2d", env.observation_space, env.action_space, num_passages=2
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    AtTgt = pred_name_to_pred["AtTgt"]
    NotAtTgt = pred_name_to_pred["NotAtTgt"]
    AtPassage = pred_name_to_pred["AtPassage"]
    NotAtPassage = pred_name_to_pred["NotAtPassage"]
    NotAtAnyPassage = pred_name_to_pred["NotAtAnyPassage"]


    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_region = obj_name_to_obj["target_region"]
    obstacles = [o for o in abstract_state.objects if o.name.startswith("obstacle")]
    assert len(obstacles) == 4, "Expected 4 obstacles in the environment"
    obstacle1 = obstacles[0]
    obstacle2 = obstacles[1]
    obstacle3 = obstacles[2]

    # Initially robot should not be at target
    assert NotAtTgt([robot, target_region]) in abstract_state.atoms
    assert AtTgt([robot, target_region]) not in abstract_state.atoms
    assert NotAtAnyPassage([robot]) in abstract_state.atoms
    assert NotAtPassage([robot, obstacle1, obstacle2]) in abstract_state.atoms
    assert NotAtPassage([robot, obstacle1, obstacle3]) not in abstract_state.atoms

    # Create state where robot is in the target region
    state1 = state.copy()
    target_x = state1.get(target_region, "x")
    target_y = state1.get(target_region, "y")
    target_width = state1.get(target_region, "width")
    target_height = state1.get(target_region, "height")
    
    # Position robot in the center of target region
    robot_x = target_x + target_width / 2
    robot_y = target_y + target_height / 2
    state1.set(robot, "x", robot_x)
    state1.set(robot, "y", robot_y)

    abstract_state1 = state_abstractor(state1)
    assert AtTgt([robot, target_region]) in abstract_state1.atoms
    assert NotAtTgt([robot, target_region]) not in abstract_state1.atoms

    # Create state where robot is at a passage
    state2 = state.copy()
    obstacle1_x = state2.get(obstacle1, "x")
    obstacle1_y = state2.get(obstacle1, "y")
    obstacle1_width = state2.get(obstacle1, "width")
    obstacle2_x = state2.get(obstacle2, "x")
    obstacle2_y = state2.get(obstacle2, "y") + state2.get(obstacle2, "height")
    assert obstacle1_x == obstacle2_x, "Obstacles should be aligned"
    # Position robot far from target
    state2.set(robot, "x", obstacle1_x + obstacle1_width / 2)
    state2.set(robot, "y", (obstacle1_y + obstacle2_y) / 2)

    abstract_state2 = state_abstractor(state2)
    assert AtPassage([robot, obstacle1, obstacle2]) in abstract_state2.atoms
    assert NotAtPassage([robot, obstacle1, obstacle2]) not in abstract_state2.atoms
    assert NotAtAnyPassage([robot]) not in abstract_state2.atoms


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
    for _ in range(200):  # More steps for motion planning
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


def test_motion2d_skills():
    """Tests for skills in the Motion2D environment."""
    env = prbench.make("prbench/Motion2D-p1-v0")
    env_models = create_bilevel_planning_models(
        "motion2d", env.observation_space, env.action_space, num_passages=1
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    MoveTo = skill_name_to_skill["MoveTo"]

    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_region = obj_name_to_obj["target_region"]

    # Test MoveTo skill
    move_to_skill = MoveTo.ground((robot, target_region))
    obs1 = _skill_test_helper(move_to_skill, env_models, env, obs0, debug=True)
    
    # Check that robot reached the target region
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    At = pred_name_to_pred["At"]
    assert At([robot, target_region]) in abstract_state1.atoms


@pytest.mark.parametrize(
    "num_passages, max_abstract_plans, samples_per_step",
    [
        (1, 1, 10),
        (2, 1, 10),
        (3, 1, 10),
    ],
)
def test_motion2d_bilevel_planning(
    num_passages, max_abstract_plans, samples_per_step
):
    """Tests for bilevel planning in the Motion2D environment.

    Note that we only test a small number of passages to keep tests fast. Use experiment
    scripts to evaluate at scale.
    """

    env = prbench.make(
        f"prbench/Motion2D-p{num_passages}-v0", render_mode="rgb_array"
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Motion2D-p{num_passages}"
        )

    env_models = create_bilevel_planning_models(
        "motion2d",
        env.observation_space,
        env.action_space,
        num_passages=num_passages,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
        planning_timeout=30.0,  # Timeout for motion planning
    )
    obs, info = env.reset(seed=123)
    total_reward = 0
    agent.reset(obs, info)
    for _ in range(5000):  # Max steps for motion planning task
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break
    else:
        assert False, "Did not terminate successfully"

    env.close()