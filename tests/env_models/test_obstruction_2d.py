"""Tests for obstruction_2d.py."""
from conftest import MAKE_VIDEOS
import pytest
import prbench
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models
from prbench_bilevel_planning.agent import BilevelPlanningAgent

prbench.register_all_environments()


def test_obstruction_2d_observation_to_state():
    """Tests for observation_to_state() in the Obstruction2D environment."""
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models("obstruction2d", env.observation_space, env.action_space,
                                                num_obstructions=1)
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset()
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)  # states are hashable for bilevel planning
    


@pytest.mark.parametrize("num_obstructions", [0])  # TODO[0, 1, 2]
def test_obstruction_2d_bilevel_planning(num_obstructions):
    """Tests for bilevel planning in the Obstruction2D environment.
    
    Note that we only test a small number of obstructions to keep tests fast. Use
    experiment scripts to evaluate at scale.
    """

    env = prbench.make(f"prbench/Obstruction2D-o{num_obstructions}-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        # TODO fix this...
        env = RecordVideo(env, "unit_test_videos") #, name_prefix=env.spec.id)

    env_models = create_bilevel_planning_models("obstruction2d", env.observation_space, env.action_space,
                                                num_obstructions=num_obstructions)
    agent = BilevelPlanningAgent(env_models, seed=123)

    obs, info = env.reset(seed=123)

    import imageio.v2 as iio
    imgs = [env.render()]
    # iio.imsave("debug-env.png", img)

    total_reward = 0
    agent.reset(obs, info)
    for _ in range(1000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)

        # TODO remove
        imgs.append(env.render())

        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break
    # TODO assert something about total reward

    env.close()

    # TODO remove
    iio.mimsave("unit_test_videos/obstruction2d.mp4", imgs)