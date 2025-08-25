"""Tests for tidybot3d ground.py."""

import math
from typing import Any, Dict

import numpy as np
import prbench
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs import Object, ObjectCentricState, Type

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models
from prbench_bilevel_planning.env_models.tidybot3d.ground import (
    PickController,
    PlaceController,
)
from prbench_bilevel_planning.env_models.tidybot3d.object_centric_adapter import (
    observation_to_object_centric_state,
)

prbench.register_all_environments()

@pytest.mark.parametrize("max_abstract_plans, samples_per_step", [(10, 1)])
def test_tidybot3d_bilevel_planning(max_abstract_plans, samples_per_step):
    """Tests for bilevel planning in the TidyBot3D environment.
    
    Note that we only test a small configuration to keep tests fast.
    """
    env = prbench.make("prbench/TidyBot3D-ground-o3-v0", render_mode="rgb_array", render_images=False, show_viewer=False)
    
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix="TidyBot3D-ground-o3"
        )
    
    env_models = create_bilevel_planning_models(
        "tidybot3d", env.observation_space, env.action_space
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
    for _ in range(100):  # Reduced iterations for faster testing
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break
    
    # Test should complete without errors
    env.close()


if __name__ == "__main__":
    pytest.main([__file__])
