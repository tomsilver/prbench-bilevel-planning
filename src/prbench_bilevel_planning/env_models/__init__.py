"""Dynamically load bilevel planning env models."""

import importlib.util
import sys
from pathlib import Path

from gymnasium.spaces import Space

from prbench_bilevel_planning.structs import BilevelPlanningEnvModels


def create_bilevel_planning_models(
    env_name: str, observation_space: Space, executable_space: Space, **kwargs
) -> BilevelPlanningEnvModels:
    """Load bilevel planning models for the given environment."""
    current_file = Path(__file__).resolve()
    env_path = current_file.parent / f"{env_name}.py"

    if not env_path.exists():
        raise FileNotFoundError(f"No model file found for environment: {env_path}")

    module_name = f"{env_name}_module"
    spec = importlib.util.spec_from_file_location(module_name, env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {env_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "create_bilevel_planning_models"):
        raise AttributeError(
            f"{env_path} does not define `create_bilevel_planning_models`"
        )

    return module.create_bilevel_planning_models(
        observation_space, executable_space, **kwargs
    )
