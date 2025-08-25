"""Dynamically load bilevel planning env models."""

import abc
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from bilevel_planning.structs import GroundParameterizedController, SesameModels
from geom2drobotenvs.object_types import CRVRobotType
from geom2drobotenvs.structs import SE2Pose
from geom2drobotenvs.utils import CRVRobotActionSpace
from gymnasium.spaces import Space
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState


class Geom2dRobotController(GroundParameterizedController, abc.ABC):
    """General controller for 2D robot manipulation tasks using SE2 waypoints."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        safe_y: float = 0.8,
    ) -> None:
        self._robot = objects[0]
        assert self._robot.is_instance(CRVRobotType)
        super().__init__(objects)
        self._current_params: tuple[float, ...] | float = 0.0
        self._current_plan: list[NDArray[np.float32]] | None = None
        self._current_state: ObjectCentricState | None = None
        self._safe_y = safe_y
        # Extract max deltas from action space bounds
        self._max_delta_x = action_space.high[0]
        self._max_delta_y = action_space.high[1]
        self._max_delta_theta = action_space.high[2]
        self._max_delta_arm = action_space.high[3]

    @abc.abstractmethod
    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate a waypoint plan with SE2 pose and arm length values."""

    @abc.abstractmethod
    def _get_vacuum_actions(self) -> tuple[float, float]:
        """Get vacuum actions for during and after waypoint movement."""

    def _waypoints_to_plan(
        self,
        state: ObjectCentricState,
        waypoints: list[tuple[SE2Pose, float]],
        vacuum_during_plan: float,
    ) -> list[NDArray[np.float32]]:
        curr_x = state.get(self._robot, "x")
        curr_y = state.get(self._robot, "y")
        curr_theta = state.get(self._robot, "theta")
        curr_arm = state.get(self._robot, "arm_joint")
        current_pos = (SE2Pose(curr_x, curr_y, curr_theta), curr_arm)
        waypoints = [current_pos] + waypoints
        plan: list[NDArray[np.float32]] = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            start_pose = np.array([start[0].x, start[0].y, start[0].theta, start[1]])
            end_pose = np.array([end[0].x, end[0].y, end[0].theta, end[1]])
            if np.allclose(start_pose, end_pose):
                continue
            total_dx = end[0].x - start[0].x
            total_dy = end[0].y - start[0].y
            total_dtheta = end[0].theta - start[0].theta
            total_darm = end[1] - start[1]
            num_steps = int(
                max(
                    np.ceil(abs(total_dx) / self._max_delta_x),
                    np.ceil(abs(total_dy) / self._max_delta_y),
                    np.ceil(abs(total_dtheta) / self._max_delta_theta),
                    np.ceil(abs(total_darm) / self._max_delta_arm),
                )
            )
            dx = total_dx / num_steps
            dy = total_dy / num_steps
            dtheta = total_dtheta / num_steps
            darm = total_darm / num_steps
            action = np.array(
                [dx, dy, dtheta, darm, vacuum_during_plan], dtype=np.float32
            )
            for _ in range(num_steps):
                plan.append(action)

        return plan

    def reset(self, x: ObjectCentricState, params: tuple[float, ...] | float) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._current_plan is not None and len(self._current_plan) == 0

    def step(self) -> NDArray[np.float32]:
        assert self._current_state is not None
        if self._current_plan is None:
            self._current_plan = self._generate_plan(self._current_state)
        return self._current_plan.pop(0)

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
        waypoints = self._generate_waypoints(x)
        vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
        waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
        plan_suffix: list[NDArray[np.float32]] = [
            # Change the vacuum.
            np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
        ]
        return waypoint_plan + plan_suffix


__all__ = ["create_bilevel_planning_models"]


def create_bilevel_planning_models(
    env_name: str, observation_space: Space, executable_space: Space, **kwargs
) -> SesameModels:
    """Load bilevel planning models for the given environment."""
    current_file = Path(__file__).resolve()
    
    # Try different directories based on environment type
    possible_paths = [
        current_file.parent / "geom2d" / f"{env_name}.py",
        current_file.parent / "tidybot3d" / f"{env_name}.py",
    ]
    
    env_path = None
    for path in possible_paths:
        if path.exists():
            env_path = path
            break
    
    if env_path is None:
        raise FileNotFoundError(f"No model file found for environment '{env_name}' in any of: {possible_paths}")

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
