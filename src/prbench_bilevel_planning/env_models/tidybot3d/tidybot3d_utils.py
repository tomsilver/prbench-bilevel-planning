"""Tidybot 3D utilities for bilevel planning.

This module provides core utilities for implementing tidybot pick-and-place skills
in a 3D environment using the bilevel planning framework. It includes controllers,
state management, and skill primitives based on the original tidybot implementation.
"""

import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from bilevel_planning.structs import GroundParameterizedController
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState

# Import tidybot components
try:
    from prbench.envs.tidybot.arm_controller import ArmController
    from prbench.envs.tidybot.base_controller import BaseController
    from prbench.envs.tidybot.ik_solver import TidybotIKSolver
    from prbench.envs.tidybot.tidybot_mujoco_env import MujocoEnv
except ImportError:
    # Fallback for when tidybot components are not available
    ArmController = None
    BaseController = None
    TidybotIKSolver = None
    MujocoEnv = None


class PickState(Enum):
    """States of the pick subroutine."""
    APPROACH = "approach"
    LOWER = "lower"
    GRASP = "grasp"
    LIFT = "lift"


class PlaceState(Enum):
    """States of the place subroutine."""
    APPROACH = "approach"
    LOWER = "lower"
    RELEASE = "release"
    HOME = "home"


class TidybotController(GroundParameterizedController):
    """Base controller for Tidybot 3D manipulation tasks.
    
    This controller provides the foundation for implementing pick-and-place
    skills using the tidybot's mobile base and manipulator arm.
    """

    # Base following parameters
    LOOKAHEAD_DISTANCE = 0.3
    POSITION_TOLERANCE = 0.005
    GRASP_BASE_TOLERANCE = 0.002
    PLACE_BASE_TOLERANCE = 0.02

    # Object and target locations
    PLACEMENT_X_OFFSET = 1.0
    PLACEMENT_Y_OFFSET = 0.3
    PLACEMENT_Z_OFFSET = 0.0

    # Manipulation parameters
    ROBOT_BASE_HEIGHT = 0.48
    PICK_APPROACH_HEIGHT_OFFSET = 0.25
    PICK_LOWER_DIST = 0.08
    PICK_LIFT_DIST = 0.28
    PLACE_APPROACH_HEIGHT_OFFSET = 0.10

    # Grasping parameters
    GRASP_SUCCESS_THRESHOLD = 0.7
    GRASP_PROGRESS_THRESHOLD = 0.3
    GRASP_TIMEOUT_S = 3.0
    PLACE_SUCCESS_THRESHOLD = 0.2

    def __init__(
        self,
        objects: Sequence[Object],
        max_skill_horizon: int = 100,
        ee_offset: float = 0.12,
        custom_grasp: bool = False,
    ) -> None:
        super().__init__(objects)
        self._robot = objects[0]  # Assume first object is the robot
        self._max_skill_horizon = max_skill_horizon
        self._custom_grasp = custom_grasp
        
        # Motion planning state
        self.state: str = "idle"  # States: idle, moving, manipulating
        self.current_command: Optional[Dict[str, Any]] = None
        self.base_waypoints: List[List[float]] = []
        self.current_waypoint_idx: int = 0
        self.target_ee_pos: Optional[List[float]] = None
        self.grasp_state: Optional[Union[PickState, PlaceState]] = None
        
        # Object and target locations
        self.object_location: Optional[np.ndarray] = None
        self.target_location: Optional[np.ndarray] = None
        
        # Base following parameters
        self.lookahead_position: Optional[List[float]] = None
        
        # Episode control
        self.enabled: bool = True
        self.episode_ended: bool = False
        
        # Initialize IK solver if available
        self.ik_solver: Optional[TidybotIKSolver] = None
        if TidybotIKSolver is not None:
            self.ik_solver = TidybotIKSolver(ee_offset=ee_offset)
        
        # Current plan and parameters
        self._current_plan: List[Dict[str, Any]] = []
        self._current_params: tuple[float, ...] | float = 0.0
        self._current_state: Optional[ObjectCentricState] = None
        self._step_count: int = 0

    def sample_parameters(self, rng: Optional[np.random.Generator] = None) -> Union[tuple[float, ...], float]:
        """Return default parameters for this controller.
        
        This satisfies the abstract method requirement from the base interface.
        """
        return 0.0

    def reset(self, x: ObjectCentricState, params: tuple[float, ...] | float) -> None:
        """Reset the controller for a new episode."""
        self._current_params = params
        self._current_plan = []
        self._current_state = x
        self._step_count = 0
        
        # Reset motion planning state
        self.state = "idle"
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.episode_ended = False
        self.grasp_state = None
        
        # Reset object and target locations
        self.object_location = None
        self.target_location = None
        
        # Enable policy execution
        self.enabled = True

    def terminated(self) -> bool:
        """Check if the skill execution is terminated."""
        return (
            self.episode_ended or 
            self._step_count >= self._max_skill_horizon
        )

    def step(self) -> Dict[str, Any]:
        """Execute one step of the controller."""
        assert self._current_state is not None
        self._step_count += 1
        
        if self.terminated():
            return self._create_hold_action()
        
        if len(self._current_plan) == 0:
            self._current_plan = self._generate_plan(self._current_state)
        
        if len(self._current_plan) > 0:
            return self._current_plan.pop(0)
        
        return self._create_hold_action()

    def observe(self, x: ObjectCentricState) -> None:
        """Update the controller with new observations."""
        self._current_state = x

    def _generate_plan(self, state: ObjectCentricState) -> List[Dict[str, Any]]:
        """Generate a plan based on the current state.
        
        This method should be overridden by specific skill implementations.
        """
        return []

    def _create_hold_action(self) -> Dict[str, Any]:
        """Create an action that holds the current pose."""
        return {
            "base_pose": np.array([0.0, 0.0, 0.0]),  # Hold current position
            "arm_pos": np.array([0.14, 0.0, 0.21]),  # Default arm position
            "arm_quat": np.array([1.0, 0.0, 0.0, 0.0]),  # Gripper down
            "gripper_pos": np.array([0.0]),  # Gripper open
        }

    def detect_objects_from_state(self, state: ObjectCentricState) -> List[np.ndarray]:
        """Detect objects from the relational state.
        
        This method extracts object positions from the ObjectCentricState
        and returns them in the format expected by the tidybot policy.
        """
        detected_objects: List[np.ndarray] = []
        
        # Find all cube objects in the state
        cubes: List[Tuple[np.ndarray, str]] = []
        for obj in state.data.keys():
            if "cube" in obj.name.lower():
                # Extract position from state
                x = state.get(obj, "x")
                y = state.get(obj, "y") 
                z = state.get(obj, "z")  # Height must be present in state
                
                cube_pos = np.array([x, y, z])
                cubes.append((cube_pos, obj.name))
        
        if cubes:
            # Sort cubes by x position and select the one with smallest x value
            if self._custom_grasp:
                cubes.sort(key=lambda x: x[0][1])  # Sort by y
                target_cube_pos, target_cube_name = cubes[0]
            else:
                cubes.sort(key=lambda x: x[0][0])  # Sort by x
                target_cube_pos, target_cube_name = cubes[0]
            
            detected_objects.append(target_cube_pos)
        
        return detected_objects

    def distance(
        self,
        pt1: Union[Tuple[float, float], List[float], np.ndarray],
        pt2: Union[Tuple[float, float], List[float], np.ndarray],
    ) -> float:
        """Calculate distance between two points."""
        return math.sqrt(
            (float(pt2[0]) - float(pt1[0])) ** 2 + (float(pt2[1]) - float(pt1[1])) ** 2
        )

    def restrict_heading_range(self, h: float) -> float:
        """Normalize heading to [-π, π] range."""
        return (h + math.pi) % (2 * math.pi) - math.pi

    def get_end_effector_offset(self) -> float:
        """Calculate end-effector offset based on task and gripper state."""
        return 0.55

    def dot(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Dot product helper function."""
        return a[0] * b[0] + a[1] * b[1]

    def intersect(
        self,
        d: Tuple[float, float],
        f: Tuple[float, float],
        r: float,
        use_t1: bool = False,
    ) -> Optional[float]:
        """Line-circle intersection helper."""
        a = self.dot(d, d)
        b = 2 * self.dot(f, d)
        c = self.dot(f, f) - r * r
        discriminant = (b * b) - (4 * a * c)
        if discriminant >= 0:
            if use_t1:
                t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t1 <= 1:
                    return t1
            else:
                t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t2 <= 1:
                    return t2
        return None

    def build_base_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build base command using waypoint planning logic."""
        assert command["primitive_name"] in {
            "move", "pick", "place", "toss", "shelf", "drawer"
        }

        # Base movement only
        if command["primitive_name"] == "move":
            return {
                "waypoints": command["waypoints"],
                "target_ee_pos": None,
                "position_tolerance": 0.1,
            }
        
        target_ee_pos = command["waypoints"][-1]
        end_effector_offset = self.get_end_effector_offset()
        new_waypoint = None
        reversed_waypoints = command["waypoints"][::-1]

        for idx in range(1, len(reversed_waypoints)):
            start = reversed_waypoints[idx - 1]
            end = reversed_waypoints[idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - target_ee_pos[0], start[1] - target_ee_pos[1])
            t2 = self.intersect(d, f, end_effector_offset)
            if t2 is not None:
                new_waypoint = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                break

        if new_waypoint is not None:
            # Discard all waypoints that are too close to target_ee_pos
            waypoints = reversed_waypoints[idx:][::-1] + [new_waypoint]
        else:
            # Base is too close to target end effector position and needs to back up
            curr_position = command["waypoints"][0]
            signed_dist = (
                self.distance(curr_position, target_ee_pos) - end_effector_offset
            )
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = self.restrict_heading_range(math.atan2(dy, dx))
            target_position = (
                curr_position[0] + signed_dist * math.cos(target_heading),
                curr_position[1] + signed_dist * math.sin(target_heading),
            )
            waypoints = [curr_position, target_position]

        return {
            "waypoints": waypoints,
            "target_ee_pos": target_ee_pos,
        }


class TidybotStateConverter:
    """Utility class to convert between ObjectCentricState and tidybot observations."""
    
    @staticmethod
    def state_to_obs(state: ObjectCentricState, robot: Object) -> Dict[str, Any]:
        """Convert ObjectCentricState to tidybot observation format."""
        obs = {}
        
        # Robot base pose
        base_x = state.get(robot, "x")
        base_y = state.get(robot, "y")
        base_theta = state.get(robot, "theta")
        obs["base_pose"] = np.array([base_x, base_y, base_theta])
        
        # Robot arm position and orientation
        arm_x = state.get(robot, "arm_x")
        arm_y = state.get(robot, "arm_y")
        arm_z = state.get(robot, "arm_z")
        obs["arm_pos"] = np.array([arm_x, arm_y, arm_z])
        
        # Robot arm quaternion (default to gripper pointing down)
        obs["arm_quat"] = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Gripper position (0.0 = open, 1.0 = closed)
        gripper_pos = state.get(robot, "gripper")
        obs["gripper_pos"] = np.array([gripper_pos])
        
        # Add object positions
        for obj in state.data.keys():
            if "cube" in obj.name.lower():
                obj_x = state.get(obj, "x")
                obj_y = state.get(obj, "y")
                obj_z = state.get(obj, "z")
                obs[f"{obj.name}_pos"] = np.array([obj_x, obj_y, obj_z])
                
                # Default quaternion for objects
                obs[f"{obj.name}_quat"] = np.array([0.0, 0.0, 0.0, 1.0])
        
        return obs
    
    @staticmethod
    def obs_to_state_update(obs: Dict[str, Any], robot: Object) -> Dict[Object, Dict[str, float]]:
        """Convert tidybot observation to state updates."""
        updates = {}
        
        # Update robot state
        robot_updates = {}
        if "base_pose" in obs:
            base_pose = obs["base_pose"]
            robot_updates["x"] = float(base_pose[0])
            robot_updates["y"] = float(base_pose[1])
            robot_updates["theta"] = float(base_pose[2])
        
        if "arm_pos" in obs:
            arm_pos = obs["arm_pos"]
            robot_updates["arm_x"] = float(arm_pos[0])
            robot_updates["arm_y"] = float(arm_pos[1])
            robot_updates["arm_z"] = float(arm_pos[2])
        
        if "gripper_pos" in obs:
            gripper_pos = obs["gripper_pos"]
            robot_updates["gripper"] = float(gripper_pos[0])
        
        updates[robot] = robot_updates
        
        return updates


def create_tidybot_action(
    base_pose: Optional[np.ndarray] = None,
    arm_pos: Optional[np.ndarray] = None,
    arm_quat: Optional[np.ndarray] = None,
    gripper_pos: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Create a tidybot action dictionary with default values."""
    action = {}
    
    if base_pose is not None:
        action["base_pose"] = base_pose
    else:
        action["base_pose"] = np.array([0.0, 0.0, 0.0])
    
    if arm_pos is not None:
        action["arm_pos"] = arm_pos
    else:
        action["arm_pos"] = np.array([0.14, 0.0, 0.21])
    
    if arm_quat is not None:
        action["arm_quat"] = arm_quat
    else:
        action["arm_quat"] = np.array([1.0, 0.0, 0.0, 0.0])
    
    if gripper_pos is not None:
        action["gripper_pos"] = gripper_pos
    else:
        action["gripper_pos"] = np.array([0.0])
    
    return action
