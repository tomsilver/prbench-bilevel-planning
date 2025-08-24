"""Tidybot 3D ground scene pick-and-place skills for bilevel planning.

This module implements pick-and-place skills for tidybot in a ground scene environment
using the bilevel planning framework. It provides controllers for picking up objects
from the ground and placing them at target locations.
"""

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState

from .tidybot3d_utils import (
    PickState,
    PlaceState,
    TidybotController,
    TidybotStateConverter,
    create_tidybot_action,
)


class TidybotGroundModel:
    """A ground model that wraps the Tidybot environment and updates state."""

    def __init__(
        self, env: Any, state_converter: TidybotStateConverter, robot: Object
    ) -> None:
        self._env = env
        self._state_converter = state_converter
        self._robot = robot

    def get_observation(self, x: ObjectCentricState) -> Dict[str, Any]:
        """Get the latest observation and update the state converter."""
        obs = self._env.get_observation()
        self._state_converter.update_obs(obs)
        return self._state_converter.state_to_obs(x, self._robot)


class PickController(TidybotController):
    """Controller for picking up objects from the ground.
    
    This controller implements the pick skill by navigating the robot to an object,
    positioning the arm, and grasping the object.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        max_skill_horizon: int = 100,
        ee_offset: float = 0.12,
        custom_grasp: bool = False,
        env: Optional[Any] = None,
    ) -> None:
        super().__init__(objects, max_skill_horizon, ee_offset, custom_grasp, env)
        
        # Override placement offsets for pick skill
        self.PLACEMENT_X_OFFSET = 0.1
        self.PLACEMENT_Y_OFFSET = 0.1
        self.PLACEMENT_Z_OFFSET = 0.2

    def get_end_effector_offset(self) -> float:
        """Calculate end-effector offset based on task and gripper state."""
        return 0.55

    def distance(
        self,
        pt1: Union[Tuple[float, float], List[float], np.ndarray],
        pt2: Union[Tuple[float, float], List[float], np.ndarray],
    ) -> float:
        """Calculate distance between two points."""
        return math.sqrt(
            (float(pt2[0]) - float(pt1[0])) ** 2 + (float(pt2[1]) - float(pt1[1])) ** 2
        )

    def _generate_plan(self, state: ObjectCentricState) -> List[Dict[str, Any]]:
        """Generate a pick plan based on the current state."""
        # This now acts as a state machine, returning one action at a time.
        obs = TidybotStateConverter.state_to_obs(state, self._robot)
        action = self._step_logic(obs)
        if action is None:
            self.episode_ended = True
            return [self._create_hold_action()]
        return [action]

    def _step_logic(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process one step of the pick logic state machine."""
        base_pose = obs["base_pose"]
        arm_pos = obs["arm_pos"]
        arm_quat = obs["arm_quat"]
        gripper_pos = obs["gripper_pos"]

        if self.state == "idle":
            detected_objects = self.detect_objects_from_obs(obs)
            if detected_objects:
                self.object_location = detected_objects[0]
                if (
                    not hasattr(self.object_location, "shape")
                    or len(self.object_location.shape) != 1
                    or self.object_location.shape[0] != 3
                ):
                    return None
                pick_command = {
                    "primitive_name": "pick",
                    "waypoints": [
                        base_pose[:2].tolist(),
                        self.object_location[:2].tolist(),
                    ],
                    "object_3d_pos": self.object_location.copy(),
                }
                base_command = self.build_base_command(pick_command)
                if base_command:
                    self.current_command = pick_command
                    self.base_waypoints = base_command["waypoints"]
                    self.target_ee_pos = base_command["target_ee_pos"]
                    self.current_waypoint_idx = 1
                    self.lookahead_position = None
                    self.state = "moving"
                else:
                    return None
            else:
                return None

        elif self.state == "moving":
            action = self.execute_base_movement(obs)
            if action is None:  # Base movement complete
                print("Base movement complete!")
                # Check if we're close enough for arm manipulation (from mp_policy)
                if self.target_ee_pos is not None:
                    distance_to_target = self.distance(
                        base_pose[:2], self.target_ee_pos
                    )
                    end_effector_offset = self.get_end_effector_offset()
                    diff = abs(end_effector_offset - distance_to_target)
                    print(
                        f"Distance to target EE: {distance_to_target:.3f}, EE offset: {end_effector_offset:.3f}, diff: {diff:.3f}"
                    )
                    if self.current_command["primitive_name"] == "pick":
                        base_tolerance = self.GRASP_BASE_TOLERANCE
                    else:
                        base_tolerance = self.PLACE_BASE_TOLERANCE
                    print(f"Base tolerance for {self.current_command['primitive_name']}: {base_tolerance:.3f}")
                    if diff < base_tolerance:
                        self.state = "manipulating"
                        print("Base reached target, starting arm manipulation")
                    else:
                        print(
                            f"Too far from target end effector position ({(100 * diff):.1f} cm), tolerance: {(100 * base_tolerance):.1f} cm"
                        )
                        # Instead of going to idle immediately, let's try a few more times
                        if not hasattr(self, 'positioning_attempts'):
                            self.positioning_attempts = 0
                        self.positioning_attempts += 1
                        if self.positioning_attempts >= 3:
                            print("Max positioning attempts reached, proceeding with manipulation anyway")
                            self.state = "manipulating"
                            self.positioning_attempts = 0
                        else:
                            print(f"Positioning attempt {self.positioning_attempts}/3, retrying...")
                            self.state = "idle"  # Reset to try again
                else:
                    self.state = "manipulating"  # No target_ee_pos, proceed anyway
            else:
                return action

        elif self.state == "manipulating":
            assert self.current_command is not None
            if self.grasp_state is None:
                self.grasp_state = PickState.APPROACH
                self.grasp_step_count = 0  # Initialize step counter

            target_arm_pos = arm_pos.copy()
            target_arm_quat = arm_quat.copy()
            target_gripper_pos = gripper_pos.copy()

            object_3d_pos = self.current_command["object_3d_pos"]
            global_diff = np.array(
                [
                    object_3d_pos[0] - base_pose[0],
                    object_3d_pos[1] - base_pose[1],
                    object_3d_pos[2]
                    + self.PICK_APPROACH_HEIGHT_OFFSET
                    - self.ROBOT_BASE_HEIGHT,
                ]
            )
            base_angle = base_pose[2]
            cos_angle = math.cos(-base_angle)
            sin_angle = math.sin(-base_angle)
            object_relative_pos = np.array(
                [
                    cos_angle * global_diff[0] - sin_angle * global_diff[1],
                    sin_angle * global_diff[0] + cos_angle * global_diff[1],
                    global_diff[2],
                ]
            )

            if self.grasp_state == PickState.APPROACH:
                target_arm_pos = object_relative_pos
                target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
                target_gripper_pos = np.array([0.0])
                self.grasp_step_count += 1
                # Allow more steps for approach (like the original 10 steps)
                if self.grasp_step_count >= 10 and np.allclose(arm_pos, target_arm_pos, atol=0.05):
                    self.grasp_state = PickState.LOWER
                    self.grasp_step_count = 0
                    print("Transitioning from APPROACH to LOWER")
            elif self.grasp_state == PickState.LOWER:
                target_arm_pos = object_relative_pos.copy()
                target_arm_pos[2] -= self.PICK_LOWER_DIST
                target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
                target_gripper_pos = np.array([0.0])
                self.grasp_step_count += 1
                # Allow more steps for lowering (like the original 8 steps)
                if self.grasp_step_count >= 8 and np.allclose(arm_pos, target_arm_pos, atol=0.03):
                    self.grasp_state = PickState.GRASP
                    self.grasp_step_count = 0
                    print("Transitioning from LOWER to GRASP")
            elif self.grasp_state == PickState.GRASP:
                target_arm_pos = object_relative_pos.copy()
                target_arm_pos[2] -= self.PICK_LOWER_DIST
                target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
                target_gripper_pos = np.array([1.0])
                if not hasattr(self, "grasp_start_time"):
                    self.grasp_start_time = time.time()
                    self.initial_gripper_pos = gripper_pos[0]
                self.grasp_step_count += 1
                
                gripper_closed_enough = gripper_pos[0] > self.GRASP_SUCCESS_THRESHOLD
                gripper_progress = (
                    gripper_pos[0] - self.initial_gripper_pos
                ) > self.GRASP_PROGRESS_THRESHOLD
                grasp_timeout = (time.time() - self.grasp_start_time) > self.GRASP_TIMEOUT_S
                # Allow minimum steps for grasping (like the original 15 steps)
                if self.grasp_step_count >= 15 and (gripper_closed_enough or gripper_progress or grasp_timeout):
                    self.grasp_state = PickState.LIFT
                    self.grasp_step_count = 0
                    delattr(self, "grasp_start_time")
                    delattr(self, "initial_gripper_pos")
                    print("Transitioning from GRASP to LIFT")
            elif self.grasp_state == PickState.LIFT:
                lifted_pos = object_relative_pos.copy()
                lifted_pos[2] += self.PICK_LIFT_DIST - self.PICK_LOWER_DIST
                target_arm_pos = lifted_pos
                target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
                target_gripper_pos = np.array([1.0])
                self.grasp_step_count += 1
                # Allow more steps for lifting (like the original 10 steps)
                if self.grasp_step_count >= 10 and np.allclose(arm_pos, target_arm_pos, atol=0.05):
                    self.episode_ended = True  # Pick complete
                    print("Pick manipulation completed!")

            return create_tidybot_action(
                base_pose=base_pose.copy(),
                arm_pos=target_arm_pos,
                arm_quat=target_arm_quat,
                gripper_pos=target_gripper_pos,
            )
        
        return self._create_hold_action()

    def execute_base_movement(
        self, obs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute base movement following waypoints like BaseController."""
        base_pose = obs["base_pose"]

        if self.current_waypoint_idx >= len(self.base_waypoints):
            return None

        while True:
            if self.current_waypoint_idx >= len(self.base_waypoints):
                self.lookahead_position = None
                break
            start = self.base_waypoints[self.current_waypoint_idx - 1]
            end = self.base_waypoints[self.current_waypoint_idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - base_pose[0], start[1] - base_pose[1])
            t2 = self.intersect(d, f, self.LOOKAHEAD_DISTANCE)
            if t2 is not None:
                self.lookahead_position = [
                    start[0] + t2 * d[0],
                    start[1] + t2 * d[1],
                ]
                break
            if self.current_waypoint_idx == len(self.base_waypoints) - 1:
                self.lookahead_position = None
                break
            self.current_waypoint_idx += 1

        if self.lookahead_position is None:
            target_position = self.base_waypoints[-1]
            position_error = self.distance(base_pose[:2], target_position)
            if position_error < self.POSITION_TOLERANCE:
                return None
        else:
            target_position = self.lookahead_position

        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(dy, dx)
            frac = 1.0
            if self.lookahead_position is not None:
                remaining_path_length = self.LOOKAHEAD_DISTANCE
                curr_waypoint = self.lookahead_position
                for idx in range(
                    self.current_waypoint_idx, len(self.base_waypoints)
                ):
                    next_waypoint = self.base_waypoints[idx]
                    remaining_path_length += self.distance(
                        curr_waypoint, next_waypoint
                    )
                    curr_waypoint = next_waypoint
                frac = math.sqrt(
                    self.LOOKAHEAD_DISTANCE
                    / max(remaining_path_length, self.LOOKAHEAD_DISTANCE)
                )
            heading_diff = self.restrict_heading_range(
                desired_heading - base_pose[2]
            )
            target_heading += frac * heading_diff

        return create_tidybot_action(
            base_pose=np.array(
                [target_position[0], target_position[1], target_heading]
            ),
            arm_pos=obs["arm_pos"].copy(),
            arm_quat=obs["arm_quat"].copy(),
            gripper_pos=obs["gripper_pos"].copy(),
        )


class PlaceController(TidybotController):
    """Controller for placing objects at target locations.
    
    This controller implements the place skill by navigating the robot to a target
    location, positioning the arm, and releasing the object.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        target_x: float = 1.0,
        target_y: float = 0.0,
        max_skill_horizon: int = 100,
        ee_offset: float = 0.12,
        env: Optional[Any] = None,
    ) -> None:
        super().__init__(objects, max_skill_horizon, ee_offset, env=env)
        self._target_x = target_x
        self._target_y = target_y
        
        # Set target location for placement
        self.target_location = np.array([target_x, target_y, 0.02])  # Ground level

    def reset(self, x: ObjectCentricState, params: tuple[float, ...] | float) -> None:
        """Reset and restore target placement location."""
        super().reset(x, params)
        self.target_location = np.array([self._target_x, self._target_y, 0.02])

    def _generate_plan(self, state: ObjectCentricState) -> List[Dict[str, Any]]:
        """Generate a place plan based on the current state."""
        plan = []
        
        # Convert state to observation format
        obs = TidybotStateConverter.state_to_obs(state, self._robot)
        base_pose = obs["base_pose"]
        
        # Create place command
        place_command = {
            "primitive_name": "place",
            "waypoints": [base_pose[:2].tolist(), self.target_location[:2].tolist()],
            "target_3d_pos": self.target_location.copy(),
        }
        
        # Build base movement plan
        base_command = self.build_base_command(place_command)
        if base_command:
            plan.extend(self._generate_base_movement_plan(base_command, obs))
        # Always attempt manipulation even if base movement plan is empty
        plan.extend(self._generate_place_manipulation_plan(obs))
        
        return plan

    def _generate_base_movement_plan(
        self, base_command: Dict[str, Any], obs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate base movement actions to reach the target with interpolation."""
        plan: List[Dict[str, Any]] = []
        base_pose = obs["base_pose"]
        waypoints = base_command["waypoints"]
        target_ee_pos = base_command["target_ee_pos"]
        
        # Interpolate along each waypoint segment
        step_size = 0.05  # meters per step
        current_xy = np.array([base_pose[0], base_pose[1]], dtype=float)
        current_heading = float(base_pose[2])
        
        for waypoint in waypoints[1:]:  # Skip current position
            waypoint_xy = np.array([float(waypoint[0]), float(waypoint[1])], dtype=float)
            segment_vec = waypoint_xy - current_xy
            segment_len = float(np.linalg.norm(segment_vec))
            if segment_len < 1e-6:
                current_xy = waypoint_xy
                continue
            num_steps = max(1, int(segment_len / step_size))
            for i in range(1, num_steps + 1):
                t = i / num_steps
                pos_xy = current_xy + t * segment_vec
                target_heading = current_heading
                if target_ee_pos is not None:
                    dx = float(target_ee_pos[0]) - pos_xy[0]
                    dy = float(target_ee_pos[1]) - pos_xy[1]
                    target_heading = math.atan2(dy, dx)
                action = create_tidybot_action(
                    base_pose=np.array([pos_xy[0], pos_xy[1], target_heading]),
                    arm_pos=obs["arm_pos"],
                    arm_quat=obs["arm_quat"],
                    gripper_pos=np.array([1.0]),  # Keep gripper closed during movement
                )
                plan.append(action)
            current_xy = waypoint_xy
            current_heading = target_heading
        
        return plan

    def _generate_place_manipulation_plan(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate arm manipulation actions for placing."""
        plan = []
        base_pose = obs["base_pose"]
        
        # Calculate target position relative to base
        global_diff = np.array([
            self.target_location[0] - base_pose[0],
            self.target_location[1] - base_pose[1],
            self.target_location[2] + self.PLACE_APPROACH_HEIGHT_OFFSET - self.ROBOT_BASE_HEIGHT,
        ])
        
        # Transform to base's local coordinate frame
        base_angle = base_pose[2]
        cos_angle = math.cos(-base_angle)
        sin_angle = math.sin(-base_angle)
        
        target_relative_pos = np.array([
            cos_angle * global_diff[0] - sin_angle * global_diff[1],
            sin_angle * global_diff[0] + cos_angle * global_diff[1],
            global_diff[2],
        ])
        
        # Calculate lower position (at placement height)
        global_diff_lower = np.array([
            self.target_location[0] - base_pose[0],
            self.target_location[1] - base_pose[1],
            self.target_location[2] - self.ROBOT_BASE_HEIGHT,
        ])
        
        target_relative_pos_lower = np.array([
            cos_angle * global_diff_lower[0] - sin_angle * global_diff_lower[1],
            sin_angle * global_diff_lower[0] + cos_angle * global_diff_lower[1],
            global_diff_lower[2],
        ])
        
        # Step 1: Approach - Position arm above placement location with closed gripper
        approach_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=target_relative_pos,
            arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_pos=np.array([1.0]),  # Closed
        )
        plan.extend([approach_action] * 10)
        
        # Step 2: Lower - Lower arm to placement height
        lower_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=target_relative_pos_lower,
            arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_pos=np.array([1.0]),
        )
        plan.extend([lower_action] * 8)
        
        # Step 3: Release - Open gripper to place object
        release_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=target_relative_pos_lower,
            arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_pos=np.array([0.0]),  # Open
        )
        plan.extend([release_action] * 10)
        
        # Step 4: Home - Move arm to home position
        arm_home_pos = np.array([0.14, 0.0, 0.21])
        arm_home_quat = np.array([0.707, 0.707, 0, 0])
        home_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=arm_home_pos,
            arm_quat=arm_home_quat,
            gripper_pos=np.array([0.0]),
        )
        plan.extend([home_action] * 10)
        
        return plan


class PickAndPlaceController(TidybotController):
    """Combined controller for pick-and-place operations.
    
    This controller implements a complete pick-and-place sequence by first picking
    up an object and then placing it at a target location.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        target_x: float = 1.0,
        target_y: float = 0.0,
        max_skill_horizon: int = 200,
        ee_offset: float = 0.12,
        custom_grasp: bool = False,
        env: Optional[Any] = None,
    ) -> None:
        super().__init__(objects, max_skill_horizon, ee_offset, custom_grasp, env)
        self._target_x = target_x
        self._target_y = target_y
        self._pick_phase = True
        self.grasp_start_time: float = 0.0
        self.initial_gripper_pos: float = 0.0

    def get_end_effector_offset(self) -> float:
        """Calculate end-effector offset based on task and gripper state."""
        return 0.55

    def distance(
        self,
        pt1: Union[Tuple[float, float], List[float], np.ndarray],
        pt2: Union[Tuple[float, float], List[float], np.ndarray],
    ) -> float:
        """Calculate distance between two points."""
        return math.sqrt(
            (float(pt2[0]) - float(pt1[0])) ** 2 + (float(pt2[1]) - float(pt1[1])) ** 2
        )

    def _generate_plan(self, state: ObjectCentricState) -> List[Dict[str, Any]]:
        """Generate a complete pick-and-place plan."""
        # The _step_logic is now called from the TidybotController's step method,
        # which has access to the updated obs. This method is now a thin wrapper.
        return []

    def _step_logic(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process one step of the pick-and-place logic state machine."""
        base_pose = obs["base_pose"]
        arm_pos = obs["arm_pos"]
        arm_quat = obs["arm_quat"]
        gripper_pos = obs["gripper_pos"]

        if self.state == "idle":
            detected_objects = self.detect_objects_from_obs(obs)
            if not detected_objects:
                return None
            
            self.object_location = detected_objects[0]
            if (
                not hasattr(self.object_location, "shape")
                or len(self.object_location.shape) != 1
                or self.object_location.shape[0] != 3
            ):
                return None
            
            self.target_location = np.array(
                [self._target_x, self._target_y, 0.02]
            )
            pick_command: Dict[str, Any] = {
                "primitive_name": "pick",
                "waypoints": [
                    base_pose[:2].tolist(),
                    self.object_location[:2].tolist(),
                ],
                "object_3d_pos": self.object_location.copy(),
            }
            base_command = self.build_base_command(pick_command)
            if base_command:
                self.current_command = pick_command
                self.base_waypoints = base_command["waypoints"]
                self.target_ee_pos = base_command["target_ee_pos"]
                self.current_waypoint_idx = 1
                self.lookahead_position = None
                self.state = "moving"
            else:
                return None

        elif self.state == "moving":
            action = self.execute_base_movement(obs)
            if action is None:  # Base movement complete
                print("Base movement complete!")
                # Check if we're close enough for arm manipulation (from mp_policy)
                if self.target_ee_pos is not None:
                    distance_to_target = self.distance(
                        base_pose[:2], self.target_ee_pos
                    )
                    end_effector_offset = self.get_end_effector_offset()
                    diff = abs(end_effector_offset - distance_to_target)
                    print(
                        f"Distance to target EE: {distance_to_target:.3f}, EE offset: {end_effector_offset:.3f}, diff: {diff:.3f}"
                    )
                    if self.current_command["primitive_name"] == "pick":
                        base_tolerance = self.GRASP_BASE_TOLERANCE
                    else:
                        base_tolerance = self.PLACE_BASE_TOLERANCE
                    print(f"Base tolerance for {self.current_command['primitive_name']}: {base_tolerance:.3f}")
                    if diff < base_tolerance:
                        self.state = "manipulating"
                        print("Base reached target, starting arm manipulation")
                    else:
                        print(
                            f"Too far from target end effector position ({(100 * diff):.1f} cm), tolerance: {(100 * base_tolerance):.1f} cm"
                        )
                        # Instead of going to idle immediately, let's try a few more times
                        if not hasattr(self, 'positioning_attempts'):
                            self.positioning_attempts = 0
                        self.positioning_attempts += 1
                        if self.positioning_attempts >= 3:
                            print("Max positioning attempts reached, proceeding with manipulation anyway")
                            self.state = "manipulating"
                            self.positioning_attempts = 0
                        else:
                            print(f"Positioning attempt {self.positioning_attempts}/3, retrying...")
                            self.state = "idle"  # Reset to try again
                else:
                    self.state = "manipulating"  # No target_ee_pos, proceed anyway
            else:
                return action

        elif self.state == "manipulating":
            assert self.current_command is not None
            if self._pick_phase:  # Pick manipulation
                return self._handle_pick_manipulation(obs)
            else:  # Place manipulation
                return self._handle_place_manipulation(obs)

        return self._create_hold_action()

    def _handle_pick_manipulation(
        self, obs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle the pick manipulation phase."""
        base_pose = obs["base_pose"]
        arm_pos = obs["arm_pos"]
        gripper_pos = obs["gripper_pos"]

        if self.grasp_state is None:
            self.grasp_state = PickState.APPROACH
            self.grasp_step_count = 0  # Initialize step counter

        assert self.current_command is not None
        object_3d_pos = self.current_command["object_3d_pos"]
        global_diff = np.array(
            [
                object_3d_pos[0] - base_pose[0],
                object_3d_pos[1] - base_pose[1],
                object_3d_pos[2]
                + self.PICK_APPROACH_HEIGHT_OFFSET
                - self.ROBOT_BASE_HEIGHT,
            ]
        )
        base_angle = base_pose[2]
        cos_angle = math.cos(-base_angle)
        sin_angle = math.sin(-base_angle)
        object_relative_pos = np.array(
            [
                cos_angle * global_diff[0] - sin_angle * global_diff[1],
                sin_angle * global_diff[0] + cos_angle * global_diff[1],
                global_diff[2],
            ]
        )

        target_arm_pos = arm_pos.copy()
        target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_gripper_pos = gripper_pos.copy()

        if self.grasp_state == PickState.APPROACH:
            target_arm_pos = object_relative_pos
            target_gripper_pos = np.array([0.0])
            self.grasp_step_count += 1
            if self.grasp_step_count >= 10 and np.allclose(arm_pos, target_arm_pos, atol=0.05):
                self.grasp_state = PickState.LOWER
                self.grasp_step_count = 0
                print("Pick: Transitioning from APPROACH to LOWER")
        elif self.grasp_state == PickState.LOWER:
            target_arm_pos = object_relative_pos.copy()
            target_arm_pos[2] -= self.PICK_LOWER_DIST
            target_gripper_pos = np.array([0.0])
            self.grasp_step_count += 1
            if self.grasp_step_count >= 8 and np.allclose(arm_pos, target_arm_pos, atol=0.03):
                self.grasp_state = PickState.GRASP
                self.grasp_step_count = 0
                print("Pick: Transitioning from LOWER to GRASP")
        elif self.grasp_state == PickState.GRASP:
            target_arm_pos = object_relative_pos.copy()
            target_arm_pos[2] -= self.PICK_LOWER_DIST
            target_gripper_pos = np.array([1.0])
            if self.grasp_start_time == 0.0:
                self.grasp_start_time = time.time()
                self.initial_gripper_pos = gripper_pos[0]
            self.grasp_step_count += 1
            
            gripper_closed_enough = gripper_pos[0] > self.GRASP_SUCCESS_THRESHOLD
            gripper_progress = (
                gripper_pos[0] - self.initial_gripper_pos
            ) > self.GRASP_PROGRESS_THRESHOLD
            grasp_timeout = (
                time.time() - self.grasp_start_time
            ) > self.GRASP_TIMEOUT_S
            if self.grasp_step_count >= 15 and (gripper_closed_enough or gripper_progress or grasp_timeout):
                self.grasp_state = PickState.LIFT
                self.grasp_step_count = 0
                self.grasp_start_time = 0.0
                print("Pick: Transitioning from GRASP to LIFT")
        elif self.grasp_state == PickState.LIFT:
            lifted_pos = object_relative_pos.copy()
            lifted_pos[2] += self.PICK_LIFT_DIST - self.PICK_LOWER_DIST
            target_arm_pos = lifted_pos
            target_gripper_pos = np.array([1.0])
            self.grasp_step_count += 1
            if self.grasp_step_count >= 10 and np.allclose(arm_pos, target_arm_pos, atol=0.05):
                # Transition to place
                self._pick_phase = False
                self.grasp_state = None  # Reset for place
                assert self.target_location is not None
                place_command: Dict[str, Any] = {
                    "primitive_name": "place",
                    "waypoints": [
                        base_pose[:2].tolist(),
                        self.target_location[:2].tolist(),
                    ],
                    "target_3d_pos": self.target_location.copy(),
                }
                base_command = self.build_base_command(place_command)
                if base_command:
                    self.current_command = place_command
                    self.base_waypoints = base_command["waypoints"]
                    self.target_ee_pos = base_command["target_ee_pos"]
                    self.current_waypoint_idx = 1
                    self.lookahead_position = None
                    self.state = "moving"
                    print("Pick completed, transitioning to place phase")
                else:
                    return None  # Failed to build command
        
        return create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=target_arm_pos,
            arm_quat=target_arm_quat,
            gripper_pos=target_gripper_pos,
        )
    
    def _handle_place_manipulation(
        self, obs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle the place manipulation phase."""
        base_pose = obs["base_pose"]
        arm_pos = obs["arm_pos"]
        gripper_pos = obs["gripper_pos"]

        if self.grasp_state is None:
            self.grasp_state = PlaceState.APPROACH
            self.grasp_step_count = 0  # Initialize step counter for place
        
        assert self.current_command is not None
        target_3d_pos = self.current_command["target_3d_pos"]
        
        # Approach position
        global_diff = np.array(
            [
                target_3d_pos[0] - base_pose[0],
                target_3d_pos[1] - base_pose[1],
                target_3d_pos[2]
                + self.PLACE_APPROACH_HEIGHT_OFFSET
                - self.ROBOT_BASE_HEIGHT,
            ]
        )
        base_angle = base_pose[2]
        cos_angle = math.cos(-base_angle)
        sin_angle = math.sin(-base_angle)
        target_relative_pos = np.array(
            [
                cos_angle * global_diff[0] - sin_angle * global_diff[1],
                sin_angle * global_diff[0] + cos_angle * global_diff[1],
                global_diff[2],
            ]
        )

        # Lower position
        global_diff_lower = np.array(
            [
                target_3d_pos[0] - base_pose[0],
                target_3d_pos[1] - base_pose[1],
                target_3d_pos[2] - self.ROBOT_BASE_HEIGHT,
            ]
        )
        target_relative_pos_lower = np.array(
            [
                cos_angle * global_diff_lower[0]
                - sin_angle * global_diff_lower[1],
                sin_angle * global_diff_lower[0]
                + cos_angle * global_diff_lower[1],
                global_diff_lower[2],
            ]
        )
        
        arm_home_pos = np.array([0.14, 0.0, 0.21])
        arm_home_quat = np.array([0.707, 0.707, 0, 0])

        target_arm_pos = arm_pos.copy()
        target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_gripper_pos = gripper_pos.copy()

        if self.grasp_state == PlaceState.APPROACH:
            target_arm_pos = target_relative_pos
            target_gripper_pos = np.array([1.0])
            self.grasp_step_count += 1
            if self.grasp_step_count >= 10 and np.allclose(arm_pos, target_arm_pos, atol=0.03):
                self.grasp_state = PlaceState.LOWER
                self.grasp_step_count = 0
                print("Place: Transitioning from APPROACH to LOWER")
        elif self.grasp_state == PlaceState.LOWER:
            target_arm_pos = target_relative_pos_lower
            target_gripper_pos = np.array([1.0])
            self.grasp_step_count += 1
            if self.grasp_step_count >= 8 and np.allclose(arm_pos, target_arm_pos, atol=0.02):
                self.grasp_state = PlaceState.RELEASE
                self.grasp_step_count = 0
                print("Place: Transitioning from LOWER to RELEASE")
        elif self.grasp_state == PlaceState.RELEASE:
            target_arm_pos = target_relative_pos_lower
            target_gripper_pos = np.array([0.0])
            self.grasp_step_count += 1
            if self.grasp_step_count >= 10 and gripper_pos[0] < self.PLACE_SUCCESS_THRESHOLD:
                self.grasp_state = PlaceState.HOME
                self.grasp_step_count = 0
                print("Place: Transitioning from RELEASE to HOME")
        elif self.grasp_state == PlaceState.HOME:
            target_arm_pos = arm_home_pos
            target_arm_quat = arm_home_quat
            target_gripper_pos = np.array([0.0])
            self.grasp_step_count += 1
            if self.grasp_step_count >= 10 and np.allclose(arm_pos, target_arm_pos, atol=0.03):
                self.episode_ended = True
                print("Place manipulation completed!")

        return create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=target_arm_pos,
            arm_quat=target_arm_quat,
            gripper_pos=target_gripper_pos,
        )

    def execute_base_movement(
        self, obs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute base movement following waypoints like BaseController."""
        base_pose = obs["base_pose"]

        if self.current_waypoint_idx >= len(self.base_waypoints):
            return None

        while True:
            if self.current_waypoint_idx >= len(self.base_waypoints):
                self.lookahead_position = None
                break
            start = self.base_waypoints[self.current_waypoint_idx - 1]
            end = self.base_waypoints[self.current_waypoint_idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - base_pose[0], start[1] - base_pose[1])
            t2 = self.intersect(d, f, self.LOOKAHEAD_DISTANCE)
            if t2 is not None:
                self.lookahead_position = [
                    start[0] + t2 * d[0],
                    start[1] + t2 * d[1],
                ]
                break
            if self.current_waypoint_idx == len(self.base_waypoints) - 1:
                self.lookahead_position = None
                break
            self.current_waypoint_idx += 1

        if self.lookahead_position is None:
            target_position = self.base_waypoints[-1]
            position_error = self.distance(base_pose[:2], target_position)
            if position_error < self.POSITION_TOLERANCE:
                return None
        else:
            target_position = self.lookahead_position

        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(dy, dx)
            frac = 1.0
            if self.lookahead_position is not None:
                remaining_path_length = self.LOOKAHEAD_DISTANCE
                curr_waypoint = self.lookahead_position
                for idx in range(
                    self.current_waypoint_idx, len(self.base_waypoints)
                ):
                    next_waypoint = self.base_waypoints[idx]
                    remaining_path_length += self.distance(
                        curr_waypoint, next_waypoint
                    )
                    curr_waypoint = next_waypoint
                frac = math.sqrt(
                    self.LOOKAHEAD_DISTANCE
                    / max(remaining_path_length, self.LOOKAHEAD_DISTANCE)
                )
            heading_diff = self.restrict_heading_range(
                desired_heading - base_pose[2]
            )
            target_heading += frac * heading_diff

        return create_tidybot_action(
            base_pose=np.array(
                [target_position[0], target_position[1], target_heading]
            ),
            arm_pos=obs["arm_pos"].copy(),
            arm_quat=obs["arm_quat"].copy(),
            gripper_pos=obs["gripper_pos"].copy(),
        )


def create_bilevel_planning_models(
    observation_space: Space, executable_space: Space, **kwargs
) -> SesameModels:
    """Create bilevel planning models for tidybot ground scene.
    
    This function creates the necessary models for bilevel planning in the
    tidybot ground scene environment, including pick and place controllers.
    """
    # Create dummy objects for the models (these will be replaced at runtime)
    from relational_structs import Type
    
    robot_type = Type(
        "robot", ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"]
    )
    cube_type = Type("cube", ["x", "y", "z"])
    
    robot = Object("robot", robot_type)
    cube = Object("cube1", cube_type)
    
    objects = [robot, cube]

    # The global state converter is now used, so no need to instantiate here.
    
    # Create type features for state creation
    type_features = {
        robot_type: ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"],
        cube_type: ["x", "y", "z"],
    }

    # Create controllers for different skills
    controllers = {}
    
    # Get the environment from kwargs
    env = kwargs.get("env")
    
    # Pick skill
    controllers["pick"] = lambda: PickController(
        objects, 
        max_skill_horizon=kwargs.get("max_skill_horizon", 100),
        custom_grasp=kwargs.get("custom_grasp", False),
        env=env,
    )
    
    # Place skill with parameterized target location
    controllers["place"] = lambda target_x=1.0, target_y=0.0: PlaceController(
        objects,
        target_x=target_x,
        target_y=target_y,
        max_skill_horizon=kwargs.get("max_skill_horizon", 100),
        env=env,
    )
    
    # Combined pick-and-place skill
    controllers[
        "pick_and_place"
    ] = lambda target_x=1.0, target_y=0.0: PickAndPlaceController(
        objects,
        target_x=target_x,
        target_y=target_y,
        max_skill_horizon=kwargs.get("max_skill_horizon", 200),
        custom_grasp=kwargs.get("custom_grasp", False),
        env=env,
    )
    
    # Create the ground model
    env = kwargs.get("env")
    ground_model = (
        TidybotGroundModel(env, STATE_CONVERTER, robot) if env else None
    )

    # Create SesameModels structure
    # Note: This is a simplified version - in practice, you would need to define
    # the complete abstract model, successor generators, etc.
    models = SesameModels(
        abstract_model=None,  # Would need to define abstract state/action spaces
        abstract_successor_generator=None,  # Would need to define abstract transitions
        ground_controller_generator=controllers,  # Our skill controllers
        ground_model=ground_model,  # Pass the ground model
        ground_successor_generator=None,  # Would need to define ground transitions
    )
    
    return models
