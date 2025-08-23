"""Tidybot 3D ground scene pick-and-place skills for bilevel planning.

This module implements pick-and-place skills for tidybot in a ground scene environment
using the bilevel planning framework. It provides controllers for picking up objects
from the ground and placing them at target locations.
"""

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from bilevel_planning.structs import SesameModels
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
    ) -> None:
        super().__init__(objects, max_skill_horizon, ee_offset, custom_grasp)
        
        # Override placement offsets for pick skill
        self.PLACEMENT_X_OFFSET = 0.1
        self.PLACEMENT_Y_OFFSET = 0.1
        self.PLACEMENT_Z_OFFSET = 0.2

    def _generate_plan(self, state: ObjectCentricState) -> List[Dict[str, Any]]:
        """Generate a pick plan based on the current state."""
        plan = []
        
        # Convert state to observation format
        obs = TidybotStateConverter.state_to_obs(state, self._robot)
        
        # Detect objects from the state
        detected_objects = self.detect_objects_from_state(state)
        
        if not detected_objects:
            # No objects to pick, return empty plan
            return plan
        
        # Select the target object (first detected object)
        self.object_location = detected_objects[0]
        
        # Safety check for object location
        if (
            not hasattr(self.object_location, "shape")
            or len(self.object_location.shape) != 1
            or self.object_location.shape[0] != 3
        ):
            return plan
        
        # Set target location for placing the object (relative to object position)
        if self.target_location is None:
            self.target_location = np.array([
                self.object_location[0] + self.PLACEMENT_X_OFFSET,
                self.object_location[1] + self.PLACEMENT_Y_OFFSET,
                self.object_location[2] + self.PLACEMENT_Z_OFFSET,
            ])
        
        # Generate the pick sequence
        base_pose = obs["base_pose"]
        
        # Create pick command
        pick_command = {
            "primitive_name": "pick",
            "waypoints": [base_pose[:2].tolist(), self.object_location[:2].tolist()],
            "object_3d_pos": self.object_location.copy(),
        }
        
        # Build base movement plan
        base_command = self.build_base_command(pick_command)
        if base_command:
            plan.extend(self._generate_base_movement_plan(base_command, obs))
        # Always attempt manipulation even if base movement plan is empty
        plan.extend(self._generate_pick_manipulation_plan(obs))
        
        return plan

    def _generate_base_movement_plan(
        self, base_command: Dict[str, Any], obs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate base movement actions to reach the object."""
        plan = []
        base_pose = obs["base_pose"]
        waypoints = base_command["waypoints"]
        target_ee_pos = base_command["target_ee_pos"]
        
        # Generate waypoint following actions
        for waypoint in waypoints[1:]:  # Skip current position
            target_heading = base_pose[2]  # Default to current heading
            
            if target_ee_pos is not None:
                # Calculate heading to face target
                dx = target_ee_pos[0] - waypoint[0]
                dy = target_ee_pos[1] - waypoint[1]
                target_heading = math.atan2(dy, dx)
            
            action = create_tidybot_action(
                base_pose=np.array([waypoint[0], waypoint[1], target_heading]),
                arm_pos=obs["arm_pos"],
                arm_quat=obs["arm_quat"],
                gripper_pos=obs["gripper_pos"],
            )
            plan.append(action)
        
        return plan

    def _generate_pick_manipulation_plan(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate arm manipulation actions for picking."""
        plan = []
        base_pose = obs["base_pose"]
        
        # Calculate object position relative to base
        global_diff = np.array([
            self.object_location[0] - base_pose[0],
            self.object_location[1] - base_pose[1],
            self.object_location[2] + self.PICK_APPROACH_HEIGHT_OFFSET - self.ROBOT_BASE_HEIGHT,
        ])
        
        # Transform to base's local coordinate frame
        base_angle = base_pose[2]
        cos_angle = math.cos(-base_angle)
        sin_angle = math.sin(-base_angle)
        
        object_relative_pos = np.array([
            cos_angle * global_diff[0] - sin_angle * global_diff[1],
            sin_angle * global_diff[0] + cos_angle * global_diff[1],
            global_diff[2],
        ])
        
        # Step 1: Approach - Position arm above object with open gripper
        approach_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=object_relative_pos,
            arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # Gripper down
            gripper_pos=np.array([0.0]),  # Open
        )
        plan.extend([approach_action] * 10)  # Multiple steps for smooth movement
        
        # Step 2: Lower - Lower gripper closer to object
        lower_pos = object_relative_pos.copy()
        lower_pos[2] -= self.PICK_LOWER_DIST
        lower_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=lower_pos,
            arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_pos=np.array([0.0]),
        )
        plan.extend([lower_action] * 8)
        
        # Step 3: Grasp - Close gripper
        grasp_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=lower_pos,
            arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_pos=np.array([1.0]),  # Close
        )
        plan.extend([grasp_action] * 15)  # Allow time for grasping
        
        # Step 4: Lift - Lift object
        lift_pos = object_relative_pos.copy()
        lift_pos[2] += (self.PICK_LIFT_DIST - self.PICK_LOWER_DIST)
        lift_action = create_tidybot_action(
            base_pose=base_pose.copy(),
            arm_pos=lift_pos,
            arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_pos=np.array([1.0]),
        )
        plan.extend([lift_action] * 10)
        
        return plan


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
    ) -> None:
        super().__init__(objects, max_skill_horizon, ee_offset)
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
        """Generate base movement actions to reach the target location."""
        plan = []
        base_pose = obs["base_pose"]
        waypoints = base_command["waypoints"]
        target_ee_pos = base_command["target_ee_pos"]
        
        # Generate waypoint following actions
        for waypoint in waypoints[1:]:  # Skip current position
            target_heading = base_pose[2]  # Default to current heading
            
            if target_ee_pos is not None:
                # Calculate heading to face target
                dx = target_ee_pos[0] - waypoint[0]
                dy = target_ee_pos[1] - waypoint[1]
                target_heading = math.atan2(dy, dx)
            
            action = create_tidybot_action(
                base_pose=np.array([waypoint[0], waypoint[1], target_heading]),
                arm_pos=obs["arm_pos"],
                arm_quat=obs["arm_quat"],
                gripper_pos=np.array([1.0]),  # Keep gripper closed during movement
            )
            plan.append(action)
        
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
    ) -> None:
        super().__init__(objects, max_skill_horizon, ee_offset, custom_grasp)
        self._target_x = target_x
        self._target_y = target_y
        self._pick_phase = True  # Start with pick phase

    def _generate_plan(self, state: ObjectCentricState) -> List[Dict[str, Any]]:
        """Generate a complete pick-and-place plan."""
        plan = []
        
        # Convert state to observation format
        obs = TidybotStateConverter.state_to_obs(state, self._robot)
        
        if self._pick_phase:
            # Generate pick plan
            detected_objects = self.detect_objects_from_state(state)
            if detected_objects:
                self.object_location = detected_objects[0]
                plan.extend(self._generate_pick_plan(obs))
                self._pick_phase = False  # Switch to place phase after pick
        else:
            # Generate place plan
            self.target_location = np.array([self._target_x, self._target_y, 0.02])
            plan.extend(self._generate_place_plan(obs))
        
        return plan

    def _generate_pick_plan(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate the pick portion of the plan."""
        pick_controller = PickController(
            [self._robot], max_skill_horizon=100, custom_grasp=self._custom_grasp
        )
        pick_controller.object_location = self.object_location
        return pick_controller._generate_pick_manipulation_plan(obs)

    def _generate_place_plan(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate the place portion of the plan."""
        place_controller = PlaceController(
            [self._robot], target_x=self._target_x, target_y=self._target_y
        )
        place_controller.target_location = self.target_location
        return place_controller._generate_place_manipulation_plan(obs)


def create_bilevel_planning_models(
    observation_space: Space, executable_space: Space, **kwargs
) -> SesameModels:
    """Create bilevel planning models for tidybot ground scene.
    
    This function creates the necessary models for bilevel planning in the
    tidybot ground scene environment, including pick and place controllers.
    """
    # Create dummy objects for the models (these will be replaced at runtime)
    from relational_structs import Type
    
    robot_type = Type("robot", ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"])
    cube_type = Type("cube", ["x", "y", "z"])
    
    robot = Object("robot", robot_type)
    cube = Object("cube1", cube_type)
    
    objects = [robot, cube]
    
    # Create type features for state creation
    type_features = {
        robot_type: ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"],
        cube_type: ["x", "y", "z"],
    }
    
    # Create controllers for different skills
    controllers = {}
    
    # Pick skill
    controllers["pick"] = lambda: PickController(
        objects, 
        max_skill_horizon=kwargs.get("max_skill_horizon", 100),
        custom_grasp=kwargs.get("custom_grasp", False),
    )
    
    # Place skill with parameterized target location
    controllers["place"] = lambda target_x=1.0, target_y=0.0: PlaceController(
        objects,
        target_x=target_x,
        target_y=target_y,
        max_skill_horizon=kwargs.get("max_skill_horizon", 100),
    )
    
    # Combined pick-and-place skill
    controllers["pick_and_place"] = lambda target_x=1.0, target_y=0.0: PickAndPlaceController(
        objects,
        target_x=target_x,
        target_y=target_y,
        max_skill_horizon=kwargs.get("max_skill_horizon", 200),
        custom_grasp=kwargs.get("custom_grasp", False),
    )
    
    # Create SesameModels structure
    # Note: This is a simplified version - in practice, you would need to define
    # the complete abstract model, successor generators, etc.
    models = SesameModels(
        abstract_model=None,  # Would need to define abstract state/action spaces
        abstract_successor_generator=None,  # Would need to define abstract transitions
        ground_controller_generator=controllers,  # Our skill controllers
        ground_model=None,  # Would need to define ground-level dynamics
        ground_successor_generator=None,  # Would need to define ground transitions
    )
    
    return models
