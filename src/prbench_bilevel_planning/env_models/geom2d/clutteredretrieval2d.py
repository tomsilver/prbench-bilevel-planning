"""Bilevel planning models for the cluttered retrieval 2D environment."""

from typing import Sequence

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import SE2Pose
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    run_motion_planning_for_crv_robot,
    state_has_collision,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.clutteredretrieval2d import (
    ClutteredRetrieval2DEnvSpec,
    ObjectCentricClutteredRetrieval2DEnv,
    TargetBlockType,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

from prbench_bilevel_planning.env_models.geom2d.geom2d_utils import (
    Geom2dRobotController,
)


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for cluttered retrieval 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=num_obstructions)
    env_spec = ClutteredRetrieval2DEnvSpec()

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {CRVRobotType, RectangleType, TargetBlockType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Holding = Predicate("Holding", [CRVRobotType, TargetBlockType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    predicates = {Holding, HandEmpty}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = Object("robot", CRVRobotType)
        target_block = Object("target_block", TargetBlockType)
        obstructions: set[Object] = set()
        for i in range(num_obstructions):
            obstruction = Object(f"obstruction{i}", RectangleType)
            obstructions.add(obstruction)
        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        # Only target block can be held (goal is to retrieve it)
        if target_block in suctioned_objs:
            atoms.add(GroundAtom(Holding, [robot, target_block]))
        else:
            atoms.add(GroundAtom(HandEmpty, [robot]))
        objects = {robot, target_block} | obstructions
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to retrieve the target block (hold it)."""
        del x  # not needed
        robot = Object("robot", CRVRobotType)
        target_block = Object("target_block", TargetBlockType)
        atoms = {GroundAtom(Holding, [robot, target_block])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target_block = Variable("?target_block", TargetBlockType)

    PickOperator = LiftedOperator(
        "Pick",
        [robot, target_block],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(Holding, [robot, target_block])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )
    
    PlaceOperator = LiftedOperator(
        "Place",
        [robot, target_block],
        preconditions={LiftedAtom(Holding, [robot, target_block])},
        add_effects={LiftedAtom(HandEmpty, [robot])},
        delete_effects={LiftedAtom(Holding, [robot, target_block])},
    )

    # Controllers.
    class GroundPickController(Geom2dRobotController):
        """Controller for picking the target block.

        The grasping point is sampled on all four sides of the block.
        """

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._target_block = objects[1]
            assert self._target_block.is_instance(TargetBlockType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float]:
            # Sample grasp ratio and side
            # grasp_ratio: determines position along the side ([-1.0, 1.0])
            # side: 0=left, 1=right, 2=top, 3=bottom
            grasp_ratio = rng.uniform(-1.0, 1.0)
            side = rng.choice(4)
            max_arm_length = x.get(self._robot, "arm_length")
            min_arm_length = (
                x.get(self._robot, "base_radius")
                + x.get(self._robot, "gripper_width") / 2
                + 1e-4
            )
            arm_length = rng.uniform(min_arm_length, max_arm_length)
            # Pack parameters: side determines grasp approach, ratio determines position
            return (side + grasp_ratio, arm_length)

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 1.0

        def _calculate_grasp_robot_pose(self, state: ObjectCentricState) -> SE2Pose:
            """Calculate the grasp point based on side and ratio parameters."""
            if isinstance(self._current_params, tuple):
                side_ratio, arm_length = self._current_params
                side = int(side_ratio)
                ratio = side_ratio - side
            else:
                raise ValueError("Expected tuple parameters for side_ratio and arm_length")

            # Get block properties
            block_x = state.get(self._target_block, "x")
            block_y = state.get(self._target_block, "y")
            block_theta = state.get(self._target_block, "theta")
            block_width = state.get(self._target_block, "width")
            block_height = state.get(self._target_block, "height")

            # Calculate reference point and approach direction based on side
            if side == 0:  # left side
                rel_point_dx = -block_width / 2
                rel_point_dy = ratio * block_height / 2
                custom_dx = -(arm_length + state.get(self._robot, "gripper_width"))
                custom_dy = 0.0
                custom_dtheta = 0.0
            elif side == 1:  # right side
                rel_point_dx = block_width / 2
                rel_point_dy = ratio * block_height / 2
                custom_dx = arm_length + state.get(self._robot, "gripper_width")
                custom_dy = 0.0
                custom_dtheta = np.pi
            elif side == 2:  # top side
                rel_point_dx = ratio * block_width / 2
                rel_point_dy = block_height / 2
                custom_dx = 0.0
                custom_dy = arm_length + state.get(self._robot, "gripper_width")
                custom_dtheta = -np.pi / 2
            else:  # bottom side (side == 3)
                rel_point_dx = ratio * block_width / 2
                rel_point_dy = -block_height / 2
                custom_dx = 0.0
                custom_dy = -(arm_length + state.get(self._robot, "gripper_width"))
                custom_dtheta = np.pi / 2

            rel_point = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
                rel_point_dx, rel_point_dy, 0.0
            )
            custom_pose = SE2Pose(custom_dx, custom_dy, custom_dtheta)
            target_se2_pose = rel_point * custom_pose
            return target_se2_pose

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_radius = state.get(self._robot, "base_radius")

            # Calculate grasp point and robot target position
            target_se2_pose = self._calculate_grasp_robot_pose(state)
            if isinstance(self._current_params, tuple):
                _, desired_arm_length = self._current_params
            else:
                raise ValueError("Expected tuple parameters")

            # Plan collision-free waypoints to the target pose
            mp_state = state.copy()
            mp_state.set(self._robot, "arm_joint", robot_radius)
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                mp_state, self._robot, target_se2_pose, action_space
            )
            final_waypoints: list[tuple[SE2Pose, float]] = []

            if collision_free_waypoints is not None:
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
                final_waypoints.append((target_se2_pose, desired_arm_length))
                return final_waypoints
            # If motion planning fails, raise failure
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free path to target."
            )

    class GroundPlaceController(Geom2dRobotController):
        """Controller for placing the target block in a collision-free location."""

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._target_block = objects[1]
            assert self._target_block.is_instance(TargetBlockType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float, float]:
            # Sample collision-free robot pose
            full_state = x.copy()
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                full_state.data.update(init_constant_state.data)
            
            world_x_min = env_spec.world_min_x + env_spec.robot_base_radius
            world_x_max = env_spec.world_max_x - env_spec.robot_base_radius
            world_y_min = env_spec.world_min_y + env_spec.robot_base_radius
            world_y_max = env_spec.world_max_y - env_spec.robot_base_radius
            
            max_attempts = 1000
            for _ in range(max_attempts):
                abs_x = rng.uniform(world_x_min, world_x_max)
                abs_y = rng.uniform(world_y_min, world_y_max)
                abs_theta = rng.uniform(-np.pi, np.pi)
                
                # Test this position
                test_state = full_state.copy()
                test_state.set(self._robot, "x", abs_x)
                test_state.set(self._robot, "y", abs_y)
                test_state.set(self._robot, "theta", abs_theta)
                
                # Check collision
                moving_objects = {self._robot, self._target_block}
                static_objects = set(test_state) - moving_objects
                if not state_has_collision(test_state, moving_objects, static_objects, {}):
                    break
            else:
                # Fallback to current position if no collision-free position found
                abs_x = x.get(self._robot, "x")
                abs_y = x.get(self._robot, "y")
                abs_theta = x.get(self._robot, "theta")
            
            # Return normalized parameters
            rel_x = (abs_x - world_x_min) / (world_x_max - world_x_min)
            rel_y = (abs_y - world_y_min) / (world_y_max - world_y_min)
            rel_theta = (abs_theta + np.pi) / (2 * np.pi)
            
            return (rel_x, rel_y, rel_theta)

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 1.0, 0.0

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")

            # Calculate place position from normalized parameters
            if isinstance(self._current_params, tuple):
                rel_x, rel_y, rel_theta = self._current_params
            else:
                raise ValueError("Expected tuple parameters")

            world_x_min = env_spec.world_min_x + env_spec.robot_base_radius
            world_x_max = env_spec.world_max_x - env_spec.robot_base_radius
            world_y_min = env_spec.world_min_y + env_spec.robot_base_radius
            world_y_max = env_spec.world_max_y - env_spec.robot_base_radius

            final_robot_x = world_x_min + (world_x_max - world_x_min) * rel_x
            final_robot_y = world_y_min + (world_y_max - world_y_min) * rel_y
            final_robot_theta = -np.pi + (2 * np.pi) * rel_theta
            final_robot_pose = SE2Pose(final_robot_x, final_robot_y, final_robot_theta)

            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                robot_radius,
            )

            # Plan collision-free waypoints to the target pose
            final_waypoints: list[tuple[SE2Pose, float]] = [current_wp]
            mp_state = state.copy()
            mp_state.set(self._robot, "arm_joint", robot_radius)
            init_constant_state = sim.initial_constant_state
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                mp_state, self._robot, final_robot_pose, action_space
            )
            if collision_free_waypoints is None:
                # Stay static if no path found
                return final_waypoints
            for wp in collision_free_waypoints:
                final_waypoints.append((wp, robot_radius))

            return final_waypoints

    # Lifted controllers.
    PickController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block],
        GroundPickController,
    )

    PlaceController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block],
        GroundPlaceController,
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickOperator, PickController),
        LiftedSkill(PlaceOperator, PlaceController),
    }

    # Finalize the models.
    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )