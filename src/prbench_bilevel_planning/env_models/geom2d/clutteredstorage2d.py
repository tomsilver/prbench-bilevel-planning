"""Bilevel planning models for the cluttered storage 2D environment."""

from typing import Sequence

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from geom2drobotenvs.concepts import is_inside_shelf
from geom2drobotenvs.object_types import CRVRobotType
from geom2drobotenvs.structs import (
    SE2Pose,
)
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    run_motion_planning_for_crv_robot,
    get_tool_tip_position
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.clutteredstorage2d import (
    ObjectCentricClutteredStorage2DEnv,
    ShelfType,
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
    observation_space: Space, action_space: Space, num_blocks: int
) -> SesameModels:
    """Create the env models for cluttered storage 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricClutteredStorage2DEnv(num_blocks=num_blocks)
    static_obj_state = sim.initial_constant_state.copy()

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
    types = {CRVRobotType, TargetBlockType, ShelfType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Holding = Predicate("Holding", [CRVRobotType, TargetBlockType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    NotOnShelf = Predicate("NotOnShelf", [TargetBlockType, ShelfType])
    OnShelf = Predicate("OnShelf", [TargetBlockType, ShelfType])
    predicates = {Holding, HandEmpty, NotOnShelf, OnShelf}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        # Get all target blocks in the environment
        target_blocks = x.get_objects(TargetBlockType)
        shelf = x.get_objects(ShelfType)[0]

        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        for block in target_blocks:
            if block in suctioned_objs:
                atoms.add(GroundAtom(Holding, [robot, block]))
        if not suctioned_objs:
            atoms.add(GroundAtom(HandEmpty, [robot]))

        for block in target_blocks:
            if is_inside_shelf(x, block, shelf, {}):
                atoms.add(GroundAtom(OnShelf, [block, shelf]))
            else:
                atoms.add(GroundAtom(NotOnShelf, [block, shelf]))
        objects = {robot, shelf} | set(target_blocks)

        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have all blocks on the shelf."""
        target_blocks = x.get_objects(TargetBlockType)
        shelf = x.get_objects(ShelfType)[0]
        atoms = set()
        for block in target_blocks:
            atoms.add(GroundAtom(OnShelf, [block, shelf]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", TargetBlockType)
    shelf = Variable("?shelf", ShelfType)

    PickBlockNotOnShelfOperator = LiftedOperator(
        "PickBlockNotOnShelf",
        [robot, block, shelf],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(NotOnShelf, [block, shelf]),
        },
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )
    PickBlockOnShelfOperator = LiftedOperator(
        "PickBlockOnShelf",
        [robot, block, shelf],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnShelf, [block, shelf]),
        },
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )
    PlaceBlockNotOnShelfOperator = LiftedOperator(
        "PlaceBlockNotOnShelf",
        [robot, block, shelf],
        preconditions={
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(OnShelf, [block, shelf]),
        },
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(NotOnShelf, [block, shelf]),
        },
        delete_effects={
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(OnShelf, [block, shelf]),
        },
    )
    PlaceBlockOnShelfOperator = LiftedOperator(
        "PlaceBlockOnShelf",
        [robot, block, shelf],
        preconditions={
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(NotOnShelf, [block, shelf]),
        },
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnShelf, [block, shelf]),
        },
        delete_effects={
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(NotOnShelf, [block, shelf]),
        },
    )

    # Controllers.
    class GroundPickBlockController(Geom2dRobotController):
        """Controller for grasping the block that is not on the shelf yet."""

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._block = objects[1]
            self._shelf = objects[2]

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float]:
            # Sample grasp ratio on the height of the block
            # <0.0: custom frame dx < 0
            # >0.0: custom frame dx > 0
            while True:
                grasp_ratio = rng.uniform(-1.0, 1.0)
                if grasp_ratio != 0.0:
                    break
            max_arm_length = x.get(self._robot, "arm_length")
            min_arm_length = (
                x.get(self._robot, "base_radius")
                + x.get(self._robot, "gripper_width") / 2
                + 1e-4
            )
            arm_length = rng.uniform(min_arm_length, max_arm_length)
            return (grasp_ratio, arm_length)

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 1.0

        def _calculate_grasp_robot_pose(self, state: ObjectCentricState) -> SE2Pose:
            """Calculate the actual grasp point based on ratio parameter."""
            if isinstance(self._current_params, tuple):
                grasp_ratio, arm_length = self._current_params
            else:
                raise ValueError(
                    "Expected tuple parameters for grasp ratio and arm length"
                )

            # Get block properties and grasp frame
            block_x = state.get(self._block, "x")
            block_y = state.get(self._block, "y")
            block_theta = state.get(self._block, "theta")
            rel_point_dx = state.get(self._block, "width") / 2
            rel_point = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
                rel_point_dx, 0.0, 0.0
            )

            # Relative SE2 pose w.r.t the grasp frame
            custom_dx = state.get(self._block, "width") / 2 + arm_length + \
                state.get(self._robot, "gripper_width")
            custom_dx *= -1 if grasp_ratio < 0 else 1 # Right or left side grasp
            # Custom dy is always positive.
            custom_dy = abs(grasp_ratio) * state.get(self._block, "height")
            custom_dtheta = 0.0 if grasp_ratio < 0 else np.pi
            custom_pose = SE2Pose(custom_dx, custom_dy, custom_dtheta)

            target_se2_pose = rel_point * custom_pose
            return target_se2_pose

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")

            # Calculate grasp point and robot target position
            target_se2_pose = self._calculate_grasp_robot_pose(state)
            if isinstance(self._current_params, tuple):
                _, desired_arm_length = self._current_params
            else:
                raise ValueError(
                    "Expected tuple parameters for grasp ratio and arm length"
                )

            # Plan collision-free waypoints to the target pose
            # We set the arm to be the shortest length during motion planning
            mp_state = state.copy()
            mp_state.set(self._robot, "arm_joint", robot_radius)
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                mp_state, self._robot, target_se2_pose, action_space
            )
            final_waypoints: list[tuple[SE2Pose, float]] = []
            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                state.get(self._robot, "arm_joint"),
            )

            if collision_free_waypoints is not None:
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
                final_waypoints.append((target_se2_pose, desired_arm_length))
            else:
                # Stay static
                final_waypoints.append(current_wp)

            return final_waypoints

    class GroundPlaceBlockController(Geom2dRobotController):
        """Controller for placing the block on the shelf."""

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._block = objects[1]
            self._shelf = objects[2]

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float]:
            # Sample place ratio
            # w.r.t (shelf_width - block_width)
            # and (shelf_height - block_height)
            full_state = x.copy()
            full_state.data.update(static_obj_state.data)
            relative_dx = rng.uniform(0.01, 0.99)
            # Bias towards inside the shelf
            relative_dy = rng.uniform(0.1, 0.95)
            return (relative_dx, relative_dy)

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 1.0, 0.0

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            robot_theta = state.get(self._robot, "theta")
            robot_radius = state.get(self._robot, "base_radius")
            robot_arm_length = state.get(self._robot, "arm_length")
            gripper_height = state.get(self._robot, "gripper_height")
            gripper_width = state.get(self._robot, "gripper_width")
            block_x = state.get(self._block, "x")
            block_y = state.get(self._block, "y")
            block_theta = state.get(self._block, "theta")
            block_width = state.get(self._block, "width")
            block_height = state.get(self._block, "height")
            block_curr_center = SE2Pose(block_x, block_y, block_theta) * \
                SE2Pose(block_width / 2, block_height / 2, 0.0)
            _, gripper_to_block = get_suctioned_objects(state, self._robot)[0]

            # Calculate pre-place position based on Shelf position
            shelf_x = state.get(self._shelf, "x1")
            shelf_width = state.get(self._shelf, "width1")
            shelf_y = state.get(self._shelf, "y1")
            shelf_height = state.get(self._shelf, "height1")
            assert isinstance(
                self._current_params, tuple
            ), "PlaceBlock expects tuple params"
            # x can be anywhere in the shelf width (no collision)
            x_min = shelf_x + gripper_height / 2
            x_max = shelf_x + shelf_width - gripper_height / 2
            x_min = min(x_min, x_max)
            x_max = max(x_min, x_max)
            block_desired_x_center = (
                x_min + (x_max - x_min) * self._current_params[0]
            )
            # y is confined to inside the shelf height (no collision)
            y_min = min(shelf_y + block_width / 2, shelf_y + shelf_height - block_width / 2)
            y_max = max(shelf_y + block_width / 2, shelf_y + shelf_height - block_width / 2)
            block_desired_y_center = (
                y_min + (y_max - y_min) * self._current_params[1]
            )
            # Note: The desired orientation depends on how is the blocked grasped.
            # If grasping from the left side, the block should be placed with theta = np.pi / 2
            # If grasping from the right side, the block should be placed with theta = -np.pi / 2
            gripper_x, gripper_y = get_tool_tip_position(state, self._robot)
            gripper_frame = SE2Pose(gripper_x, gripper_y, block_theta)
            relative_frame = block_curr_center.inverse * gripper_frame
            if relative_frame.x < 0:
                # Left side grasp
                block_desired_center = SE2Pose(
                    block_desired_x_center, block_desired_y_center, np.pi / 2
                )
            else:
                # Right side grasp
                block_desired_center = SE2Pose(
                    block_desired_x_center, block_desired_y_center, -np.pi / 2
                )
            gripper_final_desired_pose = block_desired_center * \
                SE2Pose(-block_width / 2, -block_height / 2, 0.0) * \
                gripper_to_block.inverse

            final_robot_y = gripper_final_desired_pose.y - robot_arm_length \
                - gripper_width

            pre_place_robot_x = gripper_final_desired_pose.x
            pre_place_robot_y = final_robot_y - shelf_height
            pre_place_pose_0 = SE2Pose(pre_place_robot_x, pre_place_robot_y, np.pi / 2)

            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                state.get(self._robot, "arm_joint"),
            )
            # Plan collision-free waypoints to the target pose
            # We set the arm to be the longest during motion planning
            final_waypoints: list[tuple[SE2Pose, float]] = []
            mp_state = state.copy()
            assert isinstance(action_space, CRVRobotActionSpace)
            collision_free_waypoints_0 = run_motion_planning_for_crv_robot(
                mp_state, self._robot, pre_place_pose_0, action_space
            )
            if collision_free_waypoints_0 is None:
                # Stay static
                return [current_wp]
            for wp in collision_free_waypoints_0:
                final_waypoints.append((wp, robot_radius))

            # Stretch the arm to the desired position
            final_waypoints.append((wp, robot_arm_length))

            mp_state.set(self._robot, "x", pre_place_robot_x)
            mp_state.set(self._robot, "y", pre_place_robot_y)
            mp_state.set(self._robot, "theta", np.pi / 2)
            mp_state.set(self._robot, "arm_joint", robot_arm_length)
            pre_place_pose_1 = SE2Pose(pre_place_robot_x, final_robot_y, np.pi / 2)
            collision_free_waypoints_1 = run_motion_planning_for_crv_robot(
                mp_state, self._robot, pre_place_pose_1, action_space
            )
            if collision_free_waypoints_1 is None:
                # Stay static
                return [current_wp]
            
            for wp in collision_free_waypoints_1:
                final_waypoints.append((wp, robot_arm_length))

            return final_waypoints

    # Lifted controllers.
    PickBlockNotOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPickBlockController,
        )
    )

    PickBlockOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPickBlockController,
        )
    )

    PlaceBlockNotOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPlaceBlockController,
        )
    )

    PlaceBlockOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPlaceBlockController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickBlockNotOnShelfOperator, PickBlockNotOnShelfController),
        LiftedSkill(PickBlockOnShelfOperator, PickBlockOnShelfController),
        LiftedSkill(PlaceBlockNotOnShelfOperator, PlaceBlockNotOnShelfController),
        LiftedSkill(PlaceBlockOnShelfOperator, PlaceBlockOnShelfController),
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
