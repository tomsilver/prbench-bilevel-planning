"""Bilevel planning models for the obstruction 2D environment."""

from typing import Sequence

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from geom2drobotenvs.concepts import is_on
from geom2drobotenvs.envs.obstruction_2d_env import TargetBlockType, TargetSurfaceType
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.structs import SE2Pose
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.obstruction2d import G2DOE as ObjectCentricObstruction2DEnv
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
    """Create the env models for obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricObstruction2DEnv(num_obstructions=num_obstructions)

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
    types = {CRVRobotType, RectangleType, TargetBlockType, TargetSurfaceType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Holding = Predicate("Holding", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTable = Predicate("OnTable", [RectangleType])
    OnTarget = Predicate("OnTarget", [RectangleType])
    predicates = {Holding, HandEmpty, OnTable, OnTarget}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = Object("robot", CRVRobotType)
        target = Object("target_block", TargetBlockType)
        target_surface = Object("target_surface", TargetSurfaceType)
        obstructions: set[Object] = set()
        for i in range(num_obstructions):
            obstruction = Object(f"obstruction{i}", RectangleType)
            obstructions.add(obstruction)
        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        for obj in suctioned_objs & (obstructions | {target}):
            atoms.add(GroundAtom(Holding, [robot, obj]))
        if not suctioned_objs:
            atoms.add(GroundAtom(HandEmpty, [robot]))
        # Add "on" atoms.
        for block in obstructions | {target}:
            if is_on(x, block, target_surface, {}):
                atoms.add(GroundAtom(OnTarget, [block]))
            elif block not in suctioned_objs:
                atoms.add(GroundAtom(OnTable, [block]))
        objects = {robot, target, target_surface} | obstructions
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is always the same in this environment."""
        del x  # not needed
        target = Object("target_block", TargetBlockType)
        atoms = {GroundAtom(OnTarget, [target])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
    )
    PickFromTargetOperator = LiftedOperator(
        "PickFromTarget",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
    )
    PlaceOnTableOperator = LiftedOperator(
        "PlaceOnTable",
        [robot, block],
        preconditions={LiftedAtom(Holding, [robot, block])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        delete_effects={LiftedAtom(Holding, [robot, block])},
    )
    PlaceOnTargetOperator = LiftedOperator(
        "PlaceOnTarget",
        [robot, block],
        preconditions={LiftedAtom(Holding, [robot, block])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
        delete_effects={LiftedAtom(Holding, [robot, block])},
    )

    # Controllers.
    class GroundPickController(Geom2dRobotController):
        """Controller for picking a block when the robot's hand is free.

        This controller uses waypoints rather than doing motion planning. This is just
        because the environment is simple enough where waypoints should always work.

        The parameters for this controller represent the grasp x position RELATIVE to
        the center of the block.
        """

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._block = objects[1]
            assert self._block.is_instance(RectangleType)

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            gripper_height = x.get(self._robot, "gripper_height")
            block_width = x.get(self._block, "width")
            params = rng.uniform(-gripper_height / 2, block_width + gripper_height / 2)
            return params

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_x = state.get(self._robot, "x")
            robot_theta = state.get(self._robot, "theta")
            robot_arm_joint = state.get(self._robot, "arm_joint")
            block_x = state.get(self._block, "x")
            if isinstance(self._current_params, (tuple, list)):
                relative_offset = self._current_params[0]
            else:
                relative_offset = self._current_params
            target_x, target_y = get_robot_transfer_position(
                self._block,
                state,
                block_x,
                robot_arm_joint,
                relative_x_offset=relative_offset,
            )
            return [
                # Start by moving to safe height (may already be there).
                (SE2Pose(robot_x, self._safe_y, robot_theta), robot_arm_joint),
                # Move to above the target block, offset by params.
                (SE2Pose(target_x, self._safe_y, robot_theta), robot_arm_joint),
                # Move down to grasp.
                (SE2Pose(target_x, target_y, robot_theta), robot_arm_joint),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 1.0

        def step(self) -> NDArray[np.float32]:
            # Always extend the arm first before planning.
            assert self._current_state is not None
            if self._current_state.get(self._robot, "arm_joint") <= 0.15:
                assert isinstance(action_space, CRVRobotActionSpace)
                return np.array([0, 0, 0, action_space.high[3], 0], dtype=np.float32)
            return super().step()

        def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
            waypoints = self._generate_waypoints(x)
            vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
            waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
            assert isinstance(action_space, CRVRobotActionSpace)
            plan_suffix: list[NDArray[np.float32]] = [
                # Change the vacuum.
                np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
                # Move up slightly to break contact.
                np.array(
                    [0, action_space.high[1], 0, 0, vacuum_after_plan], dtype=np.float32
                ),
            ]
            return waypoint_plan + plan_suffix

    class _GroundPlaceController(Geom2dRobotController):
        """Controller for placing a held block.

        This controller uses waypoints rather than doing motion planning. This is just
        because the environment is simple enough where waypoints should always work.

        The parameters for this controller represent the ABSOLUTE x position where the
        robot will release the held block.
        """

        def __init__(self, objects: Sequence[Object]) -> None:
            assert isinstance(action_space, CRVRobotActionSpace)
            super().__init__(objects, action_space)
            self._block = objects[1]
            assert self._block.is_instance(RectangleType)

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            robot_x = state.get(self._robot, "x")
            robot_theta = state.get(self._robot, "theta")
            robot_arm_joint = state.get(self._robot, "arm_joint")
            if isinstance(self._current_params, (tuple, list)):
                placement_x = self._current_params[0]
            else:
                placement_x = self._current_params
            target_x, target_y = get_robot_transfer_position(
                self._block,
                state,
                placement_x,
                robot_arm_joint,
            )

            return [
                # Start by moving to safe height (may already be there).
                (SE2Pose(robot_x, self._safe_y, robot_theta), robot_arm_joint),
                # Move to above the target position.
                (SE2Pose(target_x, self._safe_y, robot_theta), robot_arm_joint),
                # Move down to place.
                (SE2Pose(target_x, target_y, robot_theta), robot_arm_joint),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 1.0, 0.0

        def step(self) -> NDArray[np.float32]:
            # Always extend the arm first before planning.
            assert self._current_state is not None
            if self._current_state.get(self._robot, "arm_joint") <= 0.15:
                assert isinstance(action_space, CRVRobotActionSpace)
                return np.array([0, 0, 0, action_space.high[3], 0], dtype=np.float32)
            return super().step()

        def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
            waypoints = self._generate_waypoints(x)
            vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
            waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
            assert isinstance(action_space, CRVRobotActionSpace)
            plan_suffix: list[NDArray[np.float32]] = [
                # Change the vacuum.
                np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
                # Move up slightly to break contact.
                np.array(
                    [0, action_space.high[1], 0, 0, vacuum_after_plan], dtype=np.float32
                ),
            ]
            return waypoint_plan + plan_suffix

    class GroundPlaceOnTableController(_GroundPlaceController):
        """Controller for placing a held block on the table."""

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            del x  # unused
            world_min_x = 0.0
            world_max_x = 1.0
            return rng.uniform(world_min_x, world_max_x)

    class GroundPlaceOnTargetController(_GroundPlaceController):
        """Controller for placing a held block on the target."""

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            surface = x.get_objects(TargetSurfaceType)[0]
            target_x = x.get(surface, "x")
            target_width = x.get(surface, "width")
            block_x = x.get(self._block, "x")
            robot_x = x.get(self._robot, "x")
            offset_x = robot_x - block_x  # account for relative grasp
            lower_x = target_x + offset_x
            block_width = x.get(self._block, "width")
            upper_x = lower_x + (target_width - block_width)
            # This can happen if we are placing an obstruction onto the target surface.
            # Obstructions can be larger than the target surface.
            if lower_x > upper_x:
                lower_x, upper_x = upper_x, lower_x
            return rng.uniform(lower_x, upper_x)

    PickController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, block],
        GroundPickController,
    )

    PlaceOnTableController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block],
            GroundPlaceOnTableController,
        )
    )

    PlaceOnTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block],
            GroundPlaceOnTargetController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickFromTableOperator, PickController),
        LiftedSkill(PickFromTargetOperator, PickController),
        LiftedSkill(PlaceOnTableOperator, PlaceOnTableController),
        LiftedSkill(PlaceOnTargetOperator, PlaceOnTargetController),
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


def get_robot_transfer_position(
    block: Object,
    state: ObjectCentricState,
    block_x: float,
    robot_arm_joint: float,
    relative_x_offset: float = 0,
) -> tuple[float, float]:
    """Get the x, y position that the robot should be at to place or grasp the block."""
    robot = state.get_objects(CRVRobotType)[0]
    surface = state.get_objects(TargetSurfaceType)[0]
    ground = state.get(surface, "y") + state.get(surface, "height")
    padding = 1e-4
    x = block_x + relative_x_offset
    y = (
        ground
        + state.get(block, "height")
        + robot_arm_joint
        + state.get(robot, "gripper_width") / 2
        + padding
    )
    return (x, y)
