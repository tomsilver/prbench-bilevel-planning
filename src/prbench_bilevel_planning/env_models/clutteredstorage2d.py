"""Bilevel planning models for the cluttered storage 2D environment."""

import abc
from typing import Sequence

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from geom2drobotenvs.concepts import is_inside
from geom2drobotenvs.object_types import CRVRobotType
from geom2drobotenvs.structs import (
    SE2Pose,
)
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    run_motion_planning_for_crv_robot,
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


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_blocks: int
) -> SesameModels:
    """Create the env models for cluttered storage 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricClutteredStorage2DEnv(num_blocks=num_blocks)

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
            if is_inside(x, block, shelf, {}):
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
    class _CommonGroundController(GroundParameterizedController, abc.ABC):
        """Shared controller code between different actions."""

        def __init__(
            self,
            objects: Sequence[Object],
        ) -> None:
            self._robot = objects[0]
            self._block = objects[1]
            self._shelf = objects[2]
            assert self._robot.is_instance(CRVRobotType)
            super().__init__(objects)
            self._current_params: tuple[float, float] | float = 0.0
            self._current_plan: list[NDArray[np.float32]] | None = None
            self._current_state: ObjectCentricState | None = None
            # Extract max deltas from action space bounds
            assert isinstance(action_space, CRVRobotActionSpace)
            self._max_delta_x = action_space.high[0]
            self._max_delta_y = action_space.high[1]
            self._max_delta_theta = action_space.high[2]
            self._max_delta_arm = action_space.high[3]

        @abc.abstractmethod
        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[SE2Pose, float]]:
            """Generate a waypoint plan with x, y, theta, arm values."""

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
                start_pose = np.array(
                    [start[0].x, start[0].y, start[0].theta, start[1]]
                )
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

        def reset(
            self, x: ObjectCentricState, params: tuple[float, float] | float
        ) -> None:
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
            assert isinstance(action_space, CRVRobotActionSpace)
            plan_suffix: list[NDArray[np.float32]] = [
                # Change the vacuum.
                np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
            ]
            return waypoint_plan + plan_suffix

    class GroundPickBlockNotOnShelfController(_CommonGroundController):
        """Controller for grasping the block that is not on the shelf yet."""

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> tuple[float, float]:
            # Sample grasp ratio on the width of the block
            # <0.0: custom frame dy < 0
            # >0.0: custom frame dy > 0
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

        def reset(
            self, x: ObjectCentricState, params: tuple[float, float] | float
        ) -> None:
            # Override to ensure params are tuples for this controller
            if not isinstance(params, tuple):
                raise ValueError(
                    "PickBlockNotOnShelfController requires tuple parameters"
                )
            super().reset(x, params)

        def terminated(self) -> bool:
            return self._current_plan is not None and len(self._current_plan) == 0

        def step(self) -> NDArray[np.float32]:
            # Always extend the arm first before planning
            assert self._current_state is not None
            if self._current_plan is None:
                self._current_plan = self._generate_plan(self._current_state)
            return self._current_plan.pop(0)

        def observe(self, x: ObjectCentricState) -> None:
            self._current_state = x

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
            rel_point_x = block_x
            rel_point_y = block_y + state.get(self._block, "height") / 2
            rel_point = SE2Pose(rel_point_x, rel_point_y, block_theta)

            # Relative SE2 pose w.r.t the grasp frame
            custom_dx = abs(grasp_ratio) * state.get(self._block, "width")
            custom_dy = state.get(self._block, "height") / 2 + arm_length
            custom_dy *= -1 if grasp_ratio < 0 else 1
            custom_dtheta = np.pi / 2 if grasp_ratio < 0 else -np.pi / 2
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
            robot_gripper_width = state.get(self._robot, "gripper_width")
            safe_y = robot_radius + robot_gripper_width * 2

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
                final_waypoints.append(
                    (SE2Pose(robot_x, safe_y, robot_theta), robot_radius)
                )
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
                final_waypoints.append((target_se2_pose, desired_arm_length))
            else:
                # Stay static
                final_waypoints.append(current_wp)

            return final_waypoints

    class GroundPickBlockOnShelfController(GroundPickBlockNotOnShelfController):
        """Controller for grasping the block that is on the shelf."""

    class GroundPlaceBlockNotOnShelfController(GroundPickBlockNotOnShelfController):
        """Controller for placing the block not on the shelf."""

    class GroundPlaceBlockOnShelfController(GroundPickBlockNotOnShelfController):
        """Controller for placing the block on the shelf."""

    # Lifted controllers.
    PickBlockNotOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPickBlockNotOnShelfController,
        )
    )

    PickBlockOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPickBlockOnShelfController,
        )
    )

    PlaceBlockNotOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPlaceBlockNotOnShelfController,
        )
    )

    PlaceBlockOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block, shelf],
            GroundPlaceBlockOnShelfController,
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


def get_robot_transfer_position(
    block: Object,
    state: ObjectCentricState,
    block_x: float,
    robot_arm_joint: float,
    relative_x_offset: float = 0,
) -> tuple[float, float]:
    """Get the x, y position that the robot should be at to place or grasp the block."""
    robot = state.get_objects(CRVRobotType)[0]
    # In ClutteredStorage2D, objects are placed on the ground (y=0)
    ground = 0.0
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
