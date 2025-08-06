"""Bilevel planning models for the obstruction 2D environment."""

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
from geom2drobotenvs.concepts import is_on
from geom2drobotenvs.envs.obstruction_2d_env import TargetBlockType, TargetSurfaceType
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.obstruction2d import G2DOE as ObjectCentricObstruction2DEnv
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
    observation_space: Space, action_space: Space, num_obstructions: int
<<<<<<< HEAD
) -> SesameModels:
||||||| b79da94
    observation_space: Space, executable_space: Space, num_obstructions: int
) -> BilevelPlanningEnvModels:
=======
) -> BilevelPlanningEnvModels:
>>>>>>> a4cacd94554bf9191ccfa8faa4b9d20946acdcea
    """Create the env models for obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    # Make a local copy of the environment to use as the "simulator". Note that we use
    # the object-centric version of the environment because we want access to the reset
    # and step functions in there, which operate over ObjectCentricState, which we use
    # as the state representation for planning.
    sim = ObjectCentricObstruction2DEnv(num_obstructions=num_obstructions)

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # The object-centric states that are passed around in planning do not include the
    # globally constant objects, so we need to create an exemplar state that does
    # include them and then copy in the changing values before calling step().
    exemplar_state = sim.reset()[0]

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate the action."""
        # See note above re: why we can't just sim.reset(options={"init_state": x}).
        state = exemplar_state.copy()
        for obj, feats in x.data.items():
            state.data[obj] = feats
        # Now we can reset().
        sim.reset(options={"init_state": state})
        sim_obs, _, _, _, _ = sim.step(u)

        # Uncomment to debug.
        # import imageio.v2 as iio
        # import time
        # img = sim.render()
        # iio.imsave(f"debug/debug-sim-{int(time.time()*1000.0)}.png", img)

        # Now we need to extract back out the changing objects.
        next_x = x.copy()
        for obj in x:
            next_x.data[obj] = sim_obs.data[obj]
        return next_x

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
    class _CommonGroundController(GroundParameterizedController, abc.ABC):
        """Shared controller code between picking and placing."""

        def __init__(
            self,
            objects: Sequence[Object],
            safe_y: float = 0.8,
            max_delta: float = 0.025,
        ) -> None:
            robot, block = objects
            assert robot.is_instance(CRVRobotType)
            assert block.is_instance(RectangleType)
            self._robot = robot
            self._block = block
            super().__init__(objects)
            self._current_params: float = 0.0  # different meanings for subclasses
            self._current_plan: list[NDArray[np.float32]] | None = None
            self._current_state: ObjectCentricState | None = None
            self._safe_y = safe_y
            self._max_delta = max_delta

        @abc.abstractmethod
        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            """Generate a waypoint plan."""

        @abc.abstractmethod
        def _get_vacuum_actions(self) -> tuple[float, float]:
            """Get vacuum actions for during and after waypoint movement."""

        def _waypoints_to_plan(
            self,
            state: ObjectCentricState,
            waypoints: list[tuple[float, float]],
            vacuum_during_plan: float,
        ) -> list[NDArray[np.float32]]:
            current_pos = (state.get(self._robot, "x"), state.get(self._robot, "y"))
            waypoints = [current_pos] + waypoints
            plan: list[NDArray[np.float32]] = []
            for start, end in zip(waypoints[:-1], waypoints[1:]):
                if np.allclose(start, end):
                    continue
                total_dx = end[0] - start[0]
                total_dy = end[1] - start[1]
                num_steps = int(
                    max(
                        np.ceil(abs(total_dx) / self._max_delta),
                        np.ceil(abs(total_dy) / self._max_delta),
                    )
                )
                dx = total_dx / num_steps
                dy = total_dy / num_steps
                action = np.array([dx, dy, 0, 0, vacuum_during_plan], dtype=np.float32)
                for _ in range(num_steps):
                    plan.append(action)

            return plan

        def reset(self, x: ObjectCentricState, params: float) -> None:
            self._current_params = params
            self._current_plan = None
            self._current_state = x

        def terminated(self) -> bool:
            return self._current_plan is not None and len(self._current_plan) == 0

        def step(self) -> NDArray[np.float32]:
            # Always extend the arm first before planning.
            assert self._current_state is not None
            if self._current_state.get(self._robot, "arm_joint") <= 0.15:
                assert isinstance(action_space, CRVRobotActionSpace)
                return np.array([0, 0, 0, action_space.high[3], 0], dtype=np.float32)
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
                # Move up slightly to break contact.
                np.array(
                    [0, action_space.high[1], 0, 0, vacuum_after_plan], dtype=np.float32
                ),
            ]
            return waypoint_plan + plan_suffix

    class GroundPickController(_CommonGroundController):
        """Controller for picking a block when the robot's hand is free.

        This controller uses waypoints rather than doing motion planning. This is just
        because the environment is simple enough where waypoints should always work.

        The parameters for this controller represent the grasp x position RELATIVE to
        the center of the block.
        """

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            gripper_height = x.get(self._robot, "gripper_height")
            block_width = x.get(self._block, "width")
            params = rng.uniform(-gripper_height / 2, block_width + gripper_height / 2)
            return params

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            robot_x = state.get(self._robot, "x")
            block_x = state.get(self._block, "x")
            robot_arm_joint = state.get(self._robot, "arm_joint")
            target_x, target_y = get_robot_transfer_position(
                self._block,
                state,
                block_x,
                robot_arm_joint,
                relative_x_offset=self._current_params,
            )
            return [
                # Start by moving to safe height (may already be there).
                (robot_x, self._safe_y),
                # Move to above the target block, offset by params.
                (target_x, self._safe_y),
                # Move down to grasp.
                (target_x, target_y),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 0.0, 1.0

    class _GroundPlaceController(_CommonGroundController):
        """Controller for placing a held block.

        This controller uses waypoints rather than doing motion planning. This is just
        because the environment is simple enough where waypoints should always work.

        The parameters for this controller represent the ABSOLUTE x position where the
        robot will release the held block.
        """

        def _generate_waypoints(
            self, state: ObjectCentricState
        ) -> list[tuple[float, float]]:
            robot_x = state.get(self._robot, "x")
            robot_arm_joint = state.get(self._robot, "arm_joint")
            placement_x = self._current_params
            target_x, target_y = get_robot_transfer_position(
                self._block,
                state,
                placement_x,
                robot_arm_joint,
            )

            return [
                # Start by moving to safe height (may already be there).
                (robot_x, self._safe_y),
                # Move to above the target position.
                (target_x, self._safe_y),
                # Move down to place.
                (target_x, target_y),
            ]

        def _get_vacuum_actions(self) -> tuple[float, float]:
            return 1.0, 0.0

    class GroundPlaceOnTableController(_GroundPlaceController):
        """Controller for placing a held block on the table."""

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
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
