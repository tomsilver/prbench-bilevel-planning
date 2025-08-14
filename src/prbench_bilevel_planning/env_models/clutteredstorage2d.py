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
from prbench.envs.clutteredstorage2d import TargetBlockType, ShelfType
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.clutteredstorage2d import ObjectCentricClutteredStorage2DEnv
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

    # Static objects.
    static_object_state = sim.initial_constant_state.copy()

    # Predicates.
    Holding = Predicate("Holding", [CRVRobotType, TargetBlockType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    NotOnShelf = Predicate("NotOnShelf", [TargetBlockType])
    OnShelf = Predicate("OnShelf", [TargetBlockType])
    predicates = {Holding, HandEmpty, NotOnShelf, OnShelf}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        # Get all target blocks in the environment
        target_blocks = x.get_objects(TargetBlockType)
        
        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        for block in target_blocks:
            if block in suctioned_objs:
                atoms.add(GroundAtom(Holding, [robot, block]))
        if not suctioned_objs:
            atoms.add(GroundAtom(HandEmpty, [robot]))
        
        # Add "on" atoms.
        full_state = x.copy()
        full_state.data.update(static_object_state.data)
        shelf_objects = full_state.get_objects(ShelfType)
        if shelf_objects:
            shelf = shelf_objects[0]
            for block in target_blocks:
                if is_inside(full_state, block, shelf, {}):
                    atoms.add(GroundAtom(OnShelf, [block]))
                else:
                    atoms.add(GroundAtom(NotOnShelf, [block]))
            objects = {robot, shelf} | set(target_blocks)
        else:
            # If no shelf found, all blocks are not on shelf
            for block in target_blocks:
                atoms.add(GroundAtom(NotOnShelf, [block]))
            objects = {robot} | set(target_blocks)
        
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have all blocks on the shelf."""
        target_blocks = x.get_objects(TargetBlockType)
        atoms = set()
        for block in target_blocks:
            atoms.add(GroundAtom(OnShelf, [block]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    shelf = Variable("?shelf", ShelfType)
    block = Variable("?block", TargetBlockType)
    
    PickBlockNotOnShelfOperator = LiftedOperator(
        "PickBlockNotOnShelf",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(NotOnShelf, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(NotOnShelf, [block])},
    )
    PickBlockOnShelfOperator = LiftedOperator(
        "PickBlockOnShelf",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnShelf, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnShelf, [block])},
    )
    PlaceBlockNotOnShelfOperator = LiftedOperator(
        "PlaceBlockNotOnShelf",
        [robot, block],
        preconditions={LiftedAtom(Holding, [robot, block])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(NotOnShelf, [block])},
        delete_effects={LiftedAtom(Holding, [robot, block])},
    )
    PlaceBlockOnShelfOperator = LiftedOperator(
        "PlaceBlockOnShelf",
        [robot, block],
        preconditions={LiftedAtom(Holding, [robot, block])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnShelf, [block])},
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
            assert block.is_instance(TargetBlockType)
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

    class GroundPlaceNotOnShelfController(_GroundPlaceController):
        """Controller for placing a held block not on the shelf."""

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            world_min_x = 0.0
            world_max_x = 5.0
            # Avoid placing too close to the shelf
            full_state = x.copy()
            full_state.data.update(static_object_state.data)
            shelf_objects = full_state.get_objects(ShelfType)
            
            if shelf_objects:
                shelf = shelf_objects[0]
                shelf_x = full_state.get(shelf, "x")
                shelf_width = full_state.get(shelf, "width")
                shelf_left = shelf_x
                shelf_right = shelf_x + shelf_width
                
                # Sample from areas not overlapping with shelf
                if shelf_left > world_min_x and shelf_right < world_max_x:
                    # Shelf in middle, choose left or right
                    if rng.random() < 0.5:
                        return rng.uniform(world_min_x, shelf_left - 0.1)
                    else:
                        return rng.uniform(shelf_right + 0.1, world_max_x)
                elif shelf_left <= world_min_x:
                    # Shelf on left, use right side
                    return rng.uniform(shelf_right + 0.1, world_max_x)
                else:
                    # Shelf on right, use left side  
                    return rng.uniform(world_min_x, shelf_left - 0.1)
            else:
                # No shelf found, just use anywhere
                return rng.uniform(world_min_x, world_max_x)

    class GroundPlaceOnShelfController(_GroundPlaceController):
        """Controller for placing a held block on the shelf."""

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> float:
            full_state = x.copy()
            full_state.data.update(static_object_state.data)
            shelf_objects = full_state.get_objects(ShelfType)
            
            if shelf_objects:
                shelf = shelf_objects[0]
                shelf_x = full_state.get(shelf, "x")
                shelf_width = full_state.get(shelf, "width")
                block_width = x.get(self._block, "width")
                robot = x.get_objects(CRVRobotType)[0]
                robot_x = x.get(robot, "x")
                block_x = x.get(self._block, "x")
                offset_x = robot_x - block_x  # account for relative grasp
                lower_x = shelf_x + offset_x
                upper_x = lower_x + (shelf_width - block_width)
                # Ensure bounds are valid
                if lower_x > upper_x:
                    lower_x, upper_x = upper_x, lower_x
                return rng.uniform(lower_x, upper_x)
            else:
                # No shelf found, just return a default position
                return 2.5

    PickController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, block],
        GroundPickController,
    )

    PlaceNotOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block],
            GroundPlaceNotOnShelfController,
        )
    )

    PlaceOnShelfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block],
            GroundPlaceOnShelfController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickBlockNotOnShelfOperator, PickController),
        LiftedSkill(PickBlockOnShelfOperator, PickController),
        LiftedSkill(PlaceBlockNotOnShelfOperator, PlaceNotOnShelfController),
        LiftedSkill(PlaceBlockOnShelfOperator, PlaceOnShelfController),
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