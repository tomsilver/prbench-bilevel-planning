"""Bilevel planning models for the obstruction 2D environment."""

from prbench_bilevel_planning.structs import BilevelPlanningEnvModels
from gymnasium.spaces import Space
from relational_structs.spaces import ObjectCentricBoxSpace
from geom2drobotenvs.utils import CRVRobotActionSpace
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.envs.obstruction_2d_env import TargetBlockType, TargetSurfaceType
import numpy as np
from numpy.typing import NDArray
from relational_structs import Predicate, GroundAtom, LiftedOperator, LiftedAtom, Object, ObjectCentricState
from typing import Sequence
from bilevel_planning.structs import RelationalAbstractState, RelationalAbstractGoal, GroundParameterizedController, LiftedParameterizedController, LiftedSkill
import prbench
import abc


def create_bilevel_planning_models(observation_space: Space, action_space: Space,
                                   num_obstructions: int) -> BilevelPlanningEnvModels:
    """Create the env models for obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    # Make a local copy of the environment to use as the "simulator".
    sim = prbench.make(f"prbench/Obstruction2D-o{num_obstructions}-v0")
    assert sim.observation_space == observation_space
    assert sim.action_space == action_space

    # Create the transition function.
    def transition_fn(x: NDArray[np.float32], u: NDArray[np.float32]) -> NDArray[np.float32]:
        """Simulate the action."""
        sim.reset(options={"init_state": x})
        return sim.step(u)[0]
    
    # Types.
    types = {CRVRobotType, RectangleType}

    # Predicates.
    Holding = Predicate("Holding", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTable = Predicate("OnTable", [RectangleType])
    OnTarget = Predicate("OnTarget", [RectangleType])
    predicates = {Holding, HandEmpty, OnTable, OnTarget}

    # State abstractor.
    def state_abstractor(x: NDArray[np.float32]) -> set[GroundAtom]:
        """Get the abstract state for the current state."""
        # TODO
        robot = CRVRobotType("robot")
        target = TargetBlockType("target_block")
        objects = {robot, target}
        atoms = {GroundAtom(HandEmpty, [robot]), GroundAtom(OnTable, [target])}
        return RelationalAbstractState(atoms, objects)
    
    # Goal abstractor.
    def goal_abstractor(x: NDArray[np.float32]) -> set[GroundAtom]:
        """The goal is always the same in this environment."""
        target = TargetBlockType("target_block")
        atoms = {GroundAtom(OnTarget, [target])}
        return RelationalAbstractGoal(atoms, state_abstractor)
    
    # Operators.
    robot = CRVRobotType("?robot")
    block = RectangleType("?block")
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
    
        def __init__(self, objects: Sequence[Object], safe_y: float = 0.9) -> None:
            robot, block = objects
            assert robot.is_instance(CRVRobotType)
            assert block.is_instance(RectangleType)
            self._robot = robot
            self._block = block
            super().__init__(objects)
            self._current_params: float = 0.0  # different meanings for subclasses
            self._current_plan: list[NDArray[np.float32]] | None = None
            self._safe_y = safe_y

        @abc.abstractmethod
        def _generate_waypoints(self, state: ObjectCentricState) -> list[tuple[float, float]]:
            """Generate a waypoint plan."""

        @abc.abstractmethod
        def _get_vacuum_actions(self) -> tuple[float, float]:
            """Get vacuum actions for during and after waypoint movement."""

        def _waypoints_to_plan(self, state: ObjectCentricState, waypoints: list[tuple[float, float]],
                               vacuum_during_plan: float) -> list[NDArray[np.float32]]:
            # TODO
            import ipdb; ipdb.set_trace()

        def reset(self, x: NDArray[np.float32], params: float) -> None:
            self._params = params
            self._current_plan = self._generate_plan(x)

        def terminated(self) -> bool:
            return self._current_plan is not None and len(self._current_plan) == 0

        def step(self) -> NDArray[np.float32]:
            assert self._current_plan
            return self._current_plan.pop(0)

        def observe(self, x: NDArray[np.float32]) -> None:
            pass

        def _generate_plan(self, x: NDArray[np.float32]) -> list[NDArray[np.float32]]:
            state = observation_space.devectorize(x)
            waypoints = self._generate_waypoints(state)
            vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
            plan = self._waypoints_to_plan(state, waypoints, vacuum_during_plan)
            final_action = np.zeros(5, dtype=np.float32)
            final_action[-1] = vacuum_after_plan
            return plan + [final_action]
        
        def _get_transfer_y(self, state: ObjectCentricState) -> float:
            # Assumes the ground is at y=0.0.
            return state.get(self._block, "height") + state.get(self._robot, "arm_length") + state.get(self._robot, "gripper_height")



    class GroundPickController(_CommonGroundController):
        """Controller for picking a block when the robot's hand is free.
        
        This controller uses waypoints rather than doing motion planning. This is just
        because the environment is simple enough where waypoints should always work.

        The parameters for this controller represent the grasp x position RELATIVE to 
        the center of the block.
        """

        def sample_parameters(self, x: NDArray[np.float32], rng: np.random.Generator) -> float:
            state = observation_space.devectorize(x)
            gripper_width = state.get(self._robot, "gripper_width")
            block_width = state.get(self._block, "width")
            radius = (gripper_width + block_width) / 2
            return rng.uniform(-radius, radius)
        
        def _generate_waypoints(self, state: ObjectCentricState) -> list[tuple[float, float]]:
            robot_x = state.get(self._robot, "x")
            target_x = state.get(self._block, "x") + self._params
            target_y = self._get_transfer_y(state)
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


    class GroundPlaceController(_CommonGroundController):
        """Controller for placing a held block.
        
        This controller uses waypoints rather than doing motion planning. This is just
        because the environment is simple enough where waypoints should always work.

        The parameters for this controller represent the ABSOLUTE x position where the
        robot will release the held block.
        """

        def sample_parameters(self, x: NDArray[np.float32], rng: np.random.Generator) -> float:
            world_min_x = 0.0
            world_max_x = 1.0
            return rng.uniform(world_min_x, world_max_x)
        
        def _generate_waypoints(self, state: ObjectCentricState) -> list[tuple[float, float]]:
            robot_x = state.get(self._robot, "x")
            target_x = self._params
            target_y = self._get_transfer_y(state)
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

    PickController = LiftedParameterizedController(
        [robot, block],
        GroundPickController,
    )

    PlaceController = LiftedParameterizedController(
        [robot, block],
        GroundPlaceController,
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickFromTableOperator, PickController),
        LiftedSkill(PickFromTargetOperator, PickController),
        LiftedSkill(PlaceOnTableOperator, PlaceController),
        LiftedSkill(PlaceOnTargetOperator, PlaceController),
    }

    # Finalize the models.
    return BilevelPlanningEnvModels(
        observation_space,
        action_space,
        transition_fn,
        types,
        predicates,
        state_abstractor,
        goal_abstractor,
        skills
    )
