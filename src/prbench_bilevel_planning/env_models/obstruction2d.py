"""Bilevel planning models for the obstruction 2D environment."""

from prbench_bilevel_planning.structs import BilevelPlanningEnvModels
from gymnasium.spaces import Space
from prpl_utils.spaces import FunctionalSpace
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace
from geom2drobotenvs.utils import CRVRobotActionSpace, get_suctioned_objects
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
from geom2drobotenvs.envs.obstruction_2d_env import TargetBlockType, TargetSurfaceType
from geom2drobotenvs.concepts import is_on
import numpy as np
from numpy.typing import NDArray
from relational_structs import Predicate, GroundAtom, LiftedOperator, LiftedAtom, Object, ObjectCentricState
from typing import Sequence
from bilevel_planning.structs import RelationalAbstractState, RelationalAbstractGoal, GroundParameterizedController, LiftedParameterizedController, LiftedSkill
import prbench
import abc


def create_bilevel_planning_models(observation_space: Space, executable_space: Space,
                                   num_obstructions: int) -> BilevelPlanningEnvModels:
    """Create the env models for obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(executable_space, CRVRobotActionSpace)

    # Make a local copy of the environment to use as the "simulator".
    sim = prbench.make(f"prbench/Obstruction2D-o{num_obstructions}-v0")
    assert sim.observation_space == observation_space
    assert sim.action_space == executable_space
    
    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)
    
    # Convert actions into executable actions. Actions must be hashable.
    def action_to_executable(action: tuple[float, ...]) -> NDArray[np.float32]:
        """Convert actions into executables."""
        return np.array(action, dtype=np.float32)
    
    # Create the transition function.
    def transition_fn(x: ObjectCentricState, u: tuple[float, ...]) -> ObjectCentricState:
        """Simulate the action."""
        sim.reset(options={"init_state": x})
        obs, _, _, _, _ = sim.step(action_to_executable(u))
        return observation_to_state(obs)
    
    # Types.
    types = {CRVRobotType, RectangleType, TargetBlockType, TargetSurfaceType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Create the action space.
    action_space = FunctionalSpace(contains_fn=lambda x: isinstance(x, tuple))  # weak

    # Predicates.
    Holding = Predicate("Holding", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTable = Predicate("OnTable", [RectangleType])
    OnTarget = Predicate("OnTarget", [RectangleType])
    predicates = {Holding, HandEmpty, OnTable, OnTarget}

    # State abstractor.
    static_state_cache = {}  # being extra safe for now
    def state_abstractor(x: ObjectCentricState) -> set[GroundAtom]:
        """Get the abstract state for the current state."""
        robot = CRVRobotType("robot")
        target = TargetBlockType("target_block")
        target_surface = TargetSurfaceType("target_surface")
        obstructions: set[Object] = set()
        for i in range(num_obstructions):
            obstruction = RectangleType(f"obstruction{i}")
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
            if is_on(x, block, target_surface, static_state_cache):
                atoms.add(GroundAtom(OnTarget, [block]))
            elif block not in suctioned_objs:
                atoms.add(GroundAtom(OnTable, [block]))
        atoms = {GroundAtom(HandEmpty, [robot]), GroundAtom(OnTable, [target])}
        objects = {robot, target} | obstructions
        return RelationalAbstractState(atoms, objects)
    
    # Goal abstractor.
    def goal_abstractor(x: ObjectCentricState) -> set[GroundAtom]:
        """The goal is always the same in this environment."""
        del x  # not needed
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
    
        def __init__(self, objects: Sequence[Object], safe_y: float = 0.9, max_delta: float = 0.025) -> None:
            robot, block = objects
            assert robot.is_instance(CRVRobotType)
            assert block.is_instance(RectangleType)
            self._robot = robot
            self._block = block
            super().__init__(objects)
            self._current_params: float = 0.0  # different meanings for subclasses
            self._current_plan: list[NDArray[np.float32]] | None = None
            self._safe_y = safe_y
            self._max_delta = max_delta

        @abc.abstractmethod
        def _generate_waypoints(self, state: ObjectCentricState) -> list[tuple[float, float]]:
            """Generate a waypoint plan."""

        @abc.abstractmethod
        def _get_vacuum_actions(self) -> tuple[float, float]:
            """Get vacuum actions for during and after waypoint movement."""

        def _waypoints_to_plan(self, state: ObjectCentricState, waypoints: list[tuple[float, float]],
                               vacuum_during_plan: float) -> list[NDArray[np.float32]]:
            current_pos = (state.get(self._robot, "x"), state.get(self._robot, "y"))
            waypoints = [current_pos] + waypoints
            plan: list[NDArray[np.float32]] = []
            for start, end in zip(waypoints[:-1], waypoints[1:]):
                if np.allclose(start, end):
                    continue
                total_dx = end[0] - start[0]
                total_dy = end[0] - start[1]
                num_steps = int(max(np.ceil(abs(total_dx) / self._max_delta), np.ceil(abs(total_dy) / self._max_delta)))
                dx = total_dx / num_steps
                dy = total_dy / num_steps
                action = (dx, dy, 0, 0, vacuum_during_plan)
                for _ in range(num_steps):
                    plan.append(action)
            return plan

        def reset(self, x: ObjectCentricState, params: float) -> None:
            self._params = params
            self._current_plan = self._generate_plan(x)

        def terminated(self) -> bool:
            return self._current_plan is not None and len(self._current_plan) == 0

        def step(self) -> NDArray[np.float32]:
            assert self._current_plan
            return self._current_plan.pop(0)

        def observe(self, x: ObjectCentricState) -> None:
            pass

        def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
            waypoints = self._generate_waypoints(x)
            vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
            plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
            final_action = (0, 0, 0, 0, vacuum_after_plan)
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

        def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> float:
            gripper_width = x.get(self._robot, "gripper_width")
            block_width = x.get(self._block, "width")
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


    class _GroundPlaceController(_CommonGroundController):
        """Controller for placing a held block.
        
        This controller uses waypoints rather than doing motion planning. This is just
        because the environment is simple enough where waypoints should always work.

        The parameters for this controller represent the ABSOLUTE x position where the
        robot will release the held block.
        """
        
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
        
    
    class GroundPlaceOnTableController(_GroundPlaceController):
        """Controller for placing a held block on the table."""

        def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> float:
            world_min_x = 0.0
            world_max_x = 1.0
            return rng.uniform(world_min_x, world_max_x)


    class GroundPlaceOnTargetController(_GroundPlaceController):
        """Controller for placing a held block on the target."""

        def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> float:
            target_x = x.get("target_surface", "x")
            target_width = x.get("target_surface", "width")
            return rng.uniform(target_x - target_width / 2, target_x + target_width / 2)


    PickController = LiftedParameterizedController(
        [robot, block],
        GroundPickController,
    )

    PlaceOnTableController = LiftedParameterizedController(
        [robot, block],
        GroundPlaceOnTableController,
    )

    PlaceOnTargetController = LiftedParameterizedController(
        [robot, block],
        GroundPlaceOnTargetController,
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickFromTableOperator, PickController),
        LiftedSkill(PickFromTargetOperator, PickController),
        LiftedSkill(PlaceOnTableOperator, PlaceOnTableController),
        LiftedSkill(PlaceOnTargetOperator, PlaceOnTargetController),
    }

    # Finalize the models.
    return BilevelPlanningEnvModels(
        observation_space,
        executable_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        action_to_executable,
        state_abstractor,
        goal_abstractor,
        skills
    )
