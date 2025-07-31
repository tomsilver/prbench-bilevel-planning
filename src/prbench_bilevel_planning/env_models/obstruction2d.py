"""Bilevel planning models for the obstruction 2D environment."""

from prbench_bilevel_planning.structs import BilevelPlanningEnvModels
from gymnasium.spaces import Space
from relational_structs.spaces import ObjectCentricBoxSpace
from geom2drobotenvs.utils import CRVRobotActionSpace
from geom2drobotenvs.object_types import CRVRobotType, RectangleType
import numpy as np
from numpy.typing import NDArray
from relational_structs import Predicate, GroundAtom, LiftedOperator, LiftedAtom
from bilevel_planning.structs import RelationalAbstractState, RelationalAbstractGoal
import prbench


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
        target = RectangleType("target")
        objects = {robot, target}
        atoms = {GroundAtom(HandEmpty, [robot]), GroundAtom(OnTable, [target])}
        return RelationalAbstractState(atoms, objects)
    
    # Goal abstractor.
    def goal_abstractor(x: NDArray[np.float32]) -> set[GroundAtom]:
        """The goal is always the same in this environment."""
        target = RectangleType("target")
        atoms = {GroundAtom(OnTarget, [target])}
        return RelationalAbstractGoal(atoms, state_abstractor)
    
    # Operators.
    robot = CRVRobotType("?robot")
    block = RectangleType("?block")
    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        add_effects={LiftedAtom(Holding([robot, block]))},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
    )
    PickFromTargetOperator = LiftedOperator(
        "PickFromTarget",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
        add_effects={LiftedAtom(Holding([robot, block]))},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
    )
    PlaceOnTableOperator = LiftedOperator(
        "PlaceOnTable",
        [robot, block],
        preconditions={LiftedAtom(Holding([robot, block]))},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        delete_effects={LiftedAtom(Holding([robot, block]))},
    )
    PlaceOnTargetOperator = LiftedOperator(
        "PlaceOnTarget",
        [robot, block],
        preconditions={LiftedAtom(Holding([robot, block]))},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
        delete_effects={LiftedAtom(Holding([robot, block]))},
    )

    # Controllers.

    # Finalize the skills.


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
