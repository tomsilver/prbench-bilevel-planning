"""Bilevel planning models for the cluttered retrieval 2D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.clutteredretrieval2d import (
    ObjectCentricClutteredRetrieval2DEnv,
    TargetBlockType,
)
from prbench.envs.geom2d.object_types import CRVRobotType, RectangleType
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
)
from prbench_models.geom2d.envs.clutteredretrieval2d.parameterized_skills import (
    GroundPickController,
    GroundPlaceController,
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
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for cluttered retrieval 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=num_obstructions)

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
    HoldingTgt = Predicate("HoldingTgt", [CRVRobotType, TargetBlockType])
    HoldingObstruction = Predicate("HoldingObstruction", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    predicates = {HoldingTgt, HoldingObstruction, HandEmpty}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        target_block = x.get_objects(TargetBlockType)[0]
        obstructions = x.get_objects(RectangleType)
        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        # Check what the robot is holding
        if target_block in suctioned_objs:
            atoms.add(GroundAtom(HoldingTgt, [robot, target_block]))
        else:
            # Check if holding any obstruction
            held_obstruction = None
            for obstruction in obstructions:
                if obstruction in suctioned_objs:
                    held_obstruction = obstruction
                    break

            if held_obstruction is not None:
                atoms.add(GroundAtom(HoldingObstruction, [robot, held_obstruction]))
            else:
                atoms.add(GroundAtom(HandEmpty, [robot]))
        objects = {robot, target_block} | set(obstructions)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to retrieve the target block (hold it)."""
        del x  # not needed
        robot = Object("robot", CRVRobotType)
        target_block = Object("target_block", TargetBlockType)
        atoms = {GroundAtom(HoldingTgt, [robot, target_block])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target_block = Variable("?target_block", TargetBlockType)
    obstruction = Variable("?obstruction", RectangleType)

    PickTgtOperator = LiftedOperator(
        "PickTgt",
        [robot, target_block],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingTgt, [robot, target_block])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PickObstructionOperator = LiftedOperator(
        "PickObstruction",
        [robot, obstruction],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PlaceObstructionOperator = LiftedOperator(
        "PlaceObstruction",
        [robot, obstruction],
        preconditions={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        add_effects={LiftedAtom(HandEmpty, [robot])},
        delete_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
    )

    # Create partial controller classes that include the action_space
    class PickController(GroundPickController):
        """Controller for picking up an object."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    class PlaceController(GroundPlaceController):
        """Controller for placing down an object."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    # Lifted controllers.
    PickTgtController: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block],
        PickController,
    )

    PickObstructionController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstruction],
            PickController,
        )
    )

    PlaceObstructionController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstruction],
            PlaceController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickTgtOperator, PickTgtController),
        LiftedSkill(PickObstructionOperator, PickObstructionController),
        LiftedSkill(PlaceObstructionOperator, PlaceObstructionController),
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
